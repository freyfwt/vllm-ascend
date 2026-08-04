#!/usr/bin/env python3
"""Check GMM1 with a monolithic NZ weight versus an NZ tensor list.

The script deliberately tests ``torch_npu.npu_grouped_matmul`` rather than
the fused MoE operators.  It has two modes:

* ``int-w4a8``: INT8 activations and packed INT4 weights.  Intended for A3.
* ``mxfp4-w4a8``: MXFP8 activations and MXFP4 weights.  Intended for A5.

For both modes, the monolithic call uses one 3-D weight and one scale tensor.
The tensor-list call uses one independently converted 2-D FRACTAL_NZ weight
and one scale tensor per expert.  Both calls use the same input and group list.

For ``int-w4a8``, ``--model-path`` switches from synthetic weights to a focused
real-checkpoint flow.  It uses vLLM's safetensors iterator and routed-expert
weight loader, then invokes the repository's real W4A8 post-load processing for
both the monolithic and dynamic-EPLB tensor-list layouts before calling GMM1.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import regex as re
import torch
import torch_npu
from torch import nn
from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.quantization.modelslim_config import (
    AscendModelSlimConfig,
    packed_modules_model_mapping,
)

ACL_FORMAT_FRACTAL_NZ = 29
DEFAULT_GROUP_SIZES = (4, 1, 3, 2)
DEFAULT_REAL_EXPERT_IDS = (0, 1, 2, 3)
KIMI_K3_EXPERT_MODULES = ("experts.0.w1", "experts.0.w3", "experts.0.w2")
MXFP_BLOCK_SIZE = 32
REAL_WEIGHT_SUFFIXES = (
    "weight",
    "weight_scale",
    "weight_offset",
    "scale_bias",
)


@dataclass
class Comparison:
    name: str
    max_abs_error: float
    max_rel_error: float
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("int-w4a8", "mxfp4-w4a8"),
        required=True,
        help="Quantization path to exercise.",
    )
    parser.add_argument("--device", type=int, default=0, help="Logical NPU device index.")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Use a real local ModelSlim checkpoint instead of synthetic INT W4A8 weights.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        help="MoE layer to load; defaults to the first MoE layer in the quant description.",
    )
    parser.add_argument(
        "--expert-ids",
        type=int,
        nargs="+",
        help="Global expert IDs to load; defaults to 0 1 2 3 for real checkpoints.",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument(
        "--group-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_GROUP_SIZES),
        help="Token count for each expert; zero-token experts are allowed.",
    )
    return parser.parse_args()


def npu_format_name(tensor: torch.Tensor) -> str:
    return str(torch_npu.get_npu_format(tensor))


def require_nz(name: str, tensors: list[torch.Tensor]) -> None:
    formats = [npu_format_name(tensor) for tensor in tensors]
    if any(fmt != "FRACTAL_NZ" for fmt in formats):
        raise AssertionError(f"{name} must contain only FRACTAL_NZ tensors, got {formats}")


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_fp32 = actual.detach().cpu().float()
    expected_fp32 = expected.detach().cpu().float()
    abs_error = (actual_fp32 - expected_fp32).abs()
    denominator = expected_fp32.abs().clamp_min(torch.finfo(torch.float32).eps)
    return abs_error.max().item(), (abs_error / denominator).max().item()


def compare(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> Comparison:
    max_abs_error, max_rel_error = error_metrics(actual, expected)
    passed = bool(torch.allclose(actual.detach().cpu().float(), expected.detach().cpu().float(), rtol=rtol, atol=atol))
    return Comparison(
        name=name,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        passed=passed,
    )


def operator_error_summary(error: RuntimeError) -> str:
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return "\n".join(lines[-8:])


def validate_dimensions(group_sizes: list[int], hidden_size: int, output_size: int) -> None:
    if len(group_sizes) < 2:
        raise ValueError("GMM requires at least two experts for this tensor-list check.")
    if len(group_sizes) > 128:
        raise ValueError("torch_npu.npu_grouped_matmul accepts at most 128 weight tensors.")
    if any(size < 0 for size in group_sizes):
        raise ValueError(f"group sizes must be non-negative, got {group_sizes}")
    if sum(group_sizes) == 0:
        raise ValueError("at least one expert must receive a token")
    if hidden_size % MXFP_BLOCK_SIZE != 0:
        raise ValueError(f"hidden size must be divisible by {MXFP_BLOCK_SIZE}, got {hidden_size}")
    if output_size % 16 != 0:
        raise ValueError(f"output size must be divisible by 16 for FRACTAL_NZ, got {output_size}")


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def discover_first_moe_layer(quant_description: dict[str, Any]) -> int:
    pattern = re.compile(r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.\d+\.w1\.weight$")
    layer_indices = {int(match.group(1)) for key in quant_description if (match := pattern.match(key)) is not None}
    if not layer_indices:
        raise ValueError("no Kimi-K3 block_sparse_moe expert weights were found in the quant description")
    return min(layer_indices)


def validate_real_checkpoint_args(
    args: argparse.Namespace,
    quant_description: dict[str, Any],
    num_checkpoint_experts: int,
) -> tuple[int, list[int]]:
    if args.mode != "int-w4a8":
        raise ValueError("--model-path currently supports only --mode int-w4a8")
    if len(args.group_sizes) < 2:
        raise ValueError("GMM requires at least two selected experts for this tensor-list check")
    if any(group_size < 0 for group_size in args.group_sizes) or sum(args.group_sizes) == 0:
        raise ValueError(f"invalid group sizes: {args.group_sizes}")

    layer_index = args.layer_index
    if layer_index is None:
        layer_index = discover_first_moe_layer(quant_description)

    expert_ids = list(args.expert_ids or DEFAULT_REAL_EXPERT_IDS)
    if len(expert_ids) != len(args.group_sizes):
        raise ValueError(
            "--expert-ids and --group-sizes must contain the same number of entries, "
            f"got {len(expert_ids)} and {len(args.group_sizes)}"
        )
    if len(set(expert_ids)) != len(expert_ids):
        raise ValueError(f"expert IDs must be unique, got {expert_ids}")
    if any(expert_id < 0 or expert_id >= num_checkpoint_experts for expert_id in expert_ids):
        raise ValueError(f"expert IDs must be in [0, {num_checkpoint_experts}), got {expert_ids}")
    return layer_index, expert_ids


def ensure_kimi_k3_expert_mapping(model_type: str) -> None:
    expected_mapping = list(KIMI_K3_EXPERT_MODULES)
    model_mapping = packed_modules_model_mapping.setdefault(model_type, {})
    existing_mapping = model_mapping.get("experts")
    if existing_mapping is None:
        model_mapping["experts"] = expected_mapping
    elif existing_mapping != expected_mapping:
        raise ValueError(f"unexpected {model_type} expert mapping: {existing_mapping}; expected {expected_mapping}")


def build_real_checkpoint_config(
    model_path: Path,
    quant_description: dict[str, Any],
    device: torch.device,
) -> tuple[ModelConfig, VllmConfig, AscendModelSlimConfig]:
    # Kimi-K3 is a multimodal wrapper whose text_config uses the upstream
    # KimiLinear implementation.  This focused loader resolves that supported
    # text architecture without constructing the vision tower or full model.
    model_config = ModelConfig(
        model=str(model_path),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        quantization="ascend",
        language_model_only=True,
        hf_overrides={"architectures": ["KimiLinearForCausalLM"]},
    )
    model_type = model_config.hf_config.model_type
    ensure_kimi_k3_expert_mapping(model_type)

    quant_config = AscendModelSlimConfig.from_config(quant_description)
    load_config = LoadConfig(load_format="safetensors", use_tqdm_on_load=False)
    vllm_config = VllmConfig(
        model_config=model_config,
        quant_config=quant_config,
        device_config=DeviceConfig(device=device),
        load_config=load_config,
        additional_config={
            "enable_fused_mc2": 0,
            "eplb_config": {"dynamic_eplb": False},
        },
    )
    # The real W4A8 method treats expert-parallel weights as unsharded and
    # therefore uses tp_size=1 without consulting initialized TP groups.
    vllm_config.parallel_config.enable_expert_parallel = True
    clear_ascend_config()
    init_ascend_config(vllm_config)
    return model_config, vllm_config, quant_config


def build_focused_expert_layer(
    *,
    layer_prefix: str,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    model_config: ModelConfig,
    quant_config: AscendModelSlimConfig,
) -> nn.Module:
    runner = FusedMoE(
        num_experts=num_experts,
        top_k=1,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=model_config.dtype,
        quant_config=quant_config,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        prefix=layer_prefix,
        ckpt_names=("w1", "w2", "w3"),
    )
    return runner


def get_routed_experts(runner: nn.Module) -> nn.Module:
    routed_experts = getattr(runner, "routed_experts", None)
    if routed_experts is None:
        raise TypeError(f"focused FusedMoE runner has no routed_experts module: {type(runner).__name__}")
    return routed_experts


def set_dynamic_eplb_processing(routed_experts: nn.Module, enabled: bool) -> None:
    quant_method = getattr(routed_experts, "quant_method", None)
    scheme = getattr(quant_method, "quant_method", None)
    if scheme is None or not hasattr(scheme, "dynamic_eplb"):
        raise TypeError("focused routed experts did not select the Ascend W4A8 MoE quantization scheme")
    scheme.dynamic_eplb = enabled


def make_target_weight_map(
    layer_prefix: str,
    expert_ids: list[int],
    quant_description: dict[str, Any],
) -> dict[str, str]:
    target_map: dict[str, str] = {}
    missing: list[str] = []
    for local_expert_id, checkpoint_expert_id in enumerate(expert_ids):
        for projection in ("w1", "w2", "w3"):
            for suffix in REAL_WEIGHT_SUFFIXES:
                checkpoint_name = f"{layer_prefix}.{checkpoint_expert_id}.{projection}.{suffix}"
                if checkpoint_name not in quant_description:
                    missing.append(checkpoint_name)
                    continue
                target_map[checkpoint_name] = f"{local_expert_id}.{projection}.{suffix}"
    if missing:
        raise KeyError(f"real checkpoint is missing {len(missing)} required expert tensors: {missing[:4]}")
    return target_map


def load_real_expert_weights(
    *,
    model_config: ModelConfig,
    load_config: LoadConfig,
    monolithic_runner: nn.Module,
    tensor_list_runner: nn.Module,
    target_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    loader = DefaultModelLoader(load_config)
    weights = loader.get_all_weights(model_config, nn.Module())
    monolithic_experts = get_routed_experts(monolithic_runner)
    tensor_list_experts = get_routed_experts(tensor_list_runner)
    loaded: set[str] = set()
    source_metadata: dict[str, dict[str, Any]] = {}
    try:
        for checkpoint_name, tensor in weights:
            local_name = target_map.get(checkpoint_name)
            if local_name is None:
                continue
            loaded.add(checkpoint_name)
            source_metadata[local_name] = {
                "checkpoint_name": checkpoint_name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
            for routed_experts in (monolithic_experts, tensor_list_experts):
                loaded_parameters = list(routed_experts.load_weights(((local_name, tensor),)))
                if not loaded_parameters:
                    raise RuntimeError(f"vLLM routed-expert loader did not accept {checkpoint_name}")
            if len(loaded) == len(target_map):
                break
    finally:
        weights.close()

    missing = sorted(set(target_map) - loaded)
    if missing:
        raise KeyError(f"safetensors loader did not yield {len(missing)} required tensors: {missing[:4]}")
    torch.npu.synchronize()
    return source_metadata


def run_real_int_w4a8(args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.model_path.resolve()
    config_path = model_path / "config.json"
    quant_description_path = model_path / "quant_model_description.json"
    if not config_path.is_file() or not quant_description_path.is_file():
        raise FileNotFoundError(f"{model_path} must contain config.json and quant_model_description.json")

    hf_config = load_json_object(config_path)
    text_config = hf_config.get("text_config")
    if not isinstance(text_config, dict):
        raise TypeError("Kimi-K3 config.json must contain a text_config object")
    quant_description = load_json_object(quant_description_path)
    num_checkpoint_experts = int(text_config["num_experts"])
    layer_index, expert_ids = validate_real_checkpoint_args(
        args,
        quant_description,
        num_checkpoint_experts,
    )
    hidden_size = int(text_config.get("routed_expert_hidden_size", text_config["hidden_size"]))
    intermediate_size = int(text_config["moe_intermediate_size"])
    layer_prefix = f"language_model.model.layers.{layer_index}.block_sparse_moe.experts"
    target_map = make_target_weight_map(layer_prefix, expert_ids, quant_description)
    device = torch.device(f"npu:{args.device}")
    model_config, vllm_config, quant_config = build_real_checkpoint_config(
        model_path,
        quant_description,
        device,
    )

    with set_current_vllm_config(vllm_config), torch.device(device):
        common_layer_args = {
            "layer_prefix": layer_prefix,
            "num_experts": len(expert_ids),
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "model_config": model_config,
            "quant_config": quant_config,
        }
        monolithic_runner = build_focused_expert_layer(**common_layer_args)
        tensor_list_runner = build_focused_expert_layer(**common_layer_args)
        monolithic_experts = get_routed_experts(monolithic_runner)
        tensor_list_experts = get_routed_experts(tensor_list_runner)
        set_dynamic_eplb_processing(monolithic_experts, False)
        set_dynamic_eplb_processing(tensor_list_experts, True)

        source_metadata = load_real_expert_weights(
            model_config=model_config,
            load_config=vllm_config.load_config,
            monolithic_runner=monolithic_runner,
            tensor_list_runner=tensor_list_runner,
            target_map=target_map,
        )
        process_weights_after_loading(monolithic_runner, model_config, device)
        process_weights_after_loading(tensor_list_runner, model_config, device)

        monolithic_weight = monolithic_experts.w13_weight
        monolithic_scale = monolithic_experts.w13_weight_scale
        monolithic_bias = monolithic_experts.w13_scale_bias
        weight_list = [weight.view(torch.int32) for weight in tensor_list_experts.w13_weight_list]
        scale_list = tensor_list_experts.w13_weight_scale_list
        bias_list = tensor_list_experts.w13_scale_bias_list
        require_nz("real monolithic INT weight", [monolithic_weight])
        require_nz("real split INT weight", weight_list)

        torch.manual_seed(args.seed)
        num_tokens = sum(args.group_sizes)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) * 0.25
        quantized_hidden_states, per_token_scale = torch_npu.npu_dynamic_quant(
            hidden_states,
            dst_type=torch.int8,
        )
        group_list = torch.tensor(args.group_sizes, dtype=torch.int64, device=device)
        common_gmm_kwargs = {
            "x": [quantized_hidden_states],
            "per_token_scale": [per_token_scale],
            "split_item": 2,
            "group_type": 0,
            "group_list_type": 1,
            "group_list": group_list,
            "output_dtype": torch.bfloat16,
        }
        baseline = torch_npu.npu_grouped_matmul(
            weight=[monolithic_weight],
            scale=[monolithic_scale],
            bias=[monolithic_bias],
            **common_gmm_kwargs,
        )[0]

        result: dict[str, Any] = {
            "mode": "int-w4a8-real-checkpoint",
            "model_path": str(model_path),
            "layer_index": layer_index,
            "checkpoint_expert_ids": expert_ids,
            "group_sizes": args.group_sizes,
            "loading_flow": [
                "DefaultModelLoader.get_all_weights",
                "RoutedExperts.load_weights",
                "process_weights_after_loading",
                "torch_npu.npu_dynamic_quant",
                "torch_npu.npu_grouped_matmul",
            ],
            "source_tensors": source_metadata,
            "tensor_list_supported": False,
            "dtypes": {
                "x_before_quant": str(hidden_states.dtype),
                "x": str(quantized_hidden_states.dtype),
                "per_token_scale": str(per_token_scale.dtype),
                "monolithic_weight": str(monolithic_weight.dtype),
                "split_weight": [str(weight.dtype) for weight in weight_list],
                "monolithic_scale": str(monolithic_scale.dtype),
                "split_scale": [str(scale.dtype) for scale in scale_list],
                "monolithic_bias": str(monolithic_bias.dtype),
                "split_bias": [str(bias.dtype) for bias in bias_list],
            },
            "shapes": {
                "x": list(quantized_hidden_states.shape),
                "monolithic_weight": list(monolithic_weight.shape),
                "split_weight": [list(weight.shape) for weight in weight_list],
                "monolithic_scale": list(monolithic_scale.shape),
                "split_scale": [list(scale.shape) for scale in scale_list],
                "monolithic_bias": list(monolithic_bias.shape),
                "split_bias": [list(bias.shape) for bias in bias_list],
                "output": list(baseline.shape),
            },
            "formats": {
                "monolithic_weight": npu_format_name(monolithic_weight),
                "split_weight": [npu_format_name(weight) for weight in weight_list],
            },
            "comparisons": [],
        }
        try:
            tensor_list = torch_npu.npu_grouped_matmul(
                weight=weight_list,
                scale=scale_list,
                bias=bias_list,
                **common_gmm_kwargs,
            )[0]
        except RuntimeError as error:
            result["tensor_list_error"] = operator_error_summary(error)
            return result

        result["tensor_list_supported"] = True
        comparison = compare(
            "real tensor-list vs real monolithic",
            tensor_list,
            baseline,
            rtol=0.0,
            atol=0.0,
        )
        result["comparisons"] = [asdict(comparison)]
        return result


def pack_nonnegative_int4(weight: torch.Tensor) -> torch.Tensor:
    """Pack pairs of values in [0, 7] into INT8 storage."""
    if weight.device.type != "cpu" or weight.dtype != torch.int8:
        raise TypeError("INT4 packing expects a CPU torch.int8 tensor")
    if weight.shape[-1] % 2:
        raise ValueError("the output dimension must be even before INT4 packing")
    if weight.min().item() < 0 or weight.max().item() > 7:
        raise ValueError("this diagnostic uses non-negative INT4 values in [0, 7]")
    pairs = weight.reshape(*weight.shape[:-1], weight.shape[-1] // 2, 2)
    low = pairs[..., 0] & 0x0F
    high = (pairs[..., 1] & 0x0F) << 4
    return (low | high).contiguous()


def encode_int_scale(scale: torch.Tensor) -> torch.Tensor:
    """Encode FP32 dequant scales in the INT64 layout used by INT W4A8."""
    scale_np = scale.contiguous().numpy().astype(np.float32, copy=False)
    scale_bits = scale_np.view(np.uint32).astype(np.int64)
    return torch.from_numpy(scale_bits)


def int_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    per_token_scale: torch.Tensor,
    group_sizes: list[int],
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    start = 0
    for expert_id, group_size in enumerate(group_sizes):
        end = start + group_size
        if group_size:
            accumulator = x[start:end].float() @ weight[expert_id].float()
            outputs.append(
                accumulator * weight_scale[expert_id].float() * per_token_scale[start:end].float().unsqueeze(1)
            )
        start = end
    return torch.cat(outputs, dim=0)


def run_int_w4a8(args: argparse.Namespace) -> dict[str, Any]:
    group_sizes = args.group_sizes
    num_experts = len(group_sizes)
    num_tokens = sum(group_sizes)
    hidden_size = args.hidden_size
    output_size = args.output_size

    generator = torch.Generator().manual_seed(args.seed)
    x_cpu = torch.randint(-16, 17, (num_tokens, hidden_size), dtype=torch.int8, generator=generator)
    logical_weight_cpu = torch.randint(
        0,
        8,
        (num_experts, hidden_size, output_size),
        dtype=torch.int8,
        generator=generator,
    )
    packed_weight_cpu = pack_nonnegative_int4(logical_weight_cpu)

    # Non-trivial scales make the test verify that scale and weight lists stay
    # aligned.  INT W4A8 represents each FP32 scale through its uint32 bits in
    # an INT64 element.
    weight_scale_cpu = (
        torch.arange(1, num_experts + 1, dtype=torch.float32).unsqueeze(1).expand(-1, output_size) / 512.0
    ).contiguous()
    # Keep the quantization-group dimension explicit.  For packed INT4, GMM
    # interprets [E, N] as N quantization groups; [E, 1, N] is the per-channel
    # representation (and each tensor-list element therefore uses [1, N]).
    encoded_scale_cpu = encode_int_scale(weight_scale_cpu).unsqueeze(1)
    per_token_scale_cpu = torch.linspace(1.0 / 128.0, 1.0 / 64.0, num_tokens, dtype=torch.float32)

    x_npu = x_cpu.npu()
    per_token_scale_npu = per_token_scale_cpu.npu()
    group_list_npu = torch.tensor(group_sizes, dtype=torch.int64, device="npu")

    # Baseline: one 3-D NZ weight containing all experts.
    monolithic_nz_int8 = torch_npu.npu_format_cast(packed_weight_cpu.npu(), ACL_FORMAT_FRACTAL_NZ)
    monolithic_weight = monolithic_nz_int8.view(torch.int32).contiguous()
    monolithic_scale = encoded_scale_cpu.npu()
    require_nz("monolithic INT weight", [monolithic_weight])

    # Experiment: cast each expert independently.  Unbinding an already-NZ
    # 3-D tensor and cloning it can materialize ND tensors, so split before the
    # format cast to guarantee that every list element is genuinely NZ.
    weight_list = [
        torch_npu.npu_format_cast(expert_weight.npu(), ACL_FORMAT_FRACTAL_NZ).view(torch.int32).contiguous()
        for expert_weight in packed_weight_cpu.unbind(dim=0)
    ]
    scale_list = [expert_scale.npu() for expert_scale in encoded_scale_cpu.unbind(dim=0)]
    require_nz("split INT weight", weight_list)

    common_kwargs = {
        "x": [x_npu],
        "per_token_scale": [per_token_scale_npu],
        "split_item": 3,
        "group_type": 0,
        "group_list_type": 1,
        "group_list": group_list_npu,
        "output_dtype": torch.bfloat16,
    }
    baseline = torch_npu.npu_grouped_matmul(
        weight=[monolithic_weight],
        scale=[monolithic_scale],
        **common_kwargs,
    )[0]
    reference = int_reference(
        x_cpu,
        logical_weight_cpu,
        weight_scale_cpu,
        per_token_scale_cpu,
        group_sizes,
    )
    comparisons = [
        compare("monolithic vs torch reference", baseline, reference, rtol=2e-2, atol=5e-2),
    ]
    result: dict[str, Any] = {
        "mode": args.mode,
        "tensor_list_supported": False,
        "shapes": {
            "x": list(x_npu.shape),
            "monolithic_weight": list(monolithic_weight.shape),
            "split_weight": [list(weight.shape) for weight in weight_list],
            "monolithic_scale": list(monolithic_scale.shape),
            "split_scale": [list(scale.shape) for scale in scale_list],
            "output": list(baseline.shape),
        },
        "formats": {
            "monolithic_weight": npu_format_name(monolithic_weight),
            "split_weight": [npu_format_name(weight) for weight in weight_list],
        },
        "comparisons": [asdict(result) for result in comparisons],
    }
    try:
        tensor_list = torch_npu.npu_grouped_matmul(
            weight=weight_list,
            scale=scale_list,
            **common_kwargs,
        )[0]
    except RuntimeError as error:
        result["tensor_list_error"] = operator_error_summary(error)
        return result

    result["tensor_list_supported"] = True
    comparisons.extend(
        [
            compare("tensor-list vs monolithic", tensor_list, baseline, rtol=0.0, atol=0.0),
            compare("tensor-list vs torch reference", tensor_list, reference, rtol=2e-2, atol=5e-2),
        ]
    )
    result["comparisons"] = [asdict(comparison) for comparison in comparisons]
    return result


def process_mxfp_scale(scale: torch.Tensor) -> torch.Tensor:
    """Mirror W4A8 MXFP4 MoE scale processing in the quantization method."""
    num_experts, output_size, num_blocks = scale.shape
    if num_blocks % 2:
        raise ValueError(f"MXFP scale block count must be even, got {num_blocks}")
    return scale.reshape(num_experts, output_size, num_blocks // 2, 2).transpose(-3, -2)


def run_mxfp4_w4a8(args: argparse.Namespace) -> dict[str, Any]:
    required_symbols = ("float4_e2m1fn_x2", "float8_e8m0fnu", "npu_dynamic_mx_quant")
    missing = [symbol for symbol in required_symbols if not hasattr(torch_npu, symbol)]
    if missing:
        raise RuntimeError(f"installed torch_npu lacks required MXFP symbols: {missing}")

    group_sizes = args.group_sizes
    num_experts = len(group_sizes)
    num_tokens = sum(group_sizes)
    hidden_size = args.hidden_size
    output_size = args.output_size

    torch.manual_seed(args.seed)
    x_cpu = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16) * 0.25
    # npu_dynamic_mx_quant quantizes the last dimension, so use [E, N, K]
    # here and transpose only after creating the packed MXFP4 representation.
    weight_cpu = torch.randn(num_experts, output_size, hidden_size, dtype=torch.bfloat16) * 0.125

    x_quantized, per_token_scale = torch_npu.npu_dynamic_mx_quant(
        x_cpu.npu(),
        dst_type=torch.float8_e4m3fn,
        block_size=MXFP_BLOCK_SIZE,
    )
    packed_weight, raw_weight_scale = torch_npu.npu_dynamic_mx_quant(
        weight_cpu.npu(),
        dst_type=torch_npu.float4_e2m1fn_x2,
        round_mode="round",
        block_size=MXFP_BLOCK_SIZE,
    )
    processed_scale = process_mxfp_scale(raw_weight_scale)
    group_list_npu = torch.tensor(group_sizes, dtype=torch.int64, device="npu")

    # Baseline: the same layout transformation used by W4A8 MXFP4 MoE weight
    # loading, retaining all experts in one 3-D NZ tensor.
    monolithic_weight = torch_npu.npu_format_cast(
        packed_weight,
        ACL_FORMAT_FRACTAL_NZ,
        customize_dtype=torch.float8_e4m3fn,
        input_dtype=torch_npu.float4_e2m1fn_x2,
    ).transpose(1, 2)
    monolithic_scale = processed_scale
    require_nz("monolithic MXFP4 weight", [monolithic_weight])

    # Experiment: convert each expert separately so every element in the
    # weight list owns a real FRACTAL_NZ allocation.
    weight_list = [
        torch_npu.npu_format_cast(
            expert_weight,
            ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        ).transpose(0, 1)
        for expert_weight in packed_weight.unbind(dim=0)
    ]
    scale_list = list(processed_scale.unbind(dim=0))
    require_nz("split MXFP4 weight", weight_list)

    common_kwargs = {
        "x": [x_quantized],
        "per_token_scale": [per_token_scale],
        "split_item": 2,
        "group_type": 0,
        "group_list_type": 1,
        "group_list": group_list_npu,
        "output_dtype": torch.bfloat16,
        "x_dtype": torch.float8_e4m3fn,
        "weight_dtype": torch_npu.float4_e2m1fn_x2,
        "per_token_scale_dtype": torch_npu.float8_e8m0fnu,
    }
    baseline = torch_npu.npu_grouped_matmul(
        weight=[monolithic_weight],
        antiquant_scale=[monolithic_scale],
        **common_kwargs,
    )[0]

    reference_parts: list[torch.Tensor] = []
    start = 0
    for expert_id, group_size in enumerate(group_sizes):
        end = start + group_size
        if group_size:
            reference_parts.append(x_cpu[start:end].float() @ weight_cpu[expert_id].float().transpose(0, 1))
        start = end
    reference = torch.cat(reference_parts, dim=0)
    comparisons = [
        # This reference starts from the pre-quantized BF16 tensors, so allow
        # normal MXFP4/MXFP8 quantization error while still catching layout or
        # expert-to-scale misalignment.
        compare("monolithic vs torch reference", baseline, reference, rtol=2e-1, atol=2e-1),
    ]
    result: dict[str, Any] = {
        "mode": args.mode,
        "tensor_list_supported": False,
        "shapes": {
            "x": list(x_quantized.shape),
            "monolithic_weight": list(monolithic_weight.shape),
            "split_weight": [list(weight.shape) for weight in weight_list],
            "monolithic_scale": list(monolithic_scale.shape),
            "split_scale": [list(scale.shape) for scale in scale_list],
            "output": list(baseline.shape),
        },
        "formats": {
            "monolithic_weight": npu_format_name(monolithic_weight),
            "split_weight": [npu_format_name(weight) for weight in weight_list],
        },
        "comparisons": [asdict(result) for result in comparisons],
    }
    try:
        tensor_list = torch_npu.npu_grouped_matmul(
            weight=weight_list,
            antiquant_scale=scale_list,
            **common_kwargs,
        )[0]
    except RuntimeError as error:
        result["tensor_list_error"] = operator_error_summary(error)
        return result

    result["tensor_list_supported"] = True
    comparisons.extend(
        [
            compare("tensor-list vs monolithic", tensor_list, baseline, rtol=0.0, atol=0.0),
            compare("tensor-list vs torch reference", tensor_list, reference, rtol=2e-1, atol=2e-1),
        ]
    )
    result["comparisons"] = [asdict(comparison) for comparison in comparisons]
    return result


def main() -> None:
    args = parse_args()
    torch_npu.npu.set_device(args.device)
    torch_npu.npu.config.allow_internal_format = True

    if args.model_path is not None:
        result = run_real_int_w4a8(args)
    elif args.mode == "int-w4a8":
        validate_dimensions(args.group_sizes, args.hidden_size, args.output_size)
        result = run_int_w4a8(args)
    else:
        validate_dimensions(args.group_sizes, args.hidden_size, args.output_size)
        result = run_mxfp4_w4a8(args)

    result["device"] = args.device
    result["seed"] = args.seed
    result["all_passed"] = result["tensor_list_supported"] and all(
        comparison["passed"] for comparison in result["comparisons"]
    )
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
