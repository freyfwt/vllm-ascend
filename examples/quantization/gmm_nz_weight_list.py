#!/usr/bin/env python3
"""Check GMM1 with a monolithic NZ weight versus an NZ tensor list.

The script deliberately tests ``torch_npu.npu_grouped_matmul`` rather than
the fused MoE operators.  It has two modes:

* ``int-w4a8``: INT8 activations and packed INT4 weights.  Intended for A3.
* ``mxfp4-w4a8``: MXFP8 activations and MXFP4 weights.  Intended for A5.

For both modes, the monolithic call uses one 3-D weight and one scale tensor.
The tensor-list call uses one independently converted 2-D FRACTAL_NZ weight
and one scale tensor per expert.  Both calls use the same input and group list.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch_npu

ACL_FORMAT_FRACTAL_NZ = 29
DEFAULT_GROUP_SIZES = (4, 1, 3, 2)
MXFP_BLOCK_SIZE = 32


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
    encoded_scale_cpu = encode_int_scale(weight_scale_cpu)
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
    tensor_list = torch_npu.npu_grouped_matmul(
        weight=weight_list,
        scale=scale_list,
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
        compare("tensor-list vs monolithic", tensor_list, baseline, rtol=0.0, atol=0.0),
        compare("monolithic vs torch reference", baseline, reference, rtol=2e-2, atol=5e-2),
        compare("tensor-list vs torch reference", tensor_list, reference, rtol=2e-2, atol=5e-2),
    ]
    return {
        "mode": args.mode,
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


def process_mxfp_scale(scale: torch.Tensor) -> torch.Tensor:
    """Mirror W4A8 MXFP4 MoE scale processing in the quantization method."""
    num_experts, output_size, num_blocks = scale.shape
    if num_blocks % 2:
        raise ValueError(f"MXFP scale block count must be even, got {num_blocks}")
    return scale.reshape(num_experts, output_size, num_blocks // 2, 2).transpose(-3, -2).contiguous()


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
        "weight_dtype": torch_npu.float4_e2m1fn_x2,
        "scale_dtype": torch_npu.float8_e8m0fnu,
        "per_token_scale_dtype": torch_npu.float8_e8m0fnu,
    }
    baseline = torch_npu.npu_grouped_matmul(
        weight=[monolithic_weight],
        scale=[monolithic_scale],
        **common_kwargs,
    )[0]
    tensor_list = torch_npu.npu_grouped_matmul(
        weight=weight_list,
        scale=scale_list,
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
        compare("tensor-list vs monolithic", tensor_list, baseline, rtol=0.0, atol=0.0),
        # This reference starts from the pre-quantized BF16 tensors, so allow
        # normal MXFP4/MXFP8 quantization error while still catching layout or
        # expert-to-scale misalignment.
        compare("monolithic vs torch reference", baseline, reference, rtol=2e-1, atol=2e-1),
        compare("tensor-list vs torch reference", tensor_list, reference, rtol=2e-1, atol=2e-1),
    ]
    return {
        "mode": args.mode,
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


def main() -> None:
    args = parse_args()
    validate_dimensions(args.group_sizes, args.hidden_size, args.output_size)
    torch_npu.npu.set_device(args.device)
    torch_npu.npu.config.allow_internal_format = True

    if args.mode == "int-w4a8":
        result = run_int_w4a8(args)
    else:
        result = run_mxfp4_w4a8(args)

    result["device"] = args.device
    result["seed"] = args.seed
    result["all_passed"] = all(comparison["passed"] for comparison in result["comparisons"])
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
