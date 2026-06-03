#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import torch
import torch_npu  # noqa: F401
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

import vllm_ascend.ascend_config as ascend_config_module
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.sample.sampling_bridge import (
    SamplingBridge,
)
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
    rejection_sample as v2_rejection_sample,
)

SINK = None
DEFAULT_BATCH_SIZES = (1, 8, 32, 64, 96)
DEFAULT_SCENARIOS = ("regular", "spec")
DEFAULT_PATHS = ("v1_native", "v2_native", "v2_optimized")


def install_fake_ascend_config() -> None:
    ascend_config_module._ASCEND_CONFIG = SimpleNamespace(
        ascend_compilation_config=True,
        eplb_config=True,
        enable_reduce_sample=False,
        enable_async_exponential=False,
    )


def make_sampling_metadata(num_reqs: int, device: torch.device) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.ones(num_reqs, device=device, dtype=torch.float32),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0, device=device),
        presence_penalties=torch.empty(0, device=device),
        repetition_penalties=torch.empty(0, device=device),
        output_token_ids=[[] for _ in range(num_reqs)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        logprob_token_ids=None,
        spec_token_ids=None,
    )


def make_spec_metadata(
    batch_size: int,
    num_speculative_steps: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[SpecDecodeMetadata, torch.Tensor, torch.Tensor]:
    num_draft_tokens = [num_speculative_steps] * batch_size
    num_sampled_tokens = [num_speculative_steps + 1] * batch_size
    cu_num_draft_tokens = torch.tensor(num_draft_tokens, dtype=torch.int32, device=device).cumsum(0)
    cu_num_sampled_tokens = torch.tensor(num_sampled_tokens, dtype=torch.int32, device=device).cumsum(0)

    target_logits_indices: list[int] = []
    bonus_logits_indices: list[int] = []
    for req_idx in range(batch_size):
        start = req_idx * (num_speculative_steps + 1)
        target_logits_indices.extend(range(start, start + num_speculative_steps))
        bonus_logits_indices.append(start + num_speculative_steps)

    num_logits = batch_size * (num_speculative_steps + 1)
    logits_indices = torch.arange(num_logits, dtype=torch.int32, device=device)
    draft_segments = torch.randint(
        0,
        vocab_size,
        (batch_size, num_speculative_steps + 1),
        dtype=torch.int32,
        device=device,
    )
    draft_token_ids = draft_segments[:, 1:].contiguous().view(-1)
    metadata = SpecDecodeMetadata(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        cu_num_draft_tokens=cu_num_draft_tokens,
        cu_num_sampled_tokens=cu_num_sampled_tokens,
        target_logits_indices=torch.tensor(target_logits_indices, dtype=torch.int32, device=device),
        bonus_logits_indices=torch.tensor(bonus_logits_indices, dtype=torch.int32, device=device),
        logits_indices=logits_indices,
    )
    positions = torch.arange(num_logits, dtype=torch.int64, device=device)
    return metadata, draft_segments.flatten(), positions


def boost_draft_logits(
    logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_speculative_steps: int,
    boost: float,
) -> None:
    rows = []
    cols = []
    num_reqs = logits.shape[0] // (num_speculative_steps + 1)
    for req_idx in range(num_reqs):
        start = req_idx * (num_speculative_steps + 1)
        for step in range(num_speculative_steps):
            rows.append(start + step)
            cols.append(int(draft_tokens[start + step + 1].item()))
    logits[
        torch.tensor(rows, dtype=torch.int64, device=logits.device),
        torch.tensor(cols, dtype=torch.int64, device=logits.device),
    ] += boost


def make_gumbel(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    gumbel = torch.empty(shape, dtype=torch.float32, device=device)
    gumbel.exponential_()
    gumbel.log_().neg_()
    return gumbel


def make_bridge_batch(batch_size: int) -> tuple[SimpleNamespace, dict[str, SimpleNamespace]]:
    req_ids = [f"req_{idx}" for idx in range(batch_size)]
    token_ids = torch.arange(batch_size, dtype=torch.int32).view(batch_size, 1)
    input_batch = SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={req_id: idx for idx, req_id in enumerate(req_ids)},
        num_prompt_tokens=torch.ones(batch_size, dtype=torch.int32).numpy(),
        num_tokens_no_spec=torch.ones(batch_size, dtype=torch.int32).numpy(),
        num_computed_tokens_cpu=torch.ones(batch_size, dtype=torch.int32).numpy(),
        token_ids_cpu=token_ids.numpy(),
        is_token_ids=torch.ones(batch_size, 1, dtype=torch.bool).numpy(),
    )
    requests = {req_id: SimpleNamespace(sampling_params=SamplingParams(temperature=1.0)) for req_id in req_ids}
    return input_batch, requests


def benchmark(
    name: str,
    fn: Callable[[], torch.Tensor],
    warmups: int,
    iterations: int,
) -> dict[str, float | list[float] | str]:
    global SINK
    for _ in range(warmups):
        SINK = fn()
    torch.npu.synchronize()

    times_ms: list[float] = []
    for _ in range(iterations):
        torch.npu.synchronize()
        start = time.perf_counter()
        SINK = fn()
        torch.npu.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)

    times_sorted = sorted(times_ms)
    return {
        "name": name,
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "p90_ms": times_sorted[int(0.9 * (len(times_sorted) - 1))],
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "times_ms": times_ms,
    }


def make_sampling_bridge(
    batch_size: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    vocab_size: int,
    spec_steps: int,
    device: torch.device,
    enable_spec: bool,
) -> SamplingBridge:
    speculative_config = None
    if enable_spec:
        speculative_config = SimpleNamespace(
            num_speculative_tokens=spec_steps,
            rejection_sample_method="probabilistic",
        )
    return SamplingBridge(
        max_num_reqs=batch_size,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        vocab_size=vocab_size,
        device=device,
        logprobs_mode="raw_logprobs",
        speculative_config=speculative_config,
    )


def resolve_batch_sizes(args: argparse.Namespace) -> list[int]:
    if args.batch_size is not None:
        return [args.batch_size]
    if args.batch_sizes is not None:
        return args.batch_sizes
    return list(DEFAULT_BATCH_SIZES)


def run_batch_size(
    args: argparse.Namespace,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, float | int | list[float] | str]]:
    vocab_size = args.vocab_size
    spec_steps = args.spec_steps
    num_logits = batch_size * (spec_steps + 1)
    max_model_len = 1 + (args.warmups + args.iterations + 4) * (spec_steps + 1)

    sampler = AscendSampler()
    sampling_metadata = make_sampling_metadata(batch_size, device)
    cases: list[tuple[str, str, Callable[[], torch.Tensor]]] = []

    if "regular" in args.scenarios:
        regular_logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device)
        regular_seed = torch.arange(100000, 100000 + batch_size, dtype=torch.int64, device=device)
        regular_gumbel = make_gumbel((batch_size, vocab_size), device)
        regular_bridge = make_sampling_bridge(
            batch_size=batch_size,
            max_model_len=max_model_len,
            max_num_batched_tokens=batch_size,
            vocab_size=vocab_size,
            spec_steps=spec_steps,
            device=device,
            enable_spec=False,
        )
        regular_input_batch, regular_requests = make_bridge_batch(batch_size)
        regular_input_ids = torch.arange(batch_size, dtype=torch.int32, device=device)
        regular_positions = torch.arange(batch_size, dtype=torch.int64, device=device)
        regular_logits_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        regular_inputs = regular_bridge.prepare(
            regular_input_batch,
            regular_requests,
            regular_input_ids,
            regular_positions,
            regular_logits_indices,
            None,
        )
        assert regular_inputs is not None
        regular_query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        regular_query_start_loc_np = np.arange(batch_size + 1, dtype=np.int32)
        regular_seq_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
        regular_seq_lens_cpu_upper_bound = torch.ones(batch_size, dtype=torch.int32)

        def regular_v1_native() -> torch.Tensor:
            return sampler(regular_logits, sampling_metadata).sampled_token_ids

        def regular_v2_native() -> torch.Tensor:
            regular_pos = regular_inputs.positions[regular_inputs.logits_indices]
            regular_token_ids = regular_inputs.input_ids[regular_inputs.logits_indices]
            processed = regular_bridge.sampler.apply_sampling_params(
                regular_logits,
                regular_inputs.expanded_idx_mapping,
                regular_inputs.idx_mapping_np,
                regular_pos,
                regular_token_ids,
                regular_inputs.expanded_local_pos,
            )
            return gumbel_sample(
                processed,
                regular_inputs.expanded_idx_mapping,
                regular_bridge.sampler.sampling_states.temperature.gpu,
                regular_seed,
                regular_pos,
                apply_temperature=False,
            )

        def regular_v2_optimized() -> torch.Tensor:
            if not regular_bridge.update_requests(regular_input_batch, regular_requests):
                raise RuntimeError("failed to update regular bridge requests")
            sample_batch = regular_bridge.bind_batch(
                regular_input_batch,
                regular_input_ids,
                regular_positions,
                regular_logits_indices,
                None,
                query_start_loc=regular_query_start_loc,
                query_start_loc_np=regular_query_start_loc_np,
                seq_lens=regular_seq_lens,
                seq_lens_cpu_upper_bound=regular_seq_lens_cpu_upper_bound,
            )
            output = regular_bridge.sample_regular(
                regular_logits,
                sample_batch,
                regular_gumbel,
                preserve_original_logits=False,
            )
            regular_bridge.post_update(
                sample_batch,
                output.sampled_token_ids,
                output.num_sampled,
            )
            return output.sampled_token_ids

        if "v1_native" in args.paths:
            cases.append(("regular", "v1_native", regular_v1_native))
        if "v2_native" in args.paths:
            cases.append(("regular", "v2_native", regular_v2_native))
        if "v2_optimized" in args.paths:
            cases.append(("regular", "v2_optimized", regular_v2_optimized))

    if "spec" in args.scenarios:
        rejection_sampler = AscendRejectionSampler(sampler)
        spec_metadata, input_ids, positions = make_spec_metadata(batch_size, spec_steps, vocab_size, device)
        spec_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=device)
        boost_draft_logits(spec_logits, input_ids, spec_steps, boost=12.0)
        spec_bridge = make_sampling_bridge(
            batch_size=batch_size,
            max_model_len=max_model_len,
            max_num_batched_tokens=num_logits,
            vocab_size=vocab_size,
            spec_steps=spec_steps,
            device=device,
            enable_spec=True,
        )
        spec_input_batch, spec_requests = make_bridge_batch(batch_size)
        spec_inputs = spec_bridge.prepare(
            spec_input_batch,
            spec_requests,
            input_ids,
            positions,
            spec_metadata.logits_indices,
            spec_metadata,
        )
        assert spec_inputs is not None
        tokens_per_req = spec_steps + 1
        spec_query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        spec_query_start_loc.mul_(tokens_per_req)
        spec_query_start_loc_np = np.arange(batch_size + 1, dtype=np.int32) * tokens_per_req
        spec_seq_lens = torch.full((batch_size,), tokens_per_req, dtype=torch.int32, device=device)
        spec_seq_lens_cpu_upper_bound = torch.full((batch_size,), tokens_per_req, dtype=torch.int32)
        rejection_acceptance_uniform = torch.rand(num_logits, dtype=torch.float32, device=device)
        rejection_acceptance_uniform.clamp_(min=1e-20)
        rejection_recovery_gumbel = make_gumbel((batch_size, vocab_size), device)
        rejection_seed = torch.arange(200000, 200000 + batch_size, dtype=torch.int64, device=device)

        def spec_v1_native() -> torch.Tensor:
            return rejection_sampler(spec_metadata, None, spec_logits, sampling_metadata).sampled_token_ids

        def spec_v2_native() -> torch.Tensor:
            draft_tokens = spec_inputs.input_ids[spec_inputs.logits_indices]
            spec_pos = spec_inputs.positions[spec_inputs.logits_indices]
            processed = spec_bridge.sampler.apply_sampling_params(
                spec_logits,
                spec_inputs.expanded_idx_mapping,
                spec_inputs.idx_mapping_np,
                spec_pos,
                draft_tokens,
                spec_inputs.expanded_local_pos,
            )
            sampled, _ = v2_rejection_sample(
                processed,
                None,
                draft_tokens,
                spec_inputs.cu_num_logits,
                spec_pos,
                spec_inputs.idx_mapping,
                spec_inputs.expanded_idx_mapping,
                spec_inputs.expanded_local_pos,
                spec_bridge.sampler.sampling_states.temperature.gpu,
                rejection_seed,
                spec_steps,
            )
            return sampled

        def spec_v2_optimized() -> torch.Tensor:
            if not spec_bridge.update_requests(spec_input_batch, spec_requests):
                raise RuntimeError("failed to update spec bridge requests")
            sample_batch = spec_bridge.bind_batch(
                spec_input_batch,
                input_ids,
                positions,
                spec_metadata.logits_indices,
                spec_metadata,
                query_start_loc=spec_query_start_loc,
                query_start_loc_np=spec_query_start_loc_np,
                seq_lens=spec_seq_lens,
                seq_lens_cpu_upper_bound=spec_seq_lens_cpu_upper_bound,
            )
            output = spec_bridge.sample_spec(
                spec_logits,
                sample_batch,
                rejection_acceptance_uniform,
                rejection_recovery_gumbel,
                preserve_original_logits=False,
            )
            spec_bridge.post_update(
                sample_batch,
                output.sampled_token_ids,
                output.num_sampled,
            )
            return output.sampled_token_ids

        if "v1_native" in args.paths:
            cases.append(("spec", "v1_native", spec_v1_native))
        if "v2_native" in args.paths:
            cases.append(("spec", "v2_native", spec_v2_native))
        if "v2_optimized" in args.paths:
            cases.append(("spec", "v2_optimized", spec_v2_optimized))

    results = []
    for scenario, path, fn in cases:
        name = f"{scenario}/{path}"
        result = benchmark(name, fn, args.warmups, args.iterations)
        result["batch_size"] = batch_size
        result["scenario"] = scenario
        result["path"] = path
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--spec-steps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--scenarios", choices=DEFAULT_SCENARIOS, nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--paths", choices=DEFAULT_PATHS, nargs="+", default=list(DEFAULT_PATHS))
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    torch.npu.set_device(int(args.device.split(":")[-1]))
    init_device_properties_triton()
    install_fake_ascend_config()
    torch.manual_seed(20260601)

    device = torch.device(args.device)
    batch_sizes = resolve_batch_sizes(args)

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    results: list[dict[str, float | int | list[float] | str]] = []
    for batch_size in batch_sizes:
        results.extend(run_batch_size(args, batch_size, device))
        torch.npu.synchronize()
        torch.npu.empty_cache()

    base_by_group = {
        (result["batch_size"], result["scenario"]): result for result in results if result["path"] == "v1_native"
    }
    for result in results:
        base = base_by_group.get((result["batch_size"], result["scenario"]))
        if base is not None:
            result["speedup_vs_v1_native"] = float(base["mean_ms"]) / float(result["mean_ms"])

    payload = {
        "started": started,
        "device": args.device,
        "batch_sizes": batch_sizes,
        "vocab_size": args.vocab_size,
        "spec_steps": args.spec_steps,
        "scenarios": args.scenarios,
        "paths": args.paths,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "timer": "wall_clock_with_npu_synchronize",
        "notes": [
            "Measured with wall-clock time and NPU synchronization so host-side batch binding is included.",
            "Warmup iterations run first; measured iterations keep the same request batch in continuous decode state.",
            "v2_optimized includes stable update_requests, bind_batch, sampling, and post_update.",
            "v2_optimized uses prefetched random tensors, matching the model_runner overlap design.",
            "spec/v2_native uses the current v2 NPU no-draft-logits helper.",
        ],
        "results": results,
    }
    print(json.dumps(payload, indent=2))

    print("\n| batch | scenario | path | mean ms | median ms | p90 ms | speedup vs v1 |")
    print("|---:|---|---|---:|---:|---:|---:|")
    for result in results:
        speedup = result.get("speedup_vs_v1_native")
        speedup_text = "" if speedup is None else f"{float(speedup):.2f}x"
        print(
            f"| {result['batch_size']} | {result['scenario']} | {result['path']} | "
            f"{float(result['mean_ms']):.3f} | "
            f"{float(result['median_ms']):.3f} | {float(result['p90_ms']):.3f} | "
            f"{speedup_text} |"
        )


if __name__ == "__main__":
    main()
