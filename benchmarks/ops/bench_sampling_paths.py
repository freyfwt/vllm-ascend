#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

import vllm_ascend.ascend_config as ascend_config_module
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.worker.v1.sample.adapter import (
    SamplingInputBuilder,
    process_regular_logits,
    process_spec_logits,
    sample_processed_logits,
    temperature_for_sampling,
)
from vllm_ascend.worker.v1.sample.target_rejection import (
    sample_with_rejection,
)
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
    rejection_sample as v2_rejection_sample,
)

SINK = None


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

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times_ms: list[float] = []
    for _ in range(iterations):
        start.record()
        SINK = fn()
        end.record()
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--spec-steps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    torch.npu.set_device(int(args.device.split(":")[-1]))
    init_device_properties_triton()
    install_fake_ascend_config()
    torch.manual_seed(20260601)

    device = torch.device(args.device)
    batch_size = args.batch_size
    vocab_size = args.vocab_size
    spec_steps = args.spec_steps
    num_logits = batch_size * (spec_steps + 1)

    sampler = AscendSampler()
    rejection_sampler = AscendRejectionSampler(sampler)
    sampling_metadata = make_sampling_metadata(batch_size, device)
    regular_logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device)
    regular_idx_mapping = torch.arange(batch_size, dtype=torch.int32, device=device)
    regular_seed = torch.arange(100000, 100000 + batch_size, dtype=torch.int64, device=device)
    regular_pos = torch.arange(batch_size, dtype=torch.int64, device=device)
    regular_gumbel = make_gumbel((batch_size, vocab_size), device)

    spec_metadata, input_ids, positions = make_spec_metadata(batch_size, spec_steps, vocab_size, device)
    spec_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=device)
    boost_draft_logits(spec_logits, input_ids, spec_steps, boost=12.0)
    builder = SamplingInputBuilder()
    temperature = temperature_for_sampling(sampling_metadata, batch_size, device)
    spec_inputs = builder.build_spec_inputs(spec_metadata, input_ids, positions, temperature, spec_steps)
    rejection_acceptance_uniform = torch.rand(num_logits, dtype=torch.float32, device=device)
    rejection_acceptance_uniform.clamp_(min=1e-20)
    rejection_recovery_gumbel = make_gumbel((batch_size, vocab_size), device)
    rejection_seed = torch.arange(200000, 200000 + batch_size, dtype=torch.int64, device=device)

    def regular_v1_original() -> torch.Tensor:
        return sampler(regular_logits, sampling_metadata).sampled_token_ids

    def regular_v2_native() -> torch.Tensor:
        processed, _ = process_regular_logits(sampler, regular_logits, sampling_metadata)
        return gumbel_sample(
            processed,
            regular_idx_mapping,
            sampling_metadata.temperature,
            regular_seed,
            regular_pos,
            apply_temperature=False,
        )

    def regular_our_optimized() -> torch.Tensor:
        processed, greedy_tokens = process_regular_logits(sampler, regular_logits, sampling_metadata)
        return sample_processed_logits(processed, sampling_metadata, regular_gumbel, greedy_tokens)

    def rejection_v1_original() -> torch.Tensor:
        return rejection_sampler(spec_metadata, None, spec_logits, sampling_metadata).sampled_token_ids

    def rejection_v2_native() -> torch.Tensor:
        processed = process_spec_logits(sampler, rejection_sampler, spec_logits, sampling_metadata, spec_metadata)
        sampled, _ = v2_rejection_sample(
            processed,
            None,
            spec_inputs.draft_tokens,
            spec_inputs.cu_num_logits,
            spec_inputs.positions,
            spec_inputs.idx_mapping,
            spec_inputs.expanded_idx_mapping,
            spec_inputs.expanded_local_pos,
            spec_inputs.temperature,
            rejection_seed,
            spec_steps,
        )
        return sampled

    def rejection_our_optimized() -> torch.Tensor:
        processed = process_spec_logits(sampler, rejection_sampler, spec_logits, sampling_metadata, spec_metadata)
        sampled, _ = sample_with_rejection(
            processed,
            spec_inputs.draft_tokens,
            spec_inputs.cu_num_logits,
            spec_inputs.positions,
            spec_inputs.idx_mapping,
            spec_inputs.expanded_idx_mapping,
            spec_inputs.expanded_local_pos,
            spec_inputs.temperature,
            rejection_acceptance_uniform,
            rejection_recovery_gumbel,
            spec_steps,
        )
        return sampled

    def regular_our_random_prefetch() -> torch.Tensor:
        gumbel = torch.empty_like(regular_gumbel)
        gumbel.exponential_()
        gumbel.log_().neg_()
        return gumbel

    def rejection_our_random_prefetch() -> torch.Tensor:
        accept = torch.empty_like(rejection_acceptance_uniform)
        accept.uniform_()
        accept.clamp_(min=1e-20)
        recovery = torch.empty_like(rejection_recovery_gumbel)
        recovery.exponential_()
        recovery.log_().neg_()
        return recovery

    cases: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("regular/v1_original", regular_v1_original),
        ("regular/v2_native", regular_v2_native),
        ("regular/our_optimized", regular_our_optimized),
        ("rejection/v1_original", rejection_v1_original),
        ("rejection/v2_native", rejection_v2_native),
        ("rejection/our_optimized", rejection_our_optimized),
        ("regular/our_random_prefetch_extra", regular_our_random_prefetch),
        ("rejection/our_random_prefetch_extra", rejection_our_random_prefetch),
    ]

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    results = []
    for name, fn in cases:
        results.append(benchmark(name, fn, args.warmups, args.iterations))

    base_by_group = {
        "regular": next(r for r in results if r["name"] == "regular/v1_original"),
        "rejection": next(r for r in results if r["name"] == "rejection/v1_original"),
    }
    for result in results:
        group = str(result["name"]).split("/")[0]
        if group in base_by_group:
            result["speedup_vs_v1_original"] = float(base_by_group[group]["mean_ms"]) / float(result["mean_ms"])

    payload = {
        "started": started,
        "device": args.device,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "spec_steps": spec_steps,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "notes": [
            "Measured with NPU events around the sampling critical path.",
            "our_optimized rows use prefetched random tensors, matching the model_runner overlap design.",
            "the *_random_prefetch_extra rows show standalone prefetch cost, outside critical-path speedup.",
            "v2_native rejection uses the current v2 NPU no-draft-logits helper.",
        ],
        "results": results,
    }
    print(json.dumps(payload, indent=2))

    print("\\n| case | mean ms | median ms | p90 ms | speedup vs v1 |")
    print("|---|---:|---:|---:|---:|")
    for result in results:
        speedup = result.get("speedup_vs_v1_original")
        speedup_text = "" if speedup is None else f"{float(speedup):.2f}x"
        print(
            f"| {result['name']} | {float(result['mean_ms']):.3f} | "
            f"{float(result['median_ms']):.3f} | {float(result['p90_ms']):.3f} | "
            f"{speedup_text} |"
        )


if __name__ == "__main__":
    main()
