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
from vllm_ascend.utils import enable_custom_op

SINK = None
DEFAULT_BATCH_SIZES = (1, 8, 32, 64, 96)
DEFAULT_SCENARIOS = ("regular", "spec")
DEFAULT_PATHS = ("v1_native", "optimized")
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 20
DEFAULT_TOP_P = 0.95
_SAMPLING_EPS = 1e-5


def install_fake_ascend_config() -> None:
    ascend_config_module._ASCEND_CONFIG = SimpleNamespace(
        ascend_compilation_config=True,
        eplb_config=True,
        enable_reduce_sample=False,
        enable_async_exponential=False,
    )


def make_sampling_metadata(
    num_reqs: int,
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
) -> SamplingMetadata:
    all_greedy = temperature < _SAMPLING_EPS
    return SamplingMetadata(
        temperature=torch.full((num_reqs,), temperature, device=device, dtype=torch.float32),
        all_greedy=all_greedy,
        all_random=not all_greedy,
        top_p=torch.full((num_reqs,), top_p, device=device, dtype=torch.float32),
        top_k=torch.full((num_reqs,), top_k, device=device, dtype=torch.int32),
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
) -> tuple[SpecDecodeMetadata, torch.Tensor]:
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
    input_ids = torch.randint(
        0,
        vocab_size,
        (num_logits,),
        dtype=torch.int32,
        device=device,
    )
    draft_token_ids = input_ids[torch.tensor(target_logits_indices, dtype=torch.int32, device=device) + 1].contiguous()
    metadata = SpecDecodeMetadata(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        cu_num_draft_tokens=cu_num_draft_tokens,
        cu_num_sampled_tokens=cu_num_sampled_tokens,
        target_logits_indices=torch.tensor(target_logits_indices, dtype=torch.int32, device=device),
        bonus_logits_indices=torch.tensor(bonus_logits_indices, dtype=torch.int32, device=device),
        logits_indices=logits_indices,
    )
    return metadata, input_ids


def boost_draft_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    num_speculative_steps: int,
    boost: float,
) -> None:
    num_reqs = logits.shape[0] // (num_speculative_steps + 1)
    req_starts = torch.arange(
        0,
        num_reqs * (num_speculative_steps + 1),
        num_speculative_steps + 1,
        dtype=torch.int64,
        device=logits.device,
    )
    steps = torch.arange(num_speculative_steps, dtype=torch.int64, device=logits.device)
    rows = (req_starts.unsqueeze(1) + steps.unsqueeze(0)).flatten()
    cols = input_ids[rows + 1].to(torch.int64)
    logits[rows, cols] += boost


def make_gumbel(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    gumbel = torch.empty(shape, dtype=torch.float32, device=device)
    gumbel.exponential_()
    gumbel.log_().neg_()
    return gumbel


def make_rejection_randoms(
    count: int,
    num_draft_logits: int,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.npu.Event]]:
    randoms = []
    for _ in range(count):
        acceptance_uniform = torch.empty((num_draft_logits,), dtype=torch.float32, device=device)
        acceptance_uniform.uniform_()
        acceptance_uniform.clamp_(min=1e-20)
        recovery_gumbel = make_gumbel((batch_size, vocab_size), device)
        event = torch.npu.Event()
        event.record()
        randoms.append((acceptance_uniform, recovery_gumbel, event))
    torch.npu.synchronize()
    return randoms


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
    sampler = AscendSampler()
    rejection_sampler = AscendRejectionSampler(sampler)
    sampling_metadata = make_sampling_metadata(
        batch_size,
        device,
        args.temperature,
        args.top_k,
        args.top_p,
    )
    cases: list[tuple[str, str, Callable[[], torch.Tensor]]] = []

    if "regular" in args.scenarios:
        regular_logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device)
        regular_gumbels = [make_gumbel((batch_size, vocab_size), device) for _ in range(args.warmups + args.iterations)]
        regular_gumbel_idx = 0

        def regular_v1_native() -> torch.Tensor:
            return sampler(regular_logits.clone(), sampling_metadata).sampled_token_ids

        def regular_optimized() -> torch.Tensor:
            nonlocal regular_gumbel_idx
            sampler.set_gumbel_event(regular_gumbels[regular_gumbel_idx], None)
            regular_gumbel_idx += 1
            return sampler(regular_logits.clone(), sampling_metadata).sampled_token_ids

        if "v1_native" in args.paths:
            cases.append(("regular", "v1_native", regular_v1_native))
        if "optimized" in args.paths:
            cases.append(("regular", "optimized", regular_optimized))

    if "spec" in args.scenarios:
        spec_metadata, input_ids = make_spec_metadata(batch_size, spec_steps, vocab_size, device)
        spec_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=device)
        boost_draft_logits(spec_logits, input_ids, spec_steps, boost=12.0)
        spec_randoms = []
        if "optimized" in args.paths:
            spec_randoms = make_rejection_randoms(
                args.warmups + args.iterations,
                batch_size * spec_steps,
                batch_size,
                vocab_size,
                device,
            )
        spec_random_idx = 0

        def spec_v1_native() -> torch.Tensor:
            return rejection_sampler(
                spec_metadata,
                None,
                spec_logits.clone(),
                sampling_metadata,
            ).sampled_token_ids

        def spec_optimized() -> torch.Tensor:
            nonlocal spec_random_idx
            acceptance_uniform, recovery_gumbel, event = spec_randoms[spec_random_idx]
            spec_random_idx += 1
            rejection_sampler._acceptance_uniform = acceptance_uniform
            rejection_sampler._recovery_gumbel = recovery_gumbel
            rejection_sampler._random_event = event
            rejection_sampler._random_ready = True
            return rejection_sampler(
                spec_metadata,
                None,
                spec_logits.clone(),
                sampling_metadata,
                input_ids,
            ).sampled_token_ids

        if "v1_native" in args.paths:
            cases.append(("spec", "v1_native", spec_v1_native))
        if "optimized" in args.paths:
            cases.append(("spec", "optimized", spec_optimized))

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
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--scenarios", choices=DEFAULT_SCENARIOS, nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--paths", choices=DEFAULT_PATHS, nargs="+", default=list(DEFAULT_PATHS))
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    torch.npu.set_device(int(args.device.split(":")[-1]))
    init_device_properties_triton()
    if not enable_custom_op():
        raise RuntimeError("sampling path benchmark requires vllm_ascend custom ops")
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
        "sampling_params": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "scenarios": args.scenarios,
        "paths": args.paths,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "timer": "wall_clock_with_npu_synchronize",
        "notes": [
            "Measured with wall-clock time and NPU synchronization.",
            "v1_native uses the existing sampler calls without the optimized rejection operator.",
            "optimized uses prefetched regular Gumbel/rejection random tensors and the native rejection operator.",
            "The benchmark does not include ModelRunnerV2 bridge or FastSampler rows because that design is removed.",
        ],
        "results": results,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
