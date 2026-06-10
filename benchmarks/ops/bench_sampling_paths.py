#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from types import MethodType, SimpleNamespace

import torch
import torch_npu  # noqa: F401
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

import vllm_ascend.ascend_config as ascend_config_module
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm_ascend.sample.sampler import AscendSampler, _fill_gumbel
from vllm_ascend.utils import enable_custom_op

SINK = None
DEFAULT_BATCH_SIZES = (1, 8, 32, 64, 96)
DEFAULT_SCENARIOS = ("regular", "spec")
DEFAULT_PATHS = ("no_async_exponential", "v1_native", "optimized")
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


def set_enable_async_exponential(enabled: bool) -> bool:
    old_enabled = ascend_config_module._ASCEND_CONFIG.enable_async_exponential
    ascend_config_module._ASCEND_CONFIG.enable_async_exponential = enabled
    return old_enabled


def sync_get_gumbel(self, logits: torch.Tensor, generators: dict[int, torch.Generator]) -> torch.Tensor:
    gumbel = torch.empty_like(logits, dtype=torch.float32)
    _fill_gumbel(gumbel, generators)
    return gumbel


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
) -> SpecDecodeMetadata:
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
    draft_token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size * num_speculative_steps,),
        dtype=torch.int32,
        device=device,
    )
    return SpecDecodeMetadata(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        cu_num_draft_tokens=cu_num_draft_tokens,
        cu_num_sampled_tokens=cu_num_sampled_tokens,
        target_logits_indices=torch.tensor(target_logits_indices, dtype=torch.int32, device=device),
        bonus_logits_indices=torch.tensor(bonus_logits_indices, dtype=torch.int32, device=device),
        logits_indices=logits_indices,
    )


def boost_draft_logits(
    logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
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
    cols = draft_token_ids.to(torch.int64)
    logits[rows, cols] += boost


def benchmark(
    name: str,
    fn: Callable[[], torch.Tensor],
    warmups: int,
    iterations: int,
    prepare: Callable[[], None] | None = None,
) -> dict[str, float | list[float] | str]:
    global SINK
    for _ in range(warmups):
        if prepare is not None:
            prepare()
        SINK = fn()
    torch.npu.synchronize()

    times_ms: list[float] = []
    for _ in range(iterations):
        if prepare is not None:
            prepare()
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
    no_async_sampler = AscendSampler()
    no_async_sampler.topk_topp_sampler._get_gumbel = MethodType(
        sync_get_gumbel,
        no_async_sampler.topk_topp_sampler,
    )
    no_async_rejection_sampler = AscendRejectionSampler(no_async_sampler)
    no_async_rejection_sampler._can_use_optimized_rejection = lambda *_args, **_kwargs: False
    sampling_metadata = make_sampling_metadata(
        batch_size,
        device,
        args.temperature,
        args.top_k,
        args.top_p,
    )
    cases: list[tuple[str, str, Callable[[], torch.Tensor], Callable[[], None] | None]] = []

    if "regular" in args.scenarios:
        regular_logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device)
        regular_logits_by_path = {path: regular_logits.clone() for path in args.paths}

        def prepare_regular_logits(path: str) -> None:
            regular_logits_by_path[path].copy_(regular_logits, non_blocking=True)

        def prepare_regular_no_async() -> None:
            prepare_regular_logits("no_async_exponential")

        def prepare_regular_v1_native() -> None:
            prepare_regular_logits("v1_native")
            sampler.do_async_exponential(
                b_s=batch_size,
                head_dim=vocab_size,
                generators=sampling_metadata.generators,
            )

        def prepare_regular_optimized() -> None:
            prepare_regular_logits("optimized")
            sampler.prepare_async_gumbel(
                b_s=batch_size,
                head_dim=vocab_size,
                generators=sampling_metadata.generators,
            )

        def regular_no_async() -> torch.Tensor:
            old_enable_async_exponential = set_enable_async_exponential(False)
            try:
                return no_async_sampler(
                    regular_logits_by_path["no_async_exponential"],
                    sampling_metadata,
                ).sampled_token_ids
            finally:
                ascend_config_module._ASCEND_CONFIG.enable_async_exponential = old_enable_async_exponential

        def regular_v1_native() -> torch.Tensor:
            old_enable_async_exponential = set_enable_async_exponential(True)
            try:
                return sampler(regular_logits_by_path["v1_native"], sampling_metadata).sampled_token_ids
            finally:
                ascend_config_module._ASCEND_CONFIG.enable_async_exponential = old_enable_async_exponential

        def regular_optimized() -> torch.Tensor:
            old_enable_async_exponential = set_enable_async_exponential(False)
            try:
                return sampler(regular_logits_by_path["optimized"], sampling_metadata).sampled_token_ids
            finally:
                ascend_config_module._ASCEND_CONFIG.enable_async_exponential = old_enable_async_exponential

        if "no_async_exponential" in args.paths:
            cases.append(("regular", "no_async_exponential", regular_no_async, prepare_regular_no_async))
        if "v1_native" in args.paths:
            cases.append(("regular", "v1_native", regular_v1_native, prepare_regular_v1_native))
        if "optimized" in args.paths:
            cases.append(("regular", "optimized", regular_optimized, prepare_regular_optimized))

    if "spec" in args.scenarios:
        spec_metadata = make_spec_metadata(batch_size, spec_steps, vocab_size, device)
        spec_logits = torch.randn(num_logits, vocab_size, dtype=torch.float32, device=device)
        boost_draft_logits(spec_logits, spec_metadata.draft_token_ids, spec_steps, boost=12.0)
        spec_logits_by_path = {path: spec_logits.clone() for path in args.paths}

        def prepare_spec_logits(path: str) -> None:
            spec_logits_by_path[path].copy_(spec_logits, non_blocking=True)

        def prepare_spec_no_async() -> None:
            prepare_spec_logits("no_async_exponential")

        def prepare_spec_v1_native() -> None:
            prepare_spec_logits("v1_native")
            sampler.do_async_exponential(
                b_s=batch_size,
                head_dim=vocab_size,
                generators=sampling_metadata.generators,
            )

        def prepare_spec_optimized() -> None:
            prepare_spec_logits("optimized")
            rejection_sampler.prepare_async_rejection_random(
                num_logits=num_logits,
                recovery_vocab_size=vocab_size,
                num_draft_tokens=spec_metadata.num_draft_tokens,
                sampling_metadata=sampling_metadata,
            )

        def spec_no_async() -> torch.Tensor:
            old_enable_async_exponential = set_enable_async_exponential(False)
            try:
                return no_async_rejection_sampler(
                    spec_metadata,
                    None,
                    spec_logits_by_path["no_async_exponential"],
                    sampling_metadata,
                ).sampled_token_ids
            finally:
                ascend_config_module._ASCEND_CONFIG.enable_async_exponential = old_enable_async_exponential

        def spec_v1_native() -> torch.Tensor:
            old_enable_async_exponential = set_enable_async_exponential(True)
            try:
                return rejection_sampler(
                    spec_metadata,
                    None,
                    spec_logits_by_path["v1_native"],
                    sampling_metadata,
                ).sampled_token_ids
            finally:
                ascend_config_module._ASCEND_CONFIG.enable_async_exponential = old_enable_async_exponential

        def spec_optimized() -> torch.Tensor:
            old_enable_async_exponential = set_enable_async_exponential(False)
            try:
                return rejection_sampler(
                    spec_metadata,
                    None,
                    spec_logits_by_path["optimized"],
                    sampling_metadata,
                ).sampled_token_ids
            finally:
                ascend_config_module._ASCEND_CONFIG.enable_async_exponential = old_enable_async_exponential

        if "no_async_exponential" in args.paths:
            cases.append(("spec", "no_async_exponential", spec_no_async, prepare_spec_no_async))
        if "v1_native" in args.paths:
            cases.append(("spec", "v1_native", spec_v1_native, prepare_spec_v1_native))
        if "optimized" in args.paths:
            cases.append(("spec", "optimized", spec_optimized, prepare_spec_optimized))

    results = []
    for scenario, path, fn, prepare in cases:
        name = f"{scenario}/{path}"
        result = benchmark(name, fn, args.warmups, args.iterations, prepare)
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
            "Each measured iteration refreshes a per-path working logits tensor before timing "
            "so timed rows do not include input clone.",
            "no_async_exponential restores the old non-prefetched regular random path inside the benchmark only.",
            "v1_native uses the existing sampler calls without the optimized rejection operator.",
            "optimized uses prefetched regular Gumbel/rejection random tensors; "
            "spec bonus tokens are sampled from bonus logits after draft acceptance.",
        ],
        "results": results,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
