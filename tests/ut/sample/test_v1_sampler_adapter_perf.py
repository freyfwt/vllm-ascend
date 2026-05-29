# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import statistics
import time
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.sample import sampler as sampler_module
from vllm_ascend.sample.sampler import AscendSampler
from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter
from vllm_ascend.worker.v1.sample.sampling_context import V1SamplingContext

VOCAB_SIZE = 152064
BATCH_SIZES = (1, 8, 32, 128)
WARMUP_ITERS = 2
MEASURE_ITERS = 5
REPEATS = 3
TOP_K_VALUE = 50
TOP_P_VALUE = 0.9
SAMPLING_SEED = 20260529


@dataclass(frozen=True)
class SamplingCase:
    name: str
    top_k: int | None
    top_p: float | None


@dataclass(frozen=True)
class PerfResult:
    case_name: str
    batch_size: int
    vocab_size: int
    filter_impl: str
    old_ms: float
    new_ms: float

    @property
    def speedup(self) -> float:
        return self.old_ms / self.new_ms


SAMPLING_CASES = (
    SamplingCase("full_vocab_sampling", top_k=None, top_p=None),
    SamplingCase("top_k_sampling", top_k=TOP_K_VALUE, top_p=None),
    SamplingCase("top_p_sampling", top_k=None, top_p=TOP_P_VALUE),
    SamplingCase("top_k_top_p_sampling", top_k=TOP_K_VALUE, top_p=TOP_P_VALUE),
)


def is_npu_available() -> bool:
    if not hasattr(torch, "npu"):
        return False
    try:
        torch.device("npu")
        return bool(torch.npu.is_available())
    except RuntimeError:
        return False


class TestV1SamplerAdapterPerf(TestBase):
    @staticmethod
    def _make_vllm_config():
        return SimpleNamespace(
            additional_config={
                "sampling_config": {
                    "enable_sampling_optimization": True,
                    "enable_reduced_sampling": False,
                }
            },
            cache_config=SimpleNamespace(block_size=16),
            compilation_config=SimpleNamespace(pass_config=SimpleNamespace(enable_sp=False)),
            kv_transfer_config=None,
            model_config=None,
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                enable_expert_parallel=False,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                tensor_parallel_size=1,
            ),
            quant_config=None,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
            speculative_config=None,
        )

    def setUp(self):
        super().setUp()
        clear_ascend_config()
        init_ascend_config(self._make_vllm_config())

    def tearDown(self):
        clear_ascend_config()
        super().tearDown()

    @unittest.skipUnless(
        is_npu_available(),
        "NPU is required for sampling performance comparison",
    )
    def test_v1_sampler_adapter_perf_against_ascend_sampler(self):
        device = torch.device("npu:0")
        results = []

        for sampling_case in SAMPLING_CASES:
            for batch_size in BATCH_SIZES:
                results.append(self._run_case(sampling_case, batch_size, VOCAB_SIZE, device))

        self._print_perf_results(results)

    def _make_sampling_metadata(
        self,
        batch_size: int,
        case: SamplingCase,
        device: torch.device,
    ) -> SamplingMetadata:
        top_k = None
        if case.top_k is not None:
            top_k = torch.full((batch_size,), case.top_k, dtype=torch.int32, device=device)

        top_p = None
        if case.top_p is not None:
            top_p = torch.full((batch_size,), case.top_p, dtype=torch.float32, device=device)

        return SamplingMetadata(
            temperature=torch.ones(batch_size, dtype=torch.float32, device=device),
            all_greedy=False,
            all_random=True,
            top_p=top_p,
            top_k=top_k,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.zeros(batch_size, dtype=torch.float32, device=device),
            presence_penalties=torch.zeros(batch_size, dtype=torch.float32, device=device),
            repetition_penalties=torch.ones(batch_size, dtype=torch.float32, device=device),
            output_token_ids=[[] for _ in range(batch_size)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=SimpleNamespace(
                non_argmax_invariant=(),
                argmax_invariant=(),
            ),
            logprob_token_ids=None,
        )

    def _make_sampling_context(
        self,
        batch_size: int,
        device: torch.device,
    ) -> V1SamplingContext:
        req_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        return V1SamplingContext.from_model_runner_inputs(
            num_reqs=batch_size,
            positions_at_logits=torch.arange(batch_size, dtype=torch.int64, device=device),
            input_ids_at_logits=torch.arange(1000, 1000 + batch_size, dtype=torch.int64, device=device),
            req_indices_at_logits=req_indices,
            device=device,
            req_ids=tuple(f"req{i}" for i in range(batch_size)),
        )

    def _measure_ms(
        self,
        sample_once: Callable[[], SamplerOutput],
        warmup_iters: int = WARMUP_ITERS,
        measure_iters: int = MEASURE_ITERS,
    ) -> tuple[float, SamplerOutput]:
        output = sample_once()
        for _ in range(warmup_iters):
            output = sample_once()

        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(measure_iters):
            output = sample_once()
        torch.npu.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0 / measure_iters
        return elapsed_ms, output

    def _assert_sampled_token_ids(
        self,
        output: SamplerOutput,
        batch_size: int,
        vocab_size: int,
    ) -> None:
        sampled_token_ids = output.sampled_token_ids
        self.assertEqual(sampled_token_ids.shape, (batch_size, 1))
        self.assertEqual(sampled_token_ids.dtype, torch.int32)
        self.assertTrue(bool((sampled_token_ids >= 0).all().item()))
        self.assertTrue(bool((sampled_token_ids < vocab_size).all().item()))
        self.assertIsNone(output.logprobs_tensors)

    def _top_k_top_p_filter_impl(self, case: SamplingCase) -> str:
        if case.top_k is None and case.top_p is None:
            return "none"
        if hasattr(torch.ops._C_ascend, "npu_apply_top_k_top_p"):
            return "ascendc"
        return "pytorch_fallback"

    def _set_top_k_top_p_filter_impl(
        self,
        old_sampler: AscendSampler,
        filter_impl: str,
    ):
        original_apply_top_k_top_p = sampler_module.apply_top_k_top_p
        if filter_impl == "pytorch_fallback":
            sampler_module.apply_top_k_top_p = sampler_module._apply_top_k_top_p_pytorch
            old_sampler.topk_topp_sampler.apply_top_k_top_p = sampler_module._apply_top_k_top_p_pytorch
        return original_apply_top_k_top_p

    def _run_case(
        self,
        case: SamplingCase,
        batch_size: int,
        vocab_size: int,
        device: torch.device,
    ) -> PerfResult:
        torch.manual_seed(SAMPLING_SEED + batch_size)
        base_logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device)
        sampling_metadata = self._make_sampling_metadata(batch_size, case, device)
        sampling_context = self._make_sampling_context(batch_size, device)
        old_sampler = AscendSampler()
        new_sampler = V1SamplerAdapter(max_num_reqs=batch_size, device=device)
        filter_impl = self._top_k_top_p_filter_impl(case)
        original_apply_top_k_top_p = self._set_top_k_top_p_filter_impl(old_sampler, filter_impl)

        try:
            return self._run_case_with_samplers(
                case=case,
                batch_size=batch_size,
                vocab_size=vocab_size,
                sampling_metadata=sampling_metadata,
                sampling_context=sampling_context,
                base_logits=base_logits,
                old_sampler=old_sampler,
                new_sampler=new_sampler,
                filter_impl=filter_impl,
            )
        finally:
            sampler_module.apply_top_k_top_p = original_apply_top_k_top_p

    def _run_case_with_samplers(
        self,
        case: SamplingCase,
        batch_size: int,
        vocab_size: int,
        sampling_metadata: SamplingMetadata,
        sampling_context: V1SamplingContext,
        base_logits: torch.Tensor,
        old_sampler: AscendSampler,
        new_sampler: V1SamplerAdapter,
        filter_impl: str,
    ) -> PerfResult:
        self.assertTrue(new_sampler.can_sample(sampling_metadata, sampling_context))

        def run_old_sampler() -> SamplerOutput:
            return old_sampler(
                logits=base_logits.clone(),
                sampling_metadata=sampling_metadata,
            )

        def run_new_sampler() -> SamplerOutput:
            return new_sampler(
                logits=base_logits.clone(),
                sampling_metadata=sampling_metadata,
                ctx=sampling_context,
            )

        old_times = []
        new_times = []
        old_output = None
        new_output = None
        for repeat_idx in range(REPEATS):
            if repeat_idx % 2 == 0:
                old_ms, old_output = self._measure_ms(run_old_sampler)
                new_ms, new_output = self._measure_ms(run_new_sampler)
            else:
                new_ms, new_output = self._measure_ms(run_new_sampler)
                old_ms, old_output = self._measure_ms(run_old_sampler)
            old_times.append(old_ms)
            new_times.append(new_ms)

        self.assertIsNotNone(old_output)
        self.assertIsNotNone(new_output)
        self._assert_sampled_token_ids(old_output, batch_size, vocab_size)
        self._assert_sampled_token_ids(new_output, batch_size, vocab_size)

        return PerfResult(
            case_name=case.name,
            batch_size=batch_size,
            vocab_size=vocab_size,
            filter_impl=filter_impl,
            old_ms=statistics.median(old_times),
            new_ms=statistics.median(new_times),
        )

    def _print_perf_results(self, results: list[PerfResult]) -> None:
        print("\nV1 sampler adapter performance comparison (clone_included=True)")
        print("case                         bs     vocab            filter     old_ms     new_ms  speedup")
        print("------------------------------------------------------------------------------------------")
        for result in results:
            print(
                f"{result.case_name:<28}"
                f"{result.batch_size:>4}"
                f"{result.vocab_size:>10}"
                f"{result.filter_impl:>18}"
                f"{result.old_ms:>11.3f}"
                f"{result.new_ms:>11.3f}"
                f"{result.speedup:>9.3f}x"
            )


if __name__ == "__main__":
    unittest.main()
