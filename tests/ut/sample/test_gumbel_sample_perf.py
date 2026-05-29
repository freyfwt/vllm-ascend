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

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import random_sample
from vllm_ascend.worker.v2.sample import gumbel as gumbel_ops

VOCAB_SIZE = 152064
BATCH_SIZES = (1, 8, 32, 128)
WARMUP_ITERS = 1
MEASURE_ITERS = 3
REPEATS = 2
SAMPLING_SEED = 20260529


@dataclass(frozen=True)
class BenchImpl:
    name: str
    block_size: int | None = None
    apply_temperature: bool = True
    multibuffer: bool = True


@dataclass(frozen=True)
class BenchResult:
    batch_size: int
    vocab_size: int
    impl_name: str
    ms: float | None
    speedup_vs_golden: float | None
    error: str | None = None


GUMBEL_IMPLS = (
    BenchImpl("gumbel_current_bs1024", block_size=1024, apply_temperature=True),
    BenchImpl(
        "gumbel_current_bs1024_no_mb",
        block_size=1024,
        apply_temperature=True,
        multibuffer=False,
    ),
    BenchImpl("gumbel_no_temp_bs1024", block_size=1024, apply_temperature=False),
    BenchImpl("gumbel_no_temp_bs2048", block_size=2048, apply_temperature=False),
    BenchImpl("gumbel_no_temp_bs4096", block_size=4096, apply_temperature=False),
    BenchImpl(
        "gumbel_no_temp_bs4096_no_mb",
        block_size=4096,
        apply_temperature=False,
        multibuffer=False,
    ),
    BenchImpl(
        "gumbel_no_temp_bs8192_no_mb",
        block_size=8192,
        apply_temperature=False,
        multibuffer=False,
    ),
)


def is_npu_available() -> bool:
    if not hasattr(torch, "npu"):
        return False
    try:
        torch.device("npu")
        return bool(torch.npu.is_available())
    except RuntimeError:
        return False


class TestGumbelSamplePerf(TestBase):
    @unittest.skipUnless(
        is_npu_available(),
        "NPU is required for gumbel_sample performance comparison",
    )
    def test_gumbel_sample_perf_against_exponential_race_golden(self):
        device = torch.device("npu:0")
        results = []

        for batch_size in BATCH_SIZES:
            base_logits = self._make_logits(batch_size, VOCAB_SIZE, device)
            golden_ms, golden_output = self._measure_ms(
                lambda base_logits=base_logits: self._golden_exponential_race_sample(base_logits)
            )
            self._assert_token_ids(golden_output, batch_size, VOCAB_SIZE)
            results.append(
                BenchResult(
                    batch_size=batch_size,
                    vocab_size=VOCAB_SIZE,
                    impl_name="golden_exponential_race",
                    ms=golden_ms,
                    speedup_vs_golden=1.0,
                )
            )

            for impl in GUMBEL_IMPLS:
                results.append(self._run_gumbel_impl(base_logits, golden_ms, impl))

        self._print_results(results)

    def _make_logits(
        self,
        batch_size: int,
        vocab_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        torch.manual_seed(SAMPLING_SEED + batch_size)
        return torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device)

    def _golden_exponential_race_sample(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators={})

    def _run_gumbel_impl(
        self,
        logits: torch.Tensor,
        golden_ms: float,
        impl: BenchImpl,
    ) -> BenchResult:
        try:
            ms, output = self._measure_ms(lambda: self._gumbel_sample(logits, impl))
            self._assert_token_ids(output, logits.shape[0], logits.shape[1])
            return BenchResult(
                batch_size=logits.shape[0],
                vocab_size=logits.shape[1],
                impl_name=impl.name,
                ms=ms,
                speedup_vs_golden=golden_ms / ms,
            )
        except Exception as exc:  # noqa: BLE001 - keep exploratory variants non-fatal
            return BenchResult(
                batch_size=logits.shape[0],
                vocab_size=logits.shape[1],
                impl_name=impl.name,
                ms=None,
                speedup_vs_golden=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _gumbel_sample(
        self,
        logits: torch.Tensor,
        impl: BenchImpl,
    ) -> torch.Tensor:
        self.assertIsNotNone(impl.block_size)
        batch_size, vocab_size = logits.shape
        block_size = impl.block_size
        num_blocks = gumbel_ops.triton.cdiv(vocab_size, block_size)
        local_argmax = torch.empty(
            batch_size,
            num_blocks,
            dtype=torch.int64,
            device=logits.device,
        )
        local_max = torch.empty(
            batch_size,
            num_blocks,
            dtype=torch.float32,
            device=logits.device,
        )
        idx_mapping = torch.arange(batch_size, dtype=torch.int32, device=logits.device)
        temperature = torch.ones(batch_size, dtype=torch.float32, device=logits.device)
        seed = torch.arange(
            SAMPLING_SEED,
            SAMPLING_SEED + batch_size,
            dtype=torch.int64,
            device=logits.device,
        )
        pos = torch.arange(batch_size, dtype=torch.int64, device=logits.device)

        gumbel_ops._gumbel_sample_kernel[(batch_size, num_blocks)](
            local_argmax,
            local_argmax.stride(0),
            local_max,
            local_max.stride(0),
            None,
            0,
            None,
            logits,
            logits.stride(0),
            idx_mapping,
            seed,
            pos,
            temperature,
            vocab_size,
            BLOCK_SIZE=block_size,
            APPLY_TEMPERATURE=impl.apply_temperature,
            multibuffer=impl.multibuffer,
        )
        max_block_idx = local_max.argmax(dim=-1, keepdim=True)
        return local_argmax.gather(dim=-1, index=max_block_idx).view(-1)

    def _measure_ms(
        self,
        sample_once: Callable[[], torch.Tensor],
    ) -> tuple[float, torch.Tensor]:
        samples = []
        output = sample_once()
        for _ in range(WARMUP_ITERS):
            output = sample_once()

        for _ in range(REPEATS):
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(MEASURE_ITERS):
                output = sample_once()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0 / MEASURE_ITERS)

        return statistics.median(samples), output

    def _assert_token_ids(
        self,
        token_ids: torch.Tensor,
        batch_size: int,
        vocab_size: int,
    ) -> None:
        self.assertEqual(token_ids.shape, (batch_size,))
        self.assertTrue(bool((token_ids >= 0).all().item()))
        self.assertTrue(bool((token_ids < vocab_size).all().item()))

    def _print_results(self, results: list[BenchResult]) -> None:
        print("\ngumbel_sample performance comparison")
        print("golden: softmax -> exponential_ -> div -> argmax")
        print("batch     vocab impl                         ms speedup_vs_golden")
        print("----------------------------------------------------------------")
        for result in results:
            if result.error is not None:
                print(f"{result.batch_size:>5}{result.vocab_size:>10} {result.impl_name:<26} ERROR {result.error}")
                continue
            self.assertIsNotNone(result.ms)
            self.assertIsNotNone(result.speedup_vs_golden)
            print(
                f"{result.batch_size:>5}"
                f"{result.vocab_size:>10} "
                f"{result.impl_name:<26}"
                f"{result.ms:>9.3f}"
                f"{result.speedup_vs_golden:>18.3f}x"
            )


if __name__ == "__main__":
    unittest.main()
