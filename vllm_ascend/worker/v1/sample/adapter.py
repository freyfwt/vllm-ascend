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

import numpy as np
import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.worker.v1.sample.sampling_context import V1SamplingContext

_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max
_UINT64_MODULUS = 2**64


class V1SamplerAdapter:
    """Opt-in bridge from the v1 model runner to Ascend gumbel sampling.

    Phase 1b intentionally supports only normal decode rows. Unsupported
    sampling surfaces are filtered by the model runner before this adapter is
    called, and guarded here for direct use in tests or future integrations.
    """

    def __init__(
        self,
        max_num_reqs: int,
        device: torch.device,
    ):
        self._max_num_reqs = max_num_reqs
        self._device = device
        self._request_seeds: dict[str, int] = {}
        self._seeds_cpu = torch.empty(max_num_reqs, dtype=torch.int64, device="cpu")
        self._seeds_np = self._seeds_cpu.numpy()
        self._seeds_device = torch.empty(max_num_reqs, dtype=torch.int64, device=device)

    def __call__(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        ctx: V1SamplingContext,
    ) -> SamplerOutput:
        sampled = self._sample(logits, sampling_metadata, ctx).to(torch.int32)
        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=None,
        )

    def can_sample(
        self,
        sampling_metadata: SamplingMetadata,
        ctx: V1SamplingContext,
    ) -> bool:
        if ctx.num_reqs > self._max_num_reqs:
            return False
        if ctx.req_ids is None:
            return False
        if not ctx.is_identity_request_mapping:
            return False
        if sampling_metadata.max_num_logprobs is not None:
            return False
        if sampling_metadata.logprob_token_ids:
            return False
        if not sampling_metadata.no_penalties:
            return False
        if sampling_metadata.allowed_token_ids_mask is not None:
            return False
        if sampling_metadata.bad_words_token_ids and any(
            bool(words) for words in sampling_metadata.bad_words_token_ids.values()
        ):
            return False
        logitsprocs = sampling_metadata.logitsprocs
        if logitsprocs is not None:
            if list(getattr(logitsprocs, "non_argmax_invariant", ())):
                return False
            if list(getattr(logitsprocs, "argmax_invariant", ())):
                return False
        return True

    def _sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        ctx: V1SamplingContext,
    ) -> torch.Tensor:
        temperature = self._temperature_for_sampling(sampling_metadata, ctx)
        seeds = self._compute_seeds(sampling_metadata, ctx)
        top_k, top_p = self._top_k_top_p_tensors(sampling_metadata, logits.device)
        has_top_k_top_p = top_k is not None or top_p is not None
        from vllm_ascend.worker.v2.sample import gumbel as gumbel_ops

        if has_top_k_top_p:
            if gumbel_ops.can_use_compact_top_k_top_p_sample(
                logits=logits,
                idx_mapping=ctx.expanded_idx_mapping,
                temperature=temperature,
                seed=seeds,
                pos=ctx.pos,
                k=top_k,
                p=top_p,
            ):
                return gumbel_ops.compact_top_k_top_p_sample(
                    logits=logits,
                    idx_mapping=ctx.expanded_idx_mapping,
                    temperature=temperature,
                    seed=seeds,
                    pos=ctx.pos,
                    k=top_k,
                    p=top_p,
                )

            from vllm_ascend.sample.sampler import apply_top_k_top_p

            gumbel_ops.apply_temperature(logits, ctx.expanded_idx_mapping, temperature)
            logits = apply_top_k_top_p(logits, top_k, top_p)

        return gumbel_ops.gumbel_sample(
            logits=logits,
            idx_mapping=ctx.expanded_idx_mapping,
            temperature=temperature,
            seed=seeds,
            pos=ctx.pos,
            apply_temperature=not has_top_k_top_p,
        )

    def _top_k_top_p_tensors(
        self,
        sampling_metadata: SamplingMetadata,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        top_k = sampling_metadata.top_k
        top_p = sampling_metadata.top_p
        if top_k is not None:
            if not isinstance(top_k, torch.Tensor):
                raise TypeError("top_k must be a torch.Tensor or None")
            if top_k.device != device:
                raise ValueError("top_k must be on the logits device")
        if top_p is not None:
            if not isinstance(top_p, torch.Tensor):
                raise TypeError("top_p must be a torch.Tensor or None")
            if top_p.device != device:
                raise ValueError("top_p must be on the logits device")
        return top_k, top_p

    def _temperature_for_sampling(
        self,
        sampling_metadata: SamplingMetadata,
        ctx: V1SamplingContext,
    ) -> torch.Tensor:
        temperature = sampling_metadata.temperature
        if temperature is None:
            return torch.zeros(ctx.num_reqs, dtype=torch.float32, device=self._device)
        if not isinstance(temperature, torch.Tensor):
            raise TypeError("temperature must be a torch.Tensor or None")
        if temperature.device != self._device:
            raise ValueError("temperature must be on the adapter device")
        if int(temperature.shape[0]) < ctx.num_reqs:
            raise ValueError("temperature must have at least one entry per active request")
        return temperature[: ctx.num_reqs].to(dtype=torch.float32)

    def _compute_seeds(
        self,
        sampling_metadata: SamplingMetadata,
        ctx: V1SamplingContext,
    ) -> torch.Tensor:
        req_ids = ctx.req_ids
        if req_ids is None:
            raise ValueError("V1SamplerAdapter requires request IDs for seed caching")

        active_req_ids = set(req_ids)
        for cached_req_id in tuple(self._request_seeds):
            if cached_req_id not in active_req_ids:
                del self._request_seeds[cached_req_id]

        generators = sampling_metadata.generators
        for req_idx, req_id in enumerate(req_ids):
            seed = self._request_seeds.get(req_id)
            if seed is None:
                generator = generators.get(req_idx)
                if generator is not None:
                    seed = self._normalize_seed(generator.initial_seed())
                else:
                    seed = self._new_random_seed()
                self._request_seeds[req_id] = seed
            self._seeds_np[req_idx] = seed
        seeds_cpu = self._seeds_cpu[: ctx.num_reqs]
        seeds_device = self._seeds_device[: ctx.num_reqs]
        seeds_device.copy_(seeds_cpu, non_blocking=True)
        return seeds_device

    @staticmethod
    def _normalize_seed(seed: int) -> int:
        seed = int(seed)
        if seed > _NP_INT64_MAX:
            seed -= _UINT64_MODULUS
        return seed

    @staticmethod
    def _new_random_seed() -> int:
        return int(np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX, dtype=np.int64))
