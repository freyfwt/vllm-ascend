#
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

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.sample.rejection_sampler import expand_batch_to_tokens
from vllm_ascend.sample.sampler import apply_top_k_top_p

_SAMPLING_EPS = 1e-5


@dataclass
class V1SamplingRandoms:
    accept_uniform: torch.Tensor | None = None
    sample_gumbel: torch.Tensor | None = None
    resample_gumbel: torch.Tensor | None = None
    ready_event: torch.npu.Event | None = None


@dataclass
class V1SpecDecodeSamplingInputs:
    cu_num_logits: torch.Tensor
    idx_mapping: torch.Tensor
    expanded_idx_mapping: torch.Tensor
    expanded_local_pos: torch.Tensor
    draft_sampled: torch.Tensor
    positions: torch.Tensor
    temperature: torch.Tensor


@triton.jit
def _fill_spec_decode_mapping_kernel(
    cu_num_logits_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    local_pos = tl.arange(0, BLOCK_SIZE)
    offset = start_idx + local_pos
    mask = offset < end_idx
    tl.store(expanded_idx_mapping_ptr + offset, req_idx, mask=mask)
    tl.store(expanded_local_pos_ptr + offset, local_pos, mask=mask)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


class V1V2SamplingInputBuilder:
    def __init__(self) -> None:
        self._buffers: dict[str, torch.Tensor] = {}

    def build_spec_decode_inputs(
        self,
        metadata: SpecDecodeMetadata,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        temperature: torch.Tensor,
        num_speculative_steps: int,
    ) -> V1SpecDecodeSamplingInputs:
        num_reqs = len(metadata.num_draft_tokens)
        num_logits = int(metadata.logits_indices.shape[0])
        device = input_ids.device

        cu_num_logits = self._buffer(
            "cu_num_logits", (num_reqs + 1,), torch.int32, device
        )
        cu_num_logits[0] = 0
        cu_num_logits[1:].copy_(metadata.cu_num_sampled_tokens)

        idx_mapping = self._idx_mapping(num_reqs, device)
        expanded_idx_mapping = self._buffer(
            "expanded_idx_mapping", (num_logits,), torch.int32, device
        )
        expanded_local_pos = self._buffer(
            "expanded_local_pos", (num_logits,), torch.int32, device
        )
        block_size = _next_power_of_2(num_speculative_steps + 1)
        _fill_spec_decode_mapping_kernel[(num_reqs,)](
            cu_num_logits,
            expanded_idx_mapping,
            expanded_local_pos,
            BLOCK_SIZE=block_size,
            num_warps=1,
        )

        logits_indices = metadata.logits_indices
        return V1SpecDecodeSamplingInputs(
            cu_num_logits=cu_num_logits,
            idx_mapping=idx_mapping,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            draft_sampled=input_ids[logits_indices],
            positions=positions[logits_indices],
            temperature=temperature,
        )

    def _buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        buffer = self._buffers.get(name)
        if (
            buffer is None
            or buffer.dtype != dtype
            or buffer.device != device
            or any(buffer.shape[i] < shape[i] for i in range(len(shape)))
        ):
            buffer = torch.empty(shape, dtype=dtype, device=device)
            self._buffers[name] = buffer
        return buffer[tuple(slice(0, dim) for dim in shape)]

    def _idx_mapping(self, num_reqs: int, device: torch.device) -> torch.Tensor:
        name = "idx_mapping"
        buffer = self._buffers.get(name)
        if buffer is None or buffer.device != device or buffer.shape[0] < num_reqs:
            buffer = torch.arange(num_reqs, dtype=torch.int32, device=device)
            self._buffers[name] = buffer
        return buffer[:num_reqs]


class V1SamplingRandomManager:
    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._stream = torch.npu.Stream(device=device)
        self._event = torch.npu.Event()
        self._buffers: dict[str, torch.Tensor] = {}

    def prepare_regular(
        self,
        num_reqs: int,
        vocab_size: int,
        sampling_metadata: SamplingMetadata,
    ) -> V1SamplingRandoms:
        if sampling_metadata.all_greedy:
            return V1SamplingRandoms()
        sample_gumbel = self._buffer(
            "sample_gumbel", (num_reqs, vocab_size), torch.float32
        )
        self._record_randoms(
            lambda: self._fill_gumbel(sample_gumbel, sampling_metadata.generators)
        )
        return V1SamplingRandoms(sample_gumbel=sample_gumbel, ready_event=self._event)

    def prepare_spec_decode(
        self,
        num_logits: int,
        num_reqs: int,
        vocab_size: int,
        num_draft_tokens: list[int],
        sampling_metadata: SamplingMetadata,
    ) -> V1SamplingRandoms:
        if sampling_metadata.all_greedy:
            return V1SamplingRandoms(
                accept_uniform=self._buffer("accept_uniform", (1,), torch.float32),
                resample_gumbel=self._buffer("resample_gumbel", (1, 1), torch.float32),
            )

        accept_uniform = self._buffer(
            "accept_uniform", (num_logits,), torch.float32
        )
        resample_gumbel = self._buffer(
            "resample_gumbel", (num_reqs, vocab_size), torch.float32
        )

        def fill() -> None:
            accept_uniform.uniform_()
            offset = 0
            for req_idx, num_draft in enumerate(num_draft_tokens):
                num_rows = num_draft + 1
                generator = sampling_metadata.generators.get(req_idx)
                if generator is not None:
                    accept_uniform[offset : offset + num_draft].uniform_(
                        generator=generator
                    )
                offset += num_rows
            accept_uniform.clamp_(min=1e-20)
            self._fill_gumbel(resample_gumbel, sampling_metadata.generators)

        self._record_randoms(fill)
        return V1SamplingRandoms(
            accept_uniform=accept_uniform,
            resample_gumbel=resample_gumbel,
            ready_event=self._event,
        )

    def wait(self, randoms: V1SamplingRandoms | None) -> None:
        if randoms is not None and randoms.ready_event is not None:
            torch.npu.current_stream().wait_event(randoms.ready_event)

    def _buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        buffer = self._buffers.get(name)
        if (
            buffer is None
            or buffer.dtype != dtype
            or buffer.device != self._device
            or any(buffer.shape[i] < shape[i] for i in range(len(shape)))
        ):
            buffer = torch.empty(shape, dtype=dtype, device=self._device)
            self._buffers[name] = buffer
        return buffer[tuple(slice(0, dim) for dim in shape)]

    def _record_randoms(self, fill) -> None:
        current_stream = torch.npu.current_stream()
        self._stream.wait_stream(current_stream)
        with torch.npu.stream(self._stream):
            fill()
            self._event.record()

    @staticmethod
    def _fill_gumbel(
        gumbel: torch.Tensor,
        generators: dict[int, torch.Generator],
    ) -> None:
        gumbel.exponential_()
        for req_idx, generator in generators.items():
            if req_idx < gumbel.shape[0]:
                gumbel[req_idx].exponential_(generator=generator)
        gumbel.log_().neg_()


def temperature_for_sampling(
    sampling_metadata: SamplingMetadata,
    num_reqs: int,
    device: torch.device,
) -> torch.Tensor:
    if sampling_metadata.temperature is None:
        return torch.zeros(num_reqs, dtype=torch.float32, device=device)
    return sampling_metadata.temperature[:num_reqs].to(
        device=device, dtype=torch.float32
    )


def process_regular_logits(
    sampler,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    logits = logits.to(torch.float32)
    logits = sampler.apply_logits_processors(
        logits, sampling_metadata, predict_bonus_token=False
    )
    greedy_sampled = None
    if not sampling_metadata.all_random:
        greedy_sampled = sampler.greedy_sample(logits)
        if sampling_metadata.all_greedy:
            return logits, greedy_sampled
    return _apply_single_step_constraints(logits, sampling_metadata), greedy_sampled


def process_spec_decode_logits(
    sampler,
    rejection_sampler,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    metadata: SpecDecodeMetadata,
) -> torch.Tensor:
    processed = torch.empty_like(logits, dtype=torch.float32)

    target_indices = metadata.target_logits_indices
    if int(target_indices.numel()) > 0:
        target_logits = logits[target_indices].to(torch.float32)
        target_logits = rejection_sampler.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )
        target_logits = _apply_expanded_constraints(
            target_logits, metadata.cu_num_draft_tokens, sampling_metadata
        )
        processed[target_indices] = target_logits

    bonus_indices = metadata.bonus_logits_indices
    bonus_logits = logits[bonus_indices].to(torch.float32)
    bonus_logits = sampler.apply_logits_processors(
        bonus_logits, sampling_metadata, predict_bonus_token=True
    )
    processed[bonus_indices] = _apply_single_step_constraints(
        bonus_logits, sampling_metadata
    )
    return processed


def sample_from_processed_logits(
    processed_logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    sample_gumbel: torch.Tensor | None,
    greedy_sampled: torch.Tensor | None,
) -> torch.Tensor:
    if sampling_metadata.all_greedy:
        if greedy_sampled is None:
            raise RuntimeError("greedy_sampled is required for greedy sampling")
        return greedy_sampled
    if sample_gumbel is None:
        raise RuntimeError("sample_gumbel is required for random sampling")
    sample_gumbel = sample_gumbel[
        : processed_logits.shape[0], : processed_logits.shape[1]
    ]
    random_sampled = (processed_logits + sample_gumbel).argmax(dim=-1).view(-1)
    if sampling_metadata.all_random:
        return random_sampled
    if greedy_sampled is None or sampling_metadata.temperature is None:
        raise RuntimeError("mixed sampling requires greedy tokens and temperature")
    return torch.where(
        sampling_metadata.temperature < _SAMPLING_EPS,
        greedy_sampled,
        random_sampled,
    )


def _apply_single_step_constraints(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    if sampling_metadata.all_greedy:
        return logits
    assert sampling_metadata.temperature is not None
    temperature = sampling_metadata.temperature
    if not sampling_metadata.all_random:
        temperature = torch.where(
            temperature < _SAMPLING_EPS, torch.ones_like(temperature), temperature
        )
    logits.div_(temperature.unsqueeze(-1))
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)
    return _apply_top_k_top_p(logits, sampling_metadata.top_k, sampling_metadata.top_p)


def _apply_expanded_constraints(
    logits: torch.Tensor,
    cu_num_tokens: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    if sampling_metadata.all_greedy:
        return logits
    assert sampling_metadata.temperature is not None
    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_tokens,
        num_tokens,
        replace_from=0,
        replace_to=1,
    )
    logits.div_(temperature.unsqueeze(-1))
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)

    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k, cu_num_tokens, num_tokens
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p, cu_num_tokens, num_tokens
        )
    return _apply_top_k_top_p(logits, top_k, top_p)


def _apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
) -> torch.Tensor:
    logits = apply_top_k_top_p(logits, top_k, top_p)
    if isinstance(logits, tuple):
        raise RuntimeError("reduced sampling logits are not supported here")
    return logits
