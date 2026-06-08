# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.sample.rejection_ops import (
    RejectionWorkspace,
    rejection_sample,
)
from vllm_ascend.sample.rejection_sampler import (
    apply_sampling_constraints as apply_rejection_sampling_constraints,
)
from vllm_ascend.sample.sampler import (
    AscendSampler,
)
from vllm_ascend.sample.sampler import (
    apply_top_k_top_p as npu_apply_top_k_top_p,
)

_SAMPLING_EPS = 1e-5


def _buffer_slice(
    buffers: dict[str, torch.Tensor],
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    buffer = buffers.get(name)
    if (
        buffer is None
        or buffer.dtype != dtype
        or buffer.device != device
        or any(buffer.shape[i] < shape[i] for i in range(len(shape)))
    ):
        buffer = torch.empty(shape, dtype=dtype, device=device)
        buffers[name] = buffer
    return buffer[tuple(slice(0, dim) for dim in shape)]


@dataclass
class SamplingNoise:
    acceptance_uniform: torch.Tensor | None = None
    sampling_gumbel: torch.Tensor | None = None
    recovery_gumbel: torch.Tensor | None = None
    ready_event: torch.npu.Event | None = None


@dataclass
class SpecSamplingTensors:
    cu_num_logits: torch.Tensor
    idx_mapping: torch.Tensor
    expanded_idx_mapping: torch.Tensor
    expanded_local_pos: torch.Tensor


class NoiseManager:
    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._stream = torch.npu.Stream(device=device)
        self._event = torch.npu.Event()
        self._buffers: dict[str, torch.Tensor] = {}

    def prepare_regular_noise(
        self,
        num_reqs: int,
        vocab_size: int,
        sampling_metadata: SamplingMetadata,
    ) -> SamplingNoise:
        if sampling_metadata.all_greedy:
            return SamplingNoise()
        sampling_gumbel = self._buffer("sampling_gumbel", (num_reqs, vocab_size), torch.float32)
        self._record_noise(lambda: self._fill_gumbel(sampling_gumbel, sampling_metadata.generators))
        return SamplingNoise(sampling_gumbel=sampling_gumbel, ready_event=self._event)

    def prepare_spec_noise(
        self,
        num_logits: int,
        num_reqs: int,
        vocab_size: int,
        num_draft_tokens: list[int],
        sampling_metadata: SamplingMetadata,
    ) -> SamplingNoise:
        if sampling_metadata.all_greedy:
            return SamplingNoise()

        acceptance_uniform = self._buffer("acceptance_uniform", (num_logits,), torch.float32)
        recovery_gumbel = self._buffer("recovery_gumbel", (num_reqs, vocab_size), torch.float32)

        def fill() -> None:
            acceptance_uniform.uniform_()
            offset = 0
            for req_idx, num_draft in enumerate(num_draft_tokens):
                num_rows = num_draft + 1
                generator = sampling_metadata.generators.get(req_idx)
                if generator is not None:
                    acceptance_uniform[offset : offset + num_draft].uniform_(generator=generator)
                offset += num_rows
            acceptance_uniform.clamp_(min=1e-20)
            self._fill_gumbel(recovery_gumbel, sampling_metadata.generators)

        self._record_noise(fill)
        return SamplingNoise(
            acceptance_uniform=acceptance_uniform,
            recovery_gumbel=recovery_gumbel,
            ready_event=self._event,
        )

    def wait(self, noise: SamplingNoise | None) -> None:
        if noise is not None and noise.ready_event is not None:
            torch.npu.current_stream().wait_event(noise.ready_event)

    def _buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return _buffer_slice(self._buffers, name, shape, dtype, self._device)

    def _record_noise(self, fill) -> None:
        current_stream = torch.npu.current_stream()
        # Noise buffers are reused across iterations; do not overwrite them
        # while the current stream may still be reading the previous contents.
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


def apply_regular_sampling_params(
    logits: torch.Tensor,
    sampler: AscendSampler,
    sampling_metadata: SamplingMetadata,
    *,
    predict_bonus_token: bool = False,
) -> torch.Tensor:
    logits = logits.to(torch.float32)
    logits = sampler.apply_logits_processors(
        logits,
        sampling_metadata,
        predict_bonus_token,
    )
    if sampling_metadata.all_greedy:
        return logits

    assert sampling_metadata.temperature is not None
    logits = sampler.apply_temperature(
        logits,
        sampling_metadata.temperature,
        sampling_metadata.all_random,
    )
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)
    return npu_apply_top_k_top_p(
        logits,
        sampling_metadata.top_k,
        sampling_metadata.top_p,
    )


def sample_logits(
    processed_logits: torch.Tensor,
    temperature: torch.Tensor,
    sampling_gumbel: torch.Tensor | None,
    all_random: bool = False,
    all_greedy: bool = False,
    add_gumbel_inplace: bool = False,
) -> torch.Tensor:
    if sampling_gumbel is not None:
        sampling_gumbel = sampling_gumbel[
            : processed_logits.shape[0],
            : processed_logits.shape[1],
        ]
    if sampling_gumbel is not None and all_random:
        if add_gumbel_inplace and processed_logits.dtype == sampling_gumbel.dtype:
            return processed_logits.add_(sampling_gumbel).argmax(dim=-1).view(-1)
        return (processed_logits + sampling_gumbel).argmax(dim=-1).view(-1)

    greedy_tokens = processed_logits.argmax(dim=-1).view(-1)
    if sampling_gumbel is None or all_greedy:
        return greedy_tokens
    if add_gumbel_inplace and processed_logits.dtype == sampling_gumbel.dtype:
        random_tokens = processed_logits.add_(sampling_gumbel).argmax(dim=-1).view(-1)
    else:
        random_tokens = (processed_logits + sampling_gumbel).argmax(dim=-1).view(-1)
    return torch.where(
        temperature.to(dtype=torch.float32) < _SAMPLING_EPS,
        greedy_tokens,
        random_tokens,
    )


@triton.jit
def _expand_idx_mapping_kernel(
    idx_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_tokens
    req_idx_value = tl.load(idx_mapping_ptr + req_idx)
    tl.store(expanded_idx_mapping_ptr + start_idx + block, req_idx_value, mask=mask)
    tl.store(expanded_local_pos_ptr + start_idx + block, block, mask=mask)


class FastSampler:
    def __init__(
        self,
        max_num_reqs: int,
        device: torch.device,
        speculative_config: Any | None,
    ) -> None:
        self._max_num_reqs = max_num_reqs
        self._device = device
        self._buffers: dict[str, torch.Tensor] = {}
        self._arange_sizes: dict[str, int] = {}
        self._workspace = RejectionWorkspace()
        self._num_speculative_steps = speculative_config.num_speculative_tokens if speculative_config is not None else 0
        self._dummy_uniform = torch.empty(1, dtype=torch.float32, device=device)
        self._dummy_gumbel = torch.empty((1, 1), dtype=torch.float32, device=device)

    def sample_regular(
        self,
        logits: torch.Tensor,
        sampler: AscendSampler,
        sampling_metadata: SamplingMetadata,
        sampling_gumbel: torch.Tensor | None,
    ) -> SamplerOutput:
        processed_logits = apply_regular_sampling_params(
            logits,
            sampler,
            sampling_metadata,
        )
        sampled = sample_logits(
            processed_logits,
            sampling_metadata.temperature,
            sampling_gumbel,
            sampling_metadata.all_random,
            sampling_metadata.all_greedy,
            add_gumbel_inplace=True,
        )
        return SamplerOutput(
            sampled_token_ids=sampled.to(torch.int32).view(-1, 1),
            logprobs_tensors=None,
        )

    def sample_spec(
        self,
        logits: torch.Tensor,
        sampler: AscendSampler,
        rejection_sampler: Any,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
        input_ids: torch.Tensor,
        acceptance_uniform: torch.Tensor | None,
        recovery_gumbel: torch.Tensor | None,
    ) -> SamplerOutput:
        processed_logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        target_logits = processed_logits[metadata.target_logits_indices]
        target_logits = rejection_sampler.apply_logits_processors(
            target_logits,
            sampling_metadata,
            metadata,
        )
        target_logits = apply_rejection_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
            getattr(rejection_sampler, "top_k", None),
        )
        if isinstance(target_logits, tuple):
            raise RuntimeError("fast rejection sampling does not support reduced sampling")
        processed_logits[metadata.target_logits_indices] = target_logits

        bonus_logits = processed_logits[metadata.bonus_logits_indices]
        bonus_logits = apply_regular_sampling_params(
            bonus_logits,
            sampler,
            sampling_metadata,
            predict_bonus_token=True,
        )
        processed_logits[metadata.bonus_logits_indices] = bonus_logits

        spec_tensors = self.prepare_spec_tensors(metadata)
        draft_tokens = input_ids[metadata.logits_indices]
        sampled, _num_sampled = rejection_sample(
            processed_logits,
            draft_tokens,
            None,
            spec_tensors.cu_num_logits,
            spec_tensors.idx_mapping,
            spec_tensors.expanded_idx_mapping,
            spec_tensors.expanded_local_pos,
            sampling_metadata.temperature,
            acceptance_uniform if acceptance_uniform is not None else self._dummy_uniform,
            recovery_gumbel if recovery_gumbel is not None else self._dummy_gumbel,
            self._num_speculative_steps,
            workspace=self._workspace,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
        )

    def prepare_spec_tensors(
        self,
        metadata: SpecDecodeMetadata,
    ) -> SpecSamplingTensors:
        num_reqs = len(metadata.num_draft_tokens)
        num_logits = int(metadata.logits_indices.shape[0])
        if num_reqs > self._max_num_reqs:
            raise RuntimeError(f"fast sampler got {num_reqs} requests, exceeding max_num_reqs={self._max_num_reqs}")

        cu_num_logits = self._buffer("cu_num_logits", (num_reqs + 1,), torch.int32)
        cu_num_logits[0].zero_()
        cu_num_logits[1:].copy_(metadata.cu_num_sampled_tokens, non_blocking=True)

        idx_mapping = self._arange_buffer("idx_mapping", num_reqs)
        expanded_idx_mapping = self._buffer("expanded_idx_mapping", (num_logits,), torch.int32)
        expanded_local_pos = self._buffer("expanded_local_pos", (num_logits,), torch.int32)
        block_size = triton.next_power_of_2(max(self._num_speculative_steps + 1, 1))
        _expand_idx_mapping_kernel[(num_reqs,)](
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            cu_num_logits,
            BLOCK_SIZE=block_size,
        )
        return SpecSamplingTensors(
            cu_num_logits=cu_num_logits,
            idx_mapping=idx_mapping,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
        )

    def _buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return _buffer_slice(self._buffers, name, shape, dtype, self._device)

    def _arange_buffer(self, name: str, size: int) -> torch.Tensor:
        out = self._buffer(name, (size,), torch.int32)
        if self._arange_sizes.get(name, 0) < size:
            out.copy_(
                torch.arange(size, dtype=torch.int32, device=self._device),
                non_blocking=True,
            )
            self._arange_sizes[name] = size
        return out
