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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.triton_utils import tl, triton
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu.input_batch import InputBatch as GpuInputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler as GpuSampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler as GpuRejectionSampler,
)
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.sample.rejection_ops import (
    RejectionWorkspace,
    rejection_sample,
)

_SAMPLING_EPS = 1e-5
_NPU_BUFFER_OVERRIDES_INSTALLED = False


class _DeviceBackedTensor:
    _device: torch.device = torch.device("cpu")

    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int = 2,
    ) -> None:
        del max_concurrency
        self.dtype = dtype
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()
        self.gpu = torch.zeros(size, dtype=dtype, device=self._device)

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        cls._device = device

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            self.gpu.copy_(self.cpu, non_blocking=True)
            return self.gpu

        self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)
        return self.gpu[:n]


class _DeviceStagedWriteTensor:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int = 2,
        uva_instead_of_gpu: bool = False,
    ) -> None:
        del max_concurrency, uva_instead_of_gpu
        self.num_rows = size if isinstance(size, int) else size[0]
        self.dtype = dtype
        self.device = device
        self.gpu = torch.zeros(size, dtype=dtype, device=device)
        self._staged_write_indices: list[int] = []
        self._staged_write_starts: list[int] = []
        self._staged_write_contents: list[int | float] = []
        self._staged_write_cu_lens: list[int] = []

    def stage_write(
        self,
        index: int,
        start: int,
        x: Iterable[int] | Iterable[float],
    ) -> None:
        assert index >= 0
        assert start >= 0
        values = list(x)
        if not values:
            return
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(start)
        self._staged_write_contents.extend(values)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def stage_write_elem(self, index: int, x: int) -> None:
        assert index >= 0
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(0)
        self._staged_write_contents.append(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def apply_write(self) -> None:
        cu_start = 0
        for index, start, cu_end in zip(
            self._staged_write_indices,
            self._staged_write_starts,
            self._staged_write_cu_lens,
        ):
            values = self._staged_write_contents[cu_start:cu_end]
            value_tensor = torch.tensor(values, dtype=self.dtype, device=self.device)
            if self.gpu.ndim == 1:
                self.gpu[index : index + len(values)].copy_(value_tensor, non_blocking=True)
            else:
                end = start + len(values)
                self.gpu[index, start:end].copy_(value_tensor, non_blocking=True)
            cu_start = cu_end
        self.clear_staged_writes()

    def clear_staged_writes(self) -> None:
        self._staged_write_indices.clear()
        self._staged_write_starts.clear()
        self._staged_write_contents.clear()
        self._staged_write_cu_lens.clear()


def _install_npu_sampling_buffer_overrides(device: torch.device) -> None:
    global _NPU_BUFFER_OVERRIDES_INSTALLED
    if device.type != "npu":
        return

    _DeviceBackedTensor.set_device(device)
    if _NPU_BUFFER_OVERRIDES_INSTALLED:
        return

    from vllm.v1.worker.gpu import states as request_states_module
    from vllm.v1.worker.gpu.sample import bad_words as bad_words_module
    from vllm.v1.worker.gpu.sample import logit_bias as logit_bias_module
    from vllm.v1.worker.gpu.sample import penalties as penalties_module
    from vllm.v1.worker.gpu.sample import states as sampling_states_module

    request_states_module.StagedWriteTensor = _DeviceStagedWriteTensor
    request_states_module.UvaBackedTensor = _DeviceBackedTensor
    sampling_states_module.UvaBackedTensor = _DeviceBackedTensor
    penalties_module.UvaBackedTensor = _DeviceBackedTensor
    bad_words_module.StagedWriteTensor = _DeviceStagedWriteTensor
    bad_words_module.UvaBackedTensor = _DeviceBackedTensor
    logit_bias_module.StagedWriteTensor = _DeviceStagedWriteTensor
    logit_bias_module.UvaBackedTensor = _DeviceBackedTensor
    _NPU_BUFFER_OVERRIDES_INSTALLED = True


@dataclass
class SamplingNoise:
    acceptance_uniform: torch.Tensor | None = None
    sampling_gumbel: torch.Tensor | None = None
    recovery_gumbel: torch.Tensor | None = None
    ready_event: torch.npu.Event | None = None


@triton.jit
def _fill_spec_decode_mapping_kernel(
    cu_num_logits_ptr,
    idx_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    local_pos = tl.arange(0, BLOCK_SIZE)
    offset = start_idx + local_pos
    mask = offset < end_idx
    tl.store(expanded_idx_mapping_ptr + offset, req_state_idx, mask=mask)
    tl.store(expanded_local_pos_ptr + offset, local_pos, mask=mask)


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    idx_mapping_stride,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_rows,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
):
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    req_state_idx = tl.load(idx_mapping_ptr + row_idx * idx_mapping_stride)
    total_len = tl.load(total_len_ptr + req_state_idx)
    num_sampled = tl.load(num_sampled_ptr + row_idx)

    if num_sampled > 0:
        token_id = tl.load(sampled_tokens_ptr + row_idx * sampled_tokens_stride + num_sampled - 1)
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
        tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + row_idx * sampled_tokens_stride + i)
        token_ptr = output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + token_id
        count = tl.load(token_ptr)
        tl.store(token_ptr, count + 1)
        tl.store(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
            token_id,
        )

    query_start = tl.load(query_start_loc_ptr + row_idx)
    query_end = tl.load(query_start_loc_ptr + row_idx + 1)
    num_rejected = tl.load(num_rejected_ptr + row_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + query_end - query_start - num_rejected)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


class GpuBatchView:
    def __init__(self, max_num_reqs: int, device: torch.device) -> None:
        self._max_num_reqs = max_num_reqs
        self._device = device
        self._buffers: dict[str, torch.Tensor] = {}
        self._np_buffers: dict[str, np.ndarray] = {}
        self._regular_num_scheduled_tokens = np.ones(max_num_reqs, dtype=np.int32)
        self._spec_mapping_cache_key: tuple[int, tuple[int, ...], int] | None = None
        self._empty_np = np.empty(0, dtype=np.int32)
        empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        self.input_batch = GpuInputBatch(
            req_ids=[],
            num_reqs=0,
            num_reqs_after_padding=0,
            idx_mapping=empty_i32,
            idx_mapping_np=self._empty_np,
            expanded_idx_mapping=empty_i32,
            expanded_local_pos=empty_i32,
            num_scheduled_tokens=self._empty_np,
            num_tokens=0,
            num_tokens_after_padding=0,
            num_draft_tokens=0,
            query_start_loc=empty_i32,
            query_start_loc_np=self._empty_np,
            seq_lens=empty_i32,
            seq_lens_cpu_upper_bound=torch.empty(0, dtype=torch.int32),
            dcp_local_seq_lens=None,
            input_ids=empty_i32,
            positions=empty_i64,
            logits_indices=empty_i32,
            cu_num_logits=empty_i32,
            cu_num_logits_np=self._empty_np,
            has_structured_output_reqs=False,
        )

    def bind_regular(
        self,
        req_ids: list[str],
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_start_loc_np: np.ndarray,
        seq_lens: torch.Tensor,
        seq_lens_cpu_upper_bound: torch.Tensor,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> GpuInputBatch:
        num_reqs = len(req_ids)
        cu_num_logits = self._arange_buffer("regular_cu_num_logits", num_reqs + 1)
        cu_num_logits_np = self._arange_np("regular_cu_num_logits_np", num_reqs + 1)
        expanded_local_pos = self._zero_buffer("regular_expanded_local_pos", num_reqs)
        self._set_common_fields(
            req_ids=req_ids,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=self._regular_num_scheduled_tokens[:num_reqs],
            num_tokens=int(input_ids.shape[0]),
            num_draft_tokens=0,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
            input_ids=input_ids,
            positions=positions,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
        )
        return self.input_batch

    def bind_spec(
        self,
        req_ids: list[str],
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        metadata: SpecDecodeMetadata,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_start_loc_np: np.ndarray,
        seq_lens: torch.Tensor,
        seq_lens_cpu_upper_bound: torch.Tensor,
        num_speculative_steps: int,
        dcp_local_seq_lens: torch.Tensor | None = None,
        mapping_generation: int | None = None,
    ) -> GpuInputBatch:
        num_reqs = len(metadata.num_draft_tokens)
        num_logits = int(metadata.logits_indices.shape[0])
        draft_token_key = tuple(int(num_draft) for num_draft in metadata.num_draft_tokens)
        cache_key = None if mapping_generation is None else (mapping_generation, draft_token_key, num_logits)

        cu_num_logits = self._buffer("spec_cu_num_logits", (num_reqs + 1,), torch.int32)
        num_logits_per_req = self._np_buffer("spec_num_logits_per_req", num_reqs)
        cu_num_logits_np = self._np_buffer("spec_cu_num_logits_np", num_reqs + 1)

        expanded_idx_mapping = self._buffer(
            "spec_expanded_idx_mapping",
            (num_logits,),
            torch.int32,
        )
        expanded_local_pos = self._buffer(
            "spec_expanded_local_pos",
            (num_logits,),
            torch.int32,
        )
        if cache_key is None or cache_key != self._spec_mapping_cache_key:
            cu_num_logits[0] = 0
            cu_num_logits[1:].copy_(metadata.cu_num_sampled_tokens, non_blocking=True)
            num_logits_per_req[:] = np.asarray(metadata.num_draft_tokens, dtype=np.int32) + 1
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits_per_req, out=cu_num_logits_np[1:])
            block_size = _next_power_of_2(num_speculative_steps + 1)
            _fill_spec_decode_mapping_kernel[(num_reqs,)](
                cu_num_logits,
                idx_mapping,
                expanded_idx_mapping,
                expanded_local_pos,
                BLOCK_SIZE=block_size,
                num_warps=1,
            )
            self._spec_mapping_cache_key = cache_key

        self._set_common_fields(
            req_ids=req_ids,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_logits_per_req,
            num_tokens=int(input_ids.shape[0]),
            num_draft_tokens=int(np.sum(metadata.num_draft_tokens)),
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
            input_ids=input_ids,
            positions=positions,
            logits_indices=metadata.logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
        )
        return self.input_batch

    def num_sampled_ones(self, num_reqs: int) -> torch.Tensor:
        out = self._buffer("num_sampled_ones", (num_reqs,), torch.int32)
        out.fill_(1)
        return out

    def num_rejected_buffer(self, num_reqs: int) -> torch.Tensor:
        return self._buffer("num_rejected", (num_reqs,), torch.int32)

    def _set_common_fields(
        self,
        *,
        req_ids: list[str],
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        num_scheduled_tokens: np.ndarray,
        num_tokens: int,
        num_draft_tokens: int,
        query_start_loc: torch.Tensor,
        query_start_loc_np: np.ndarray,
        seq_lens: torch.Tensor,
        seq_lens_cpu_upper_bound: torch.Tensor,
        dcp_local_seq_lens: torch.Tensor | None,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
    ) -> None:
        batch = self.input_batch
        batch.req_ids = req_ids
        batch.num_reqs = len(req_ids)
        batch.num_reqs_after_padding = len(req_ids)
        batch.idx_mapping = idx_mapping
        batch.idx_mapping_np = idx_mapping_np
        batch.expanded_idx_mapping = expanded_idx_mapping
        batch.expanded_local_pos = expanded_local_pos
        batch.num_scheduled_tokens = num_scheduled_tokens
        batch.num_tokens = num_tokens
        batch.num_tokens_after_padding = num_tokens
        batch.num_draft_tokens = num_draft_tokens
        batch.query_start_loc = query_start_loc
        batch.query_start_loc_np = query_start_loc_np
        batch.seq_lens = seq_lens
        batch.seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound
        batch.dcp_local_seq_lens = dcp_local_seq_lens
        batch.input_ids = input_ids
        batch.positions = positions
        batch.logits_indices = logits_indices
        batch.cu_num_logits = cu_num_logits
        batch.cu_num_logits_np = cu_num_logits_np
        batch.has_structured_output_reqs = False

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

    def _zero_buffer(
        self,
        name: str,
        size: int,
    ) -> torch.Tensor:
        out = self._buffer(name, (size,), torch.int32)
        out.zero_()
        return out

    def _arange_buffer(self, name: str, size: int) -> torch.Tensor:
        out = self._buffer(name, (size,), torch.int32)
        cached_size_name = f"{name}_size"
        if getattr(self, cached_size_name, 0) < size:
            out.copy_(torch.arange(size, dtype=torch.int32, device=self._device))
            setattr(self, cached_size_name, size)
        return out

    def _np_buffer(self, name: str, size: int) -> np.ndarray:
        buffer = self._np_buffers.get(name)
        if buffer is None or buffer.shape[0] < size:
            buffer = np.empty(max(size, self._max_num_reqs + 1), dtype=np.int32)
            self._np_buffers[name] = buffer
        return buffer[:size]

    def _arange_np(self, name: str, size: int) -> np.ndarray:
        out = self._np_buffer(name, size)
        cached_size_name = f"{name}_size"
        if getattr(self, cached_size_name, 0) < size:
            out[:] = np.arange(size, dtype=np.int32)
            setattr(self, cached_size_name, size)
        return out


class SamplingInputBuilder(GpuBatchView):
    def build_regular_inputs(
        self,
        req_ids: list[str],
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
    ) -> GpuInputBatch:
        idx_mapping = self._copy_idx_mapping(idx_mapping_np)
        query_start_loc = self._arange_buffer("compat_regular_query_start_loc", len(req_ids) + 1)
        query_start_loc_np = self._arange_np("compat_regular_query_start_loc_np", len(req_ids) + 1)
        seq_lens = self._zero_buffer("compat_regular_seq_lens", len(req_ids))
        seq_lens_cpu_upper_bound = torch.from_numpy(self._regular_num_scheduled_tokens[: len(req_ids)])
        return self.bind_regular(
            req_ids,
            idx_mapping,
            idx_mapping_np,
            input_ids,
            positions,
            logits_indices,
            query_start_loc,
            query_start_loc_np,
            seq_lens,
            seq_lens_cpu_upper_bound,
        )

    def build_spec_inputs(
        self,
        req_ids: list[str],
        idx_mapping_np: np.ndarray,
        metadata: SpecDecodeMetadata,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        num_speculative_steps: int,
    ) -> GpuInputBatch:
        idx_mapping = self._copy_idx_mapping(idx_mapping_np)
        query_start_loc = self._buffer(
            "compat_spec_query_start_loc",
            (len(req_ids) + 1,),
            torch.int32,
        )
        query_start_loc[0] = 0
        query_start_loc[1:].copy_(metadata.cu_num_sampled_tokens, non_blocking=True)
        query_start_loc_np = self._np_buffer("compat_spec_query_start_loc_np", len(req_ids) + 1)
        query_start_loc_np[0] = 0
        np.cumsum(
            np.asarray(metadata.num_draft_tokens, dtype=np.int32) + 1,
            out=query_start_loc_np[1:],
        )
        seq_lens = self._zero_buffer("compat_spec_seq_lens", len(req_ids))
        seq_lens_cpu_upper_bound = torch.from_numpy(query_start_loc_np[1:].copy())
        return self.bind_spec(
            req_ids,
            idx_mapping,
            idx_mapping_np,
            metadata,
            input_ids,
            positions,
            query_start_loc,
            query_start_loc_np,
            seq_lens,
            seq_lens_cpu_upper_bound,
            num_speculative_steps,
        )

    def _copy_idx_mapping(self, idx_mapping_np: np.ndarray) -> torch.Tensor:
        values = idx_mapping_np.astype(np.int32, copy=False)
        out = self._buffer("compat_idx_mapping", (int(values.shape[0]),), torch.int32)
        out.copy_(torch.from_numpy(values), non_blocking=True)
        return out


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
            return SamplingNoise(
                acceptance_uniform=self._buffer("acceptance_uniform", (1,), torch.float32),
                recovery_gumbel=self._buffer("recovery_gumbel", (1, 1), torch.float32),
            )

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

    def _record_noise(self, fill) -> None:
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


SamplingNoiseManager = NoiseManager


def sample_logits(
    processed_logits: torch.Tensor,
    temperature: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    sampling_gumbel: torch.Tensor | None,
    all_random: bool = False,
    all_greedy: bool = False,
) -> torch.Tensor:
    if sampling_gumbel is not None:
        sampling_gumbel = sampling_gumbel[
            : processed_logits.shape[0],
            : processed_logits.shape[1],
        ]
    if sampling_gumbel is not None and all_random:
        return (processed_logits + sampling_gumbel).argmax(dim=-1).view(-1)

    greedy_tokens = processed_logits.argmax(dim=-1).view(-1)
    if sampling_gumbel is None or all_greedy:
        return greedy_tokens
    random_tokens = (processed_logits + sampling_gumbel).argmax(dim=-1).view(-1)
    row_temperature = temperature[expanded_idx_mapping].to(dtype=torch.float32)
    return torch.where(
        row_temperature < _SAMPLING_EPS,
        greedy_tokens,
        random_tokens,
    )


sample_processed_logits = sample_logits


def apply_params(
    sampler: GpuSampler,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    pos: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    preserve_original_logits: bool,
) -> torch.Tensor:
    if preserve_original_logits:
        # Delegate to the upstream implementation when raw logits must remain
        # available for later consumers such as raw logprobs.
        return sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
        )

    if logits.dtype == torch.float32:
        processed_logits = logits
    else:
        processed_logits = logits.to(torch.float32)

    sampler.logit_bias_state.apply_logit_bias(
        processed_logits,
        expanded_idx_mapping,
        idx_mapping_np,
        pos,
    )
    sampler.penalties_state.apply_penalties(
        processed_logits,
        expanded_idx_mapping,
        idx_mapping_np,
        input_ids,
        expanded_local_pos,
        sampler.num_speculative_tokens,
    )
    sampler.bad_words_state.apply_bad_words(
        processed_logits,
        expanded_idx_mapping,
        idx_mapping_np,
        input_ids,
        expanded_local_pos,
    )
    sampler.sampling_states.apply_temperature(
        processed_logits,
        expanded_idx_mapping,
        idx_mapping_np,
    )
    sampler.sampling_states.apply_min_p(
        processed_logits,
        expanded_idx_mapping,
        idx_mapping_np,
    )
    return sampler.sampling_states.apply_top_k_top_p(
        processed_logits,
        expanded_idx_mapping,
        idx_mapping_np,
    )


apply_sampling_params = apply_params


class NPURejectionSampler(GpuRejectionSampler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._workspace = RejectionWorkspace()

    def sample_with_prefetched_noise(
        self,
        logits: torch.Tensor,
        input_batch: GpuInputBatch,
        acceptance_uniform: torch.Tensor,
        recovery_gumbel: torch.Tensor,
        preserve_original_logits: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rejection_sample_method != "probabilistic":
            raise RuntimeError("fast sampling only supports probabilistic rejection")

        draft_tokens = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
        processed_logits = apply_params(
            self.sampler,
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            draft_tokens,
            input_batch.expanded_local_pos,
            preserve_original_logits,
        )
        return rejection_sample(
            processed_logits,
            draft_tokens,
            None,
            input_batch.cu_num_logits,
            pos,
            input_batch.idx_mapping,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            acceptance_uniform,
            recovery_gumbel,
            self.num_speculative_steps,
            workspace=self._workspace,
        )


AscendGpuRejectionBridge = NPURejectionSampler


class SamplingBridge:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        device: torch.device,
        logprobs_mode: str,
        speculative_config: Any | None,
    ) -> None:
        _install_npu_sampling_buffer_overrides(device)
        num_speculative_steps = speculative_config.num_speculative_tokens if speculative_config is not None else 0
        self.req_states = RequestState(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            num_speculative_steps=num_speculative_steps,
            vocab_size=vocab_size,
            device=device,
        )
        self.sampler = GpuSampler(
            max_num_reqs=max_num_reqs,
            vocab_size=vocab_size,
            device=device,
            req_states=self.req_states,
            logprobs_mode=logprobs_mode,
            num_speculative_tokens=num_speculative_steps + 1,
        )
        self.rejection_sampler: NPURejectionSampler | None = None
        if speculative_config is not None and speculative_config.rejection_sample_method == "probabilistic":
            self.rejection_sampler = NPURejectionSampler(
                self.sampler,
                speculative_config,
                device,
            )
        self._batch_view = GpuBatchView(max_num_reqs, device)
        self._compat_builder = SamplingInputBuilder(max_num_reqs, device)
        self._idx_mapping = torch.empty(max_num_reqs, dtype=torch.int32, device=device)
        self._idx_mapping_np_storage = np.empty(max_num_reqs, dtype=np.int32)
        self._idx_mapping_req_ids: tuple[str, ...] = ()
        self._idx_mapping_generation = 0

    def prepare(
        self,
        input_batch: Any,
        requests: dict[str, Any],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> GpuInputBatch | None:
        if not self.update_requests(input_batch, requests):
            return None

        req_ids = list(input_batch.req_ids)
        idx_mapping_np = self._idx_mapping_np_storage[: len(req_ids)]
        if spec_decode_metadata is None:
            return self._compat_builder.build_regular_inputs(
                req_ids,
                idx_mapping_np,
                input_ids,
                positions,
                logits_indices,
            )

        return self._compat_builder.build_spec_inputs(
            req_ids,
            idx_mapping_np,
            spec_decode_metadata,
            input_ids,
            positions,
            self.req_states.num_speculative_steps,
        )

    def bind_batch(
        self,
        input_batch: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata | None,
        query_start_loc: torch.Tensor | None = None,
        query_start_loc_np: np.ndarray | None = None,
        seq_lens: torch.Tensor | None = None,
        seq_lens_cpu_upper_bound: torch.Tensor | None = None,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> GpuInputBatch:
        req_ids = list(input_batch.req_ids)
        num_reqs = len(req_ids)
        idx_mapping = self._idx_mapping[:num_reqs]
        idx_mapping_np = self._idx_mapping_np_storage[:num_reqs]

        if query_start_loc is None:
            query_start_loc = self._batch_view._arange_buffer(
                "compat_query_start_loc",
                num_reqs + 1,
            )
        if query_start_loc_np is None:
            query_start_loc_np = self._batch_view._arange_np(
                "compat_query_start_loc_np",
                num_reqs + 1,
            )
        if seq_lens is None:
            seq_lens = self._batch_view._zero_buffer("compat_seq_lens", num_reqs)
        if seq_lens_cpu_upper_bound is None:
            seq_lens_cpu_upper_bound = torch.from_numpy(self._batch_view._regular_num_scheduled_tokens[:num_reqs])

        if spec_decode_metadata is None:
            return self._batch_view.bind_regular(
                req_ids,
                idx_mapping,
                idx_mapping_np,
                input_ids,
                positions,
                logits_indices,
                query_start_loc,
                query_start_loc_np,
                seq_lens,
                seq_lens_cpu_upper_bound,
                dcp_local_seq_lens,
            )

        return self._batch_view.bind_spec(
            req_ids,
            idx_mapping,
            idx_mapping_np,
            spec_decode_metadata,
            input_ids,
            positions,
            query_start_loc,
            query_start_loc_np,
            seq_lens,
            seq_lens_cpu_upper_bound,
            self.req_states.num_speculative_steps,
            dcp_local_seq_lens,
            mapping_generation=self._idx_mapping_generation,
        )

    def sample_regular(
        self,
        logits: torch.Tensor,
        input_batch: GpuInputBatch,
        sampling_gumbel: torch.Tensor | None,
        preserve_original_logits: bool = True,
    ) -> SamplerOutput:
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]
        processed_logits = apply_params(
            self.sampler,
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            input_ids,
            input_batch.expanded_local_pos,
            preserve_original_logits,
        )
        sampled = sample_logits(
            processed_logits,
            self.sampler.sampling_states.temperature.gpu,
            input_batch.expanded_idx_mapping,
            sampling_gumbel,
            *self._get_regular_sampling_modes(input_batch.idx_mapping_np),
        )
        num_sampled = (
            self._batch_view.num_sampled_ones(input_batch.num_reqs)
            if hasattr(self, "_batch_view")
            else sampled.new_ones(sampled.shape[0], dtype=torch.int32)
        )
        return SamplerOutput(
            sampled_token_ids=sampled.to(torch.int32).view(-1, 1),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
        )

    def sample_spec(
        self,
        logits: torch.Tensor,
        input_batch: GpuInputBatch,
        acceptance_uniform: torch.Tensor,
        recovery_gumbel: torch.Tensor,
        preserve_original_logits: bool = True,
    ) -> SamplerOutput:
        if self.rejection_sampler is None:
            raise RuntimeError("rejection sampler is required for spec decoding")
        sampled, num_sampled = self.rejection_sampler.sample_with_prefetched_noise(
            logits,
            input_batch,
            acceptance_uniform,
            recovery_gumbel,
            preserve_original_logits,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
        )

    def post_update(
        self,
        input_batch: GpuInputBatch,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
    ) -> None:
        num_reqs = input_batch.num_reqs
        num_rejected = self._batch_view.num_rejected_buffer(num_reqs)
        torch.sub(
            input_batch.query_start_loc[1 : num_reqs + 1],
            input_batch.query_start_loc[:num_reqs],
            out=num_rejected,
        )
        num_rejected.sub_(num_sampled[:num_reqs].to(dtype=torch.int32))

        _post_update_kernel[(num_reqs,)](
            input_batch.idx_mapping,
            input_batch.idx_mapping.stride(0),
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            self.sampler.penalties_state.output_bin_counts,
            self.sampler.penalties_state.output_bin_counts.stride(0),
            sampled_token_ids,
            sampled_token_ids.stride(0),
            num_reqs,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
            self.req_states.all_token_ids.gpu,
            self.req_states.all_token_ids.gpu.stride(0),
            self.req_states.total_len.gpu,
            num_warps=1,
        )

    def reset(self) -> None:
        for req_id in list(self.req_states.req_id_to_index):
            self.req_states.remove_request(req_id)
        self._idx_mapping_req_ids = ()
        self._idx_mapping_generation += 1

    def _get_regular_sampling_modes(
        self,
        idx_mapping_np: np.ndarray,
    ) -> tuple[bool, bool]:
        temperature_np = self.sampler.sampling_states.temperature.np[idx_mapping_np]
        all_random = bool(np.all(temperature_np >= _SAMPLING_EPS))
        all_greedy = bool(np.all(temperature_np < _SAMPLING_EPS))
        return all_random, all_greedy

    def update_requests(
        self,
        input_batch: Any,
        requests: dict[str, Any],
    ) -> bool:
        active_req_ids = tuple(input_batch.req_ids)
        active_req_id_set = set(active_req_ids)
        changed = False
        for req_id in list(self.req_states.req_id_to_index):
            if req_id not in active_req_id_set:
                changed |= self.req_states.remove_request(req_id)

        for req_id in active_req_ids:
            if req_id in self.req_states.req_id_to_index:
                continue
            request = requests.get(req_id)
            if request is None or request.sampling_params is None:
                return False
            req_index = input_batch.req_id_to_index[req_id]
            prompt_len = int(input_batch.num_prompt_tokens[req_index])
            total_len = int(input_batch.num_tokens_no_spec[req_index])
            if not self._has_all_token_ids(input_batch, req_index, total_len):
                return False
            all_token_ids = input_batch.token_ids_cpu[
                req_index,
                :total_len,
            ].tolist()
            num_computed_tokens = min(
                int(input_batch.num_computed_tokens_cpu[req_index]),
                total_len,
            )
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=all_token_ids,
                num_computed_tokens=num_computed_tokens,
            )
            req_state_idx = self.req_states.req_id_to_index[req_id]
            self.sampler.add_request(
                req_state_idx,
                prompt_len,
                request.sampling_params,
            )
            changed = True

        if changed:
            self.req_states.apply_staged_writes()
            self.sampler.apply_staged_writes()
            self._idx_mapping_req_ids = ()
        self._refresh_idx_mapping(active_req_ids)
        return True

    def _sync_request_states(
        self,
        input_batch: Any,
        requests: dict[str, Any],
    ) -> bool:
        return self.update_requests(input_batch, requests)

    def _refresh_idx_mapping(self, req_ids: tuple[str, ...]) -> None:
        if req_ids == self._idx_mapping_req_ids:
            return
        num_reqs = len(req_ids)
        for i, req_id in enumerate(req_ids):
            self._idx_mapping_np_storage[i] = self.req_states.req_id_to_index[req_id]
        if num_reqs:
            self._idx_mapping[:num_reqs].copy_(
                torch.from_numpy(self._idx_mapping_np_storage[:num_reqs]),
                non_blocking=True,
            )
        self._idx_mapping_req_ids = req_ids
        self._idx_mapping_generation += 1

    @staticmethod
    def _has_all_token_ids(
        input_batch: Any,
        req_index: int,
        total_len: int,
    ) -> bool:
        is_token_ids = getattr(input_batch, "is_token_ids", None)
        if is_token_ids is None:
            return True
        return bool(is_token_ids[req_index, :total_len].all())
