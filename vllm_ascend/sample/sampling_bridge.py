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
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu.input_batch import InputBatch as GpuInputBatch
from vllm.v1.worker.gpu.sample.sampler import Sampler as GpuSampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler as GpuRejectionSampler,
)
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.sample.rejection_ops import sample_with_rejection

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


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


class SamplingInputBuilder:
    def __init__(self) -> None:
        self._buffers: dict[str, torch.Tensor] = {}

    def build_regular_inputs(
        self,
        req_ids: list[str],
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
    ) -> GpuInputBatch:
        num_reqs = len(req_ids)
        device = input_ids.device
        idx_mapping = self._copy_idx_mapping(idx_mapping_np, device)
        expanded_local_pos = self._zero_buffer("regular_expanded_local_pos", num_reqs, device)
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
        cu_num_logits = self._copy_int32_np("regular_cu_num_logits", cu_num_logits_np, device)

        return GpuInputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
            num_tokens=int(input_ids.shape[0]),
            num_tokens_after_padding=int(input_ids.shape[0]),
            num_draft_tokens=0,
            query_start_loc=cu_num_logits,
            query_start_loc_np=cu_num_logits_np,
            seq_lens=expanded_local_pos,
            seq_lens_cpu_upper_bound=torch.from_numpy(np.ones(num_reqs, dtype=np.int32)),
            dcp_local_seq_lens=None,
            input_ids=input_ids,
            positions=positions,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
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
        num_reqs = len(metadata.num_draft_tokens)
        num_logits = int(metadata.logits_indices.shape[0])
        device = input_ids.device

        cu_num_logits = self._buffer("cu_num_logits", (num_reqs + 1,), torch.int32, device)
        cu_num_logits[0] = 0
        cu_num_logits[1:].copy_(metadata.cu_num_sampled_tokens)
        num_logits_per_req = np.asarray(metadata.num_draft_tokens, dtype=np.int32) + 1
        cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
        cu_num_logits_np[0] = 0
        np.cumsum(num_logits_per_req, out=cu_num_logits_np[1:])

        idx_mapping = self._copy_idx_mapping(idx_mapping_np, device)
        expanded_idx_mapping = self._buffer("expanded_idx_mapping", (num_logits,), torch.int32, device)
        expanded_local_pos = self._buffer("expanded_local_pos", (num_logits,), torch.int32, device)
        block_size = _next_power_of_2(num_speculative_steps + 1)
        _fill_spec_decode_mapping_kernel[(num_reqs,)](
            cu_num_logits,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            BLOCK_SIZE=block_size,
            num_warps=1,
        )

        return GpuInputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_logits_per_req,
            num_tokens=int(input_ids.shape[0]),
            num_tokens_after_padding=int(input_ids.shape[0]),
            num_draft_tokens=int(np.sum(metadata.num_draft_tokens)),
            query_start_loc=cu_num_logits,
            query_start_loc_np=cu_num_logits_np,
            seq_lens=self._zero_buffer("spec_seq_lens", num_reqs, device),
            seq_lens_cpu_upper_bound=torch.from_numpy(num_logits_per_req.copy()),
            dcp_local_seq_lens=None,
            input_ids=input_ids,
            positions=positions,
            logits_indices=metadata.logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=False,
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

    def _copy_idx_mapping(self, idx_mapping_np: np.ndarray, device: torch.device) -> torch.Tensor:
        return self._copy_int32_np("idx_mapping", idx_mapping_np, device)

    def _copy_int32_np(
        self,
        name: str,
        values: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        values = values.astype(np.int32, copy=False)
        out = self._buffer(name, (int(values.shape[0]),), torch.int32, device)
        out.copy_(torch.from_numpy(values), non_blocking=True)
        return out

    def _zero_buffer(
        self,
        name: str,
        size: int,
        device: torch.device,
    ) -> torch.Tensor:
        out = self._buffer(name, (size,), torch.int32, device)
        out.zero_()
        return out


class SamplingNoiseManager:
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


def sample_processed_logits(
    processed_logits: torch.Tensor,
    temperature: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    sampling_gumbel: torch.Tensor | None,
) -> torch.Tensor:
    greedy_tokens = processed_logits.argmax(dim=-1).view(-1)
    if sampling_gumbel is None:
        return greedy_tokens
    sampling_gumbel = sampling_gumbel[
        : processed_logits.shape[0],
        : processed_logits.shape[1],
    ]
    random_tokens = (processed_logits + sampling_gumbel).argmax(dim=-1).view(-1)
    row_temperature = temperature[expanded_idx_mapping].to(dtype=torch.float32)
    return torch.where(
        row_temperature < _SAMPLING_EPS,
        greedy_tokens,
        random_tokens,
    )


class AscendGpuRejectionBridge(GpuRejectionSampler):
    def sample_with_prefetched_noise(
        self,
        logits: torch.Tensor,
        input_batch: GpuInputBatch,
        acceptance_uniform: torch.Tensor,
        recovery_gumbel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rejection_sample_method != "probabilistic":
            raise RuntimeError("fast sampling only supports probabilistic rejection")

        draft_tokens = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            draft_tokens,
            input_batch.expanded_local_pos,
        )
        return sample_with_rejection(
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
        )


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
        self.rejection_sampler: AscendGpuRejectionBridge | None = None
        if speculative_config is not None and speculative_config.rejection_sample_method == "probabilistic":
            self.rejection_sampler = AscendGpuRejectionBridge(
                self.sampler,
                speculative_config,
                device,
            )
        self._builder = SamplingInputBuilder()

    def prepare(
        self,
        input_batch: Any,
        requests: dict[str, Any],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> GpuInputBatch | None:
        if not self._sync_request_states(input_batch, requests):
            return None

        req_ids = list(input_batch.req_ids)
        idx_mapping_np = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=len(req_ids),
        )
        if spec_decode_metadata is None:
            return self._builder.build_regular_inputs(
                req_ids,
                idx_mapping_np,
                input_ids,
                positions,
                logits_indices,
            )

        return self._builder.build_spec_inputs(
            req_ids,
            idx_mapping_np,
            spec_decode_metadata,
            input_ids,
            positions,
            self.req_states.num_speculative_steps,
        )

    def sample_regular(
        self,
        logits: torch.Tensor,
        input_batch: GpuInputBatch,
        sampling_gumbel: torch.Tensor | None,
    ) -> SamplerOutput:
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            input_ids,
            input_batch.expanded_local_pos,
        )
        sampled = sample_processed_logits(
            processed_logits,
            self.sampler.sampling_states.temperature.gpu,
            input_batch.expanded_idx_mapping,
            sampling_gumbel,
        )
        return SamplerOutput(
            sampled_token_ids=sampled.to(torch.int32).view(-1, 1),
            logprobs_tensors=None,
        )

    def sample_spec(
        self,
        logits: torch.Tensor,
        input_batch: GpuInputBatch,
        acceptance_uniform: torch.Tensor,
        recovery_gumbel: torch.Tensor,
    ) -> SamplerOutput:
        if self.rejection_sampler is None:
            raise RuntimeError("rejection sampler is required for spec decoding")
        sampled, _ = self.rejection_sampler.sample_with_prefetched_noise(
            logits,
            input_batch,
            acceptance_uniform,
            recovery_gumbel,
        )
        return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None)

    def _sync_request_states(
        self,
        input_batch: Any,
        requests: dict[str, Any],
    ) -> bool:
        for req_id in list(self.req_states.req_id_to_index):
            self.req_states.remove_request(req_id)

        for req_id in input_batch.req_ids:
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

        self.req_states.apply_staged_writes()
        self.sampler.apply_staged_writes()
        return True

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
