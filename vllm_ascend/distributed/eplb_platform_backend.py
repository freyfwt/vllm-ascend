# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend implementation of the upstream EPLB Platform Backend contract."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, cast

import torch
import torch_npu
from vllm.distributed.eplb.platform_backend import (
    EplbDeviceEvent,
    EplbDeviceRuntime,
    EplbPlatformBackend,
)

from vllm_ascend.distributed.eplb_communicator import HcclEplbCommunicator
from vllm_ascend.ops.fused_moe import eplb as _eplb_ops  # noqa: F401

if TYPE_CHECKING:
    from vllm.config import ParallelConfig
    from vllm.distributed.eplb.eplb_communicator import EplbCommunicator
    from vllm.distributed.eplb.weight_utils import (
        EplbExpertWeight,
        EplbLayerWeights,
    )
    from vllm.distributed.parallel_state import GroupCoordinator


class AscendEplbDeviceRuntime(EplbDeviceRuntime):
    """Ascend stream and event operations used by upstream EPLB."""

    def get_device_index(self, device: torch.device) -> int:
        if device.index is not None:
            return device.index
        return torch.accelerator.current_device_index()

    def set_device(self, device_index: int) -> None:
        torch.accelerator.set_device_index(device_index)

    def create_stream(self, device_index: int) -> Any:
        return torch_npu.npu.Stream(device=device_index)

    def stream_context(self, stream: Any) -> AbstractContextManager[Any]:
        return torch_npu.npu.stream(cast(torch_npu.npu.Stream, stream))

    def create_event(self, enable_timing: bool = False) -> EplbDeviceEvent:
        return cast(
            EplbDeviceEvent,
            torch_npu.npu.Event(enable_timing=enable_timing),
        )

    def synchronize(self, stream: Any | None = None) -> None:
        if stream is None:
            torch.accelerator.synchronize()
        else:
            stream.synchronize()


class AscendEplbPlatformBackend(EplbPlatformBackend):
    """Provide NPU mapping, HCCL communication, and device primitives."""

    def __init__(self) -> None:
        self._device_runtime = AscendEplbDeviceRuntime()

    @classmethod
    def resolve_communicator(cls, parallel_config: ParallelConfig) -> str:
        return "platform"

    @classmethod
    def validate_config(cls, parallel_config: ParallelConfig) -> None:
        communicator = parallel_config.eplb_config.communicator
        if communicator != "platform":
            raise ValueError(f"Ascend EPLB requires the 'platform' communicator (got {communicator!r}).")
        if parallel_config.enable_elastic_ep:
            raise ValueError("Ascend EPLB does not support Elastic EP.")

    def map_and_record(
        self,
        topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        expert_load_view: torch.Tensor,
        record_enabled: torch.Tensor,
        num_unpadded_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        return torch.ops.vllm.ascend_eplb_map_and_record(
            topk_ids,
            logical_to_physical_map,
            logical_replica_count,
            expert_load_view,
            record_enabled,
            num_unpadded_tokens,
        )

    def create_communicator(
        self,
        group_coordinator: GroupCoordinator,
        expert_weights: Sequence[EplbLayerWeights],
        expert_buffer: Sequence[EplbExpertWeight],
    ) -> EplbCommunicator:
        del expert_weights, expert_buffer
        return HcclEplbCommunicator(group_coordinator.device_group)

    @property
    def device_runtime(self) -> EplbDeviceRuntime:
        return self._device_runtime
