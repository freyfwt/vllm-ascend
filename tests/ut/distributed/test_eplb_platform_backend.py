# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import EPLBConfig, ParallelConfig
from vllm.distributed.eplb.eplb_communicator import EplbCommunicator
from vllm.distributed.eplb.rebalance_execute import move_from_buffer, move_to_buffer
from vllm.distributed.eplb.weight_utils import (
    empty_eplb_weight_like,
    get_eplb_expert_tensor,
)

import vllm_ascend.distributed.eplb_platform_backend as backend_module
from vllm_ascend.distributed.eplb_platform_backend import (
    AscendEplbPlatformBackend,
)


class NoOpCommunicator(EplbCommunicator):
    def add_send(self, tensors, dst_rank, expert_id):
        raise AssertionError("single-rank rearrangement must not send")

    def add_recv(self, tensors, src_rank, expert_id):
        raise AssertionError("single-rank rearrangement must not receive")

    def execute(self):
        pass


def test_parallel_config_discovers_ascend_backend():
    config = ParallelConfig(
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=EPLBConfig(use_async=True),
        distributed_executor_backend="mp",
    )

    assert config.eplb_config.communicator == "platform"


@pytest.mark.parametrize("use_async", [False, True])
def test_sequence_rearrangement_reuses_buffers_on_npu(use_async):
    backend = AscendEplbPlatformBackend()
    runtime = backend.device_runtime
    runtime.set_device(0)
    expert_weight = [
        torch.tensor([1, 2], device="npu"),
        torch.tensor([3, 4], device="npu"),
    ]
    layer_weights = [expert_weight]
    buffers = [empty_eplb_weight_like(expert_weight)]
    stream = runtime.create_stream(0) if use_async else None
    buffer_ptrs = [get_eplb_expert_tensor(buffers[0], expert_id).data_ptr() for expert_id in range(2)]

    old_indices = torch.tensor([0, 1]).numpy()
    new_indices = torch.tensor([1, 0]).numpy()
    for _ in range(2):
        metadata = move_to_buffer(
            num_local_experts=2,
            old_indices=old_indices,
            new_indices=new_indices,
            expert_weights=layer_weights,
            expert_weights_buffers=buffers,
            transfer_stream=stream,
            ep_rank=0,
            communicator=NoOpCommunicator(),
            device_runtime=runtime,
        )
        runtime.synchronize(stream)
        move_from_buffer(
            expert_weights=layer_weights,
            expert_weights_buffers=buffers,
            transfer_metadata=metadata,
            new_indices=new_indices,
            ep_rank=0,
        )
        runtime.synchronize(stream)
        old_indices, new_indices = new_indices, old_indices

    torch.testing.assert_close(expert_weight[0].cpu(), torch.tensor([1, 2]))
    torch.testing.assert_close(expert_weight[1].cpu(), torch.tensor([3, 4]))
    assert [get_eplb_expert_tensor(buffers[0], expert_id).data_ptr() for expert_id in range(2)] == buffer_ptrs


def test_npu_runtime_event_orders_stream_work():
    runtime = AscendEplbPlatformBackend().device_runtime
    runtime.set_device(0)
    stream = runtime.create_stream(0)
    event = runtime.create_event()
    value = torch.zeros(1, device="npu")

    with runtime.stream_context(stream):
        value.fill_(7)
        event.record(stream)
    event.synchronize()

    torch.testing.assert_close(value.cpu(), torch.tensor([7.0]))


def test_npu_map_to_physical(monkeypatch):
    monkeypatch.setattr(
        backend_module,
        "get_ep_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )
    backend = AscendEplbPlatformBackend()
    topk_ids = torch.tensor([[0, 1], [0, 0], [-1, 1]], device="npu")
    logical_map = torch.tensor([[0, 2], [1, -1]], device="npu")
    replica_count = torch.tensor([2, 1], device="npu")

    physical_ids = backend.map_to_physical(
        topk_ids,
        logical_map,
        replica_count,
    )
    backend.device_runtime.synchronize()

    torch.testing.assert_close(
        physical_ids.cpu(),
        torch.tensor([[0, 1], [2, 2], [-1, 1]]),
    )
    assert backend.load_recording_mode == "post_moe"


def test_platform_communicator_creation():
    backend = AscendEplbPlatformBackend()
    coordinator = SimpleNamespace(device_group=MagicMock())

    communicator = backend.create_communicator(coordinator, [], [])

    assert communicator.__class__.__name__ == "HcclEplbCommunicator"
