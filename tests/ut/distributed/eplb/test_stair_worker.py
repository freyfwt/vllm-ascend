from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from vllm_ascend.distributed.eplb.policy.stair_types import BalanceScore, LayerPlan, RankTopology
from vllm_ascend.distributed.eplb.stair_worker import (
    StairTransferWorker,
    TransferWork,
    transfer_one_layer,
)


class FakeCommunicator:
    def set_stream(self, stream):
        self.stream = stream

    def set_transfer_context(self, old, layer):
        pass

    def add_send(self, tensors, rank, expert):
        pass

    def add_recv(self, tensors, rank, expert):
        pass

    def execute(self):
        pass


def test_transfer_worker_builds_upstream_commit_result():
    placement = np.array([[0, 1]])
    layer = LayerPlan(
        0,
        placement.copy(),
        placement.copy(),
        np.array([[0, 0]]),
        np.array([[0, 1]]),
        0,
        BalanceScore(1, 1, 1),
        BalanceScore(1, 1, 1),
    )
    model_state = SimpleNamespace(
        communicator=FakeCommunicator(),
        model=SimpleNamespace(expert_weights=[[torch.tensor([[1], [2]])]]),
        expert_buffer=[torch.zeros(2, 1, dtype=torch.long)],
    )
    with patch(
        "vllm_ascend.distributed.eplb.stair_worker.CpuGpuEvent",
        return_value=SimpleNamespace(),
    ):
        result = transfer_one_layer(
            TransferWork(model_state, layer, RankTopology((0,), (0,))),
            ep_rank=0,
            stream=None,
        )
    assert result.layer_idx == 0
    torch.testing.assert_close(result.new_physical_to_logical_map, torch.tensor([0, 1]))


def test_worker_close_acknowledges_a_staged_result():
    worker = StairTransferWorker.__new__(StairTransferWorker)
    worker._queue = MagicMock()
    worker._thread = MagicMock()
    worker._thread.is_alive.side_effect = [True, False]
    result = SimpleNamespace(consumed_event=MagicMock())
    state = SimpleNamespace(pending_result=result)
    worker._current = SimpleNamespace(model_state=state)

    worker.close()

    assert state.pending_result is None
    result.consumed_event.record.assert_called_once_with()
