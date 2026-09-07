import numpy as np
import torch

from vllm_ascend.distributed.eplb.policy.stair_types import BalanceScore, LayerPlan, RankTopology
from vllm_ascend.distributed.eplb.transfer_adapter import stage_layer


class FakeCommunicator:
    def __init__(self):
        self.sends = []
        self.recvs = []
        self.executed = 0

    def set_stream(self, stream):
        pass

    def set_transfer_context(self, old, layer):
        self.context = (old, layer)

    def add_send(self, tensors, rank, expert):
        self.sends.append((rank, expert, [tensor.clone() for tensor in tensors]))

    def add_recv(self, tensors, rank, expert):
        self.recvs.append((rank, expert, tensors))

    def execute(self):
        self.executed += 1


def _plan():
    return LayerPlan(
        3,
        np.array([[0, 1], [2, 3]]),
        np.array([[0, 2], [1, 3]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        0,
        BalanceScore(1.2, 1.2, 1.2),
        BalanceScore(1.0, 1.0, 1.0),
    )


def test_adapter_consumes_explicit_source_for_sender():
    communicator = FakeCommunicator()
    weights = [torch.tensor([[10], [11]])]
    buffers = [torch.zeros_like(weights[0])]
    metadata = stage_layer(
        _plan(),
        weights,
        buffers,
        communicator,
        RankTopology((0, 1), (1, 0)),
        ep_rank=0,
        cuda_stream=None,
    )
    assert [(rank, expert) for rank, expert, _ in communicator.sends] == [(0, 1)]
    assert communicator.sends[0][2][0].item() == 11
    assert metadata.is_unchanged.tolist() == [True, False]


def test_adapter_posts_receive_into_planned_destination_slot():
    communicator = FakeCommunicator()
    weights = [torch.tensor([[20], [21]])]
    buffers = [torch.zeros_like(weights[0])]
    metadata = stage_layer(
        _plan(),
        weights,
        buffers,
        communicator,
        RankTopology((0, 1), (1, 0)),
        ep_rank=1,
        cuda_stream=None,
    )
    assert [(rank, expert) for rank, expert, _ in communicator.recvs] == [(1, 1)]
    assert communicator.recvs[0][2][0].data_ptr() == buffers[0][0].data_ptr()
    assert metadata.recv_dst_rows[0] == 0
    assert metadata.is_unchanged.tolist() == [False, True]
