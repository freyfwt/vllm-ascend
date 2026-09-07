import pytest
import torch

from vllm_ascend.distributed.eplb.logical_ring import LogicalLoadRing
from vllm_ascend.distributed.eplb.stair_snapshot import phase_mask, sync_snapshot


def test_phase_mask_uses_only_ranks_that_executed_the_model():
    metadata = torch.tensor(
        [
            [[1, 0], [0, 1], [1, 1], [0, 0]],
            [[0, 1], [1, 0], [1, 0], [0, 0]],
        ]
    )
    torch.testing.assert_close(phase_mask(metadata, "all"), torch.tensor([True, True, True, False]))
    torch.testing.assert_close(phase_mask(metadata, "prefill"), torch.tensor([False, False, True, False]))
    torch.testing.assert_close(phase_mask(metadata, "decode"), torch.tensor([True, True, False, False]))


def test_phase_mask_rejects_unknown_phase():
    with pytest.raises(ValueError, match="Unsupported"):
        phase_mask(torch.zeros(1, 1, 2), "idle")


def test_snapshot_resynchronizes_mismatched_ring_progress(monkeypatch):
    ring = LogicalLoadRing(2, 1, 1, torch.device("cpu"))
    ring.record(torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.long), 1)
    group = type("Group", (), {"size": lambda self: 2})()

    def all_gather(outputs, local, **_kwargs):
        outputs[0].copy_(local)
        outputs[1].copy_(torch.tensor((0, 3)))

    monkeypatch.setattr("vllm_ascend.distributed.eplb.stair_snapshot.all_gather", all_gather)
    result = sync_snapshot(
        ring,
        phase="all",
        sample_size=2,
        cpu_group=group,
        device_group=group,
        group_rank=0,
    )
    assert result is None
    assert ring.valid_size == 0
    assert ring.sample_sequence == 3
