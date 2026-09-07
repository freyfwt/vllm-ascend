import pytest
import torch

from vllm_ascend.distributed.eplb.stair_snapshot import phase_mask


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
