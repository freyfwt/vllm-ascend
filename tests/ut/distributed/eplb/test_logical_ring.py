import torch

from vllm_ascend.distributed.eplb.logical_ring import LogicalLoadRing


def test_ring_uses_mapping_from_each_recorded_step():
    ring = LogicalLoadRing(3, 1, 2, torch.device("cpu"))
    ring.record(torch.tensor([[2, 3, 5]]), torch.tensor([[0, 1, 0]]), 10)
    ring.record(torch.tensor([[2, 3, 5]]), torch.tensor([[1, 0, 1]]), 11)
    torch.testing.assert_close(ring.chronological(), torch.tensor([[[7, 3]], [[3, 7]]]))
    assert ring.chronological_keys() == (10, 11)


def test_ring_wraps_in_chronological_order_and_keeps_remainder():
    ring = LogicalLoadRing(3, 1, 1, torch.device("cpu"))
    for step in range(5):
        ring.record(torch.tensor([[step]]), torch.tensor([[0]]), step)
    torch.testing.assert_close(ring.chronological().flatten(), torch.tensor([2, 3, 4]))
    assert ring.chronological_keys() == (2, 3, 4)
    sums, lengths = ring.compressed_sums(2)
    assert lengths == (1, 2)
    torch.testing.assert_close(sums.flatten(), torch.tensor([2, 7]))
    assert len(ring.key_digest()) == 64


def test_ring_clear_preserves_allocation():
    ring = LogicalLoadRing(2, 1, 1, torch.device("cpu"))
    storage = ring.values
    ring.record(torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.long), 1)
    ring.clear()
    assert ring.values is storage
    assert ring.valid_size == 0


def test_ring_tracks_model_local_execution_phase():
    ring = LogicalLoadRing(2, 1, 1, torch.device("cpu"))
    ring.record(torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.long), 1, executed=False, has_prefill=True)
    ring.record(torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.long), 2, executed=True, has_prefill=True)
    assert ring.chronological_metadata() == ((False, False), (True, True))
