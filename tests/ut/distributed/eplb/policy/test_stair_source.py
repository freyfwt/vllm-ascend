import numpy as np

from vllm_ascend.distributed.eplb.policy.stair_source import align_slots, assign_sources
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology


def test_source_matching_respects_directed_pair_cap():
    old = np.array([[0, 1], [2, 3], [0, 2]])
    desired = [{2, 3}, {0, 1}, {0, 2}]
    topology = RankTopology((0, 0, 1), (0, 1, 2))
    sources = assign_sources(old, desired, topology, pair_cap=1)
    assert sources is not None
    pairs = [(src, dst) for (dst, _), (src, _) in sources.items()]
    assert len(pairs) == len(set(pairs))
    assert all(old[src, slot] == expert for (dst, expert), (src, slot) in sources.items())


def test_source_matching_detects_infeasible_pair_capacity():
    old = np.array([[0, 1], [2, 3]])
    desired = [{2, 3}, {0, 1}]
    topology = RankTopology((0, 1), (0, 1))
    assert assign_sources(old, desired, topology, pair_cap=1) is None


def test_slot_alignment_keeps_local_experts_stable():
    old = np.array([[0, 1], [2, 3]])
    desired = [{0, 2}, {1, 3}]
    sources = {(0, 2): (1, 0), (1, 1): (0, 1)}
    new, source_rank, source_slot = align_slots(old, desired, sources)
    np.testing.assert_array_equal(new, [[0, 2], [1, 3]])
    np.testing.assert_array_equal(source_rank, [[0, 1], [0, 1]])
    np.testing.assert_array_equal(source_slot, [[0, 0], [1, 1]])
