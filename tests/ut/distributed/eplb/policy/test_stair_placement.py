import numpy as np

from vllm_ascend.distributed.eplb.policy.stair_placement import constrained_lpt, unconstrained_lpt
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology


def test_unconstrained_lpt_balances_duplicate_free_ranks():
    placement = unconstrained_lpt(
        np.array([10.0, 6.0, 2.0, 1.0]),
        np.zeros(4),
        np.array([2, 2, 1, 1]),
        3,
        0.0,
    )
    assert placement.shape == (3, 2)
    assert all(len(set(row)) == 2 for row in placement)
    np.testing.assert_array_equal(np.bincount(placement.ravel()), [2, 2, 1, 1])


def test_constrained_lpt_emits_real_sources_with_pair_cap():
    old = np.array([[0, 1], [2, 3], [0, 2]])
    topology = RankTopology((0, 0, 1), (0, 1, 2))
    result = constrained_lpt(
        np.array([10.0, 6.0, 4.0, 1.0]),
        np.zeros(4),
        np.array([2, 1, 2, 1]),
        old,
        topology,
        z_score=0.0,
        pair_cap=1,
        max_backtracks=8,
    )
    assert result is not None
    pairs = []
    for dst, row in enumerate(result.placement):
        for slot, expert in enumerate(row):
            src, src_slot = result.source_rank[dst, slot], result.source_slot[dst, slot]
            assert old[src, src_slot] == expert
            if src != dst:
                pairs.append((int(src), dst))
    assert len(pairs) == len(set(pairs))
