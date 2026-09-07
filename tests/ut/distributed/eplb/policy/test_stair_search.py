import numpy as np

from vllm_ascend.distributed.eplb.policy.stair_search import capped_min_max, replica_candidates


def test_capped_min_max_is_stable_and_respects_rank_cap():
    result = capped_min_max(np.array([9.0, 4.0, 1.0]), np.ones(3), 3, 2)
    np.testing.assert_array_equal(result, [2, 2, 2])
    assert capped_min_max(np.ones(2), np.array([2, 2]), 1, 2) is None
    with np.testing.assert_raises(ValueError):
        capped_min_max(np.ones(2), np.array([0, 1]), 1, 2)


def test_replica_search_returns_only_complete_bounded_vectors():
    risk = np.array([12.0, 7.0, 3.0, 1.0])
    candidates = replica_candidates(
        risk,
        7,
        3,
        depth=4,
        width=2,
        max_candidates=8,
        screening_score=lambda counts: float(np.max(risk / counts)),
    )
    assert candidates
    assert len(candidates) <= 8
    assert len({tuple(value) for value in candidates}) == len(candidates)
    assert all(value.sum() == 7 and value.min() >= 1 and value.max() <= 3 for value in candidates)


def test_replica_search_is_deterministic():
    kwargs = dict(
        total_slots=6,
        num_ranks=2,
        depth=3,
        width=1,
        max_candidates=4,
        screening_score=lambda counts: float(np.max(np.array([8.0, 4.0, 2.0, 1.0]) / counts)),
    )
    first = replica_candidates(np.array([8.0, 4.0, 2.0, 1.0]), **kwargs)
    second = replica_candidates(np.array([8.0, 4.0, 2.0, 1.0]), **kwargs)
    assert [tuple(value) for value in first] == [tuple(value) for value in second]
