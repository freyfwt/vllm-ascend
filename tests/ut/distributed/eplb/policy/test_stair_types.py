import numpy as np
import pytest

from vllm_ascend.distributed.eplb.policy.stair_types import (
    BalanceScore,
    LayerPlan,
    RankTopology,
    RebalancePlan,
    validate_placement,
)


def test_validate_placement_rejects_duplicates_and_missing_experts():
    validate_placement(np.array([[0, 1], [2, 3]]), 4)
    with pytest.raises(ValueError, match="repeats"):
        validate_placement(np.array([[0, 0], [1, 2]]), 3)
    with pytest.raises(ValueError, match="cover every"):
        validate_placement(np.array([[0, 1], [1, 2]]), 4)


def test_plan_digest_is_discrete_and_deterministic():
    placement = np.array([[0, 1], [2, 3]])
    layer = LayerPlan(
        1,
        placement.copy(),
        placement.copy(),
        np.array([[0, 0], [1, 1]]),
        np.array([[0, 1], [0, 1]]),
        7,
        BalanceScore(1.2, 1.3, 1.4),
        BalanceScore(1.0, 1.0, 1.0),
    )
    plan = RebalancePlan("model", 2, 3, 4, "config", "topology", (layer,))
    changed_telemetry = LayerPlan(
        1,
        placement.copy(),
        placement.copy(),
        np.array([[0, 0], [1, 1]]),
        np.array([[0, 1], [0, 1]]),
        7,
        BalanceScore(9, 9, 9),
        BalanceScore(8, 8, 8),
    )
    assert plan.digest() == RebalancePlan("model", 2, 3, 4, "config", "topology", (changed_telemetry,)).digest()


def test_topology_requires_rank_bijection():
    topology = RankTopology((0, 0, 1, 1), (2, 3, 0, 1))
    assert topology.same_node(0, 1)
    assert not topology.same_node(0, 2)
    assert len(topology.digest()) == 64
    with pytest.raises(ValueError, match="bijection"):
        RankTopology((0, 1), (0, 0))
