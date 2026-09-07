import numpy as np

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.policy.stair import plan_rebalance
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology


def _plan(load, *, anchors=(None,), **overrides):
    old = np.array([[[0, 1], [2, 3], [0, 2]]])
    config = StairConfig(
        imbalance_threshold=1.0,
        hysteresis_enabled=False,
        min_relative_score_improvement=0.0,
        p95_regression_tolerance=1.0,
        **overrides,
    )
    return plan_rebalance(
        np.asarray(load, dtype=np.float64).reshape(-1, 1, 4),
        old,
        np.array([5]),
        anchors,
        RankTopology((0, 0, 1), (0, 1, 2)),
        config,
        model_id="model",
        planning_round=2,
        snapshot_sequence=3,
        sample_sequence=6,
        stats_schema_epoch=4,
    )


def test_planner_filters_zero_and_balanced_layers():
    assert not _plan([[0, 0, 0, 0]]).layers
    assert not _plan([[2, 1, 2, 1]]).layers


def test_planner_improves_mean_and_emits_explicit_sources():
    plan = _plan([[100, 30, 10, 1], [80, 40, 10, 1]])
    assert len(plan.layers) == 1
    layer = plan.layers[0]
    assert layer.candidate_score.mean <= layer.current_score.mean
    assert layer.base_placement_epoch == 5
    for dst, row in enumerate(layer.new_placement):
        for slot, expert in enumerate(row):
            assert layer.old_placement[layer.source_rank[dst, slot], layer.source_slot[dst, slot]] == expert
    assert len(plan.digest()) == 64


def test_hysteresis_uses_last_committed_predicted_score():
    config = StairConfig(
        imbalance_threshold=1.0,
        hysteresis_enabled=True,
        hysteresis_relative=0.5,
        hysteresis_absolute=0.5,
        min_relative_score_improvement=0.0,
        p95_regression_tolerance=1.0,
    )
    old = np.array([[[0, 1], [2, 3], [0, 2]]])
    plan = plan_rebalance(
        np.array([[[100, 30, 10, 1]]]),
        old,
        np.array([0]),
        (1.5,),
        RankTopology((0, 0, 1), (0, 1, 2)),
        config,
        model_id="model",
        planning_round=1,
        snapshot_sequence=1,
        sample_sequence=1,
        stats_schema_epoch=1,
    )
    assert not plan.layers
