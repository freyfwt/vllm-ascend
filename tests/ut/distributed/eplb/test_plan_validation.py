import dataclasses

import numpy as np
import pytest

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.plan_validation import validate_plan
from vllm_ascend.distributed.eplb.policy.stair import plan_rebalance
from vllm_ascend.distributed.eplb.policy.stair_types import BalanceScore, LayerPlan, RankTopology


def _values():
    topology = RankTopology((0, 0, 1), (0, 1, 2))
    config = StairConfig(
        imbalance_threshold=1.0,
        hysteresis_enabled=False,
        min_relative_score_improvement=0.0,
        p95_regression_tolerance=1.0,
    )
    placements = np.array([[[0, 1], [2, 3], [0, 2]]])
    plan = plan_rebalance(
        np.array([[[100, 30, 10, 1]], [[80, 40, 10, 1]]]),
        placements,
        np.array([5]),
        (None,),
        topology,
        config,
        model_id="model",
        planning_round=2,
        snapshot_sequence=3,
        sample_sequence=4,
        stats_schema_epoch=6,
    )
    assert plan.layers
    return plan, placements, topology, config


def _validate(plan, placements, topology, config, **overrides):
    values = dict(
        model_id="model",
        current_planning_round=2,
        snapshot_sequence=3,
        sample_sequence=4,
        stats_schema_epoch=6,
    )
    values.update(overrides)
    validate_plan(plan, placements, np.array([5]), topology, config, **values)


def test_parent_validation_accepts_consistent_plan():
    _validate(*_values())


def test_parent_validation_rejects_stale_epoch_and_identity():
    plan, placements, topology, config = _values()
    with pytest.raises(ValueError, match="identity"):
        _validate(plan, placements, topology, config, sample_sequence=5)
    with pytest.raises(ValueError, match="epoch"):
        validate_plan(
            plan,
            placements,
            np.array([6]),
            topology,
            config,
            model_id="model",
            current_planning_round=2,
            snapshot_sequence=3,
            sample_sequence=4,
            stats_schema_epoch=6,
        )


def test_parent_validation_rejects_forged_source():
    plan, placements, topology, config = _values()
    layer = plan.layers[0]
    source_rank = layer.source_rank.copy()
    source_rank[0, 0] = 1
    forged = LayerPlan(
        layer.layer_idx,
        layer.old_placement.copy(),
        layer.new_placement.copy(),
        source_rank,
        layer.source_slot.copy(),
        layer.base_placement_epoch,
        layer.current_score,
        layer.candidate_score,
    )
    with pytest.raises(ValueError, match="source"):
        _validate(dataclasses.replace(plan, layers=(forged,)), placements, topology, config)


def test_parent_validation_recomputes_reported_scores():
    plan, placements, topology, config = _values()
    layer = plan.layers[0]
    forged = dataclasses.replace(
        layer,
        current_score=BalanceScore(
            layer.current_score.mean + 0.1,
            layer.current_score.p95,
            layer.current_score.maximum,
        ),
    )
    load = np.array([[[100, 30, 10, 1]], [[80, 40, 10, 1]]])
    with pytest.raises(ValueError, match="submitted snapshot"):
        _validate(
            dataclasses.replace(plan, layers=(forged,)),
            placements,
            topology,
            config,
            logical_load=load,
            sample_weights=np.ones(2, dtype=np.int64),
        )
