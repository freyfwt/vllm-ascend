import time

import numpy as np

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.planner_client import PlannerClient, PlannerModel
from vllm_ascend.distributed.eplb.planner_shared_memory import SharedSnapshotShape
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology


def test_planner_client_starts_registers_and_stops_child():
    client = PlannerClient(
        (
            PlannerModel(
                "model",
                SharedSnapshotShape(2, 1, 4, 3, 2),
                RankTopology((0, 0, 1), (0, 1, 2)),
                StairConfig(sample_size=2),
            ),
        )
    )
    process = client._process
    assert client.pid > 0
    assert process.poll() is None
    client.close()
    assert process.poll() == 0


def test_planner_client_submits_and_polls_without_blocking():
    config = StairConfig(
        sample_size=2,
        imbalance_threshold=1.0,
        hysteresis_enabled=False,
        min_relative_score_improvement=0.0,
        p95_regression_tolerance=1.0,
    )
    client = PlannerClient(
        (
            PlannerModel(
                "model",
                SharedSnapshotShape(2, 1, 4, 3, 2),
                RankTopology((0, 0, 1), (0, 1, 2)),
                config,
            ),
        )
    )
    try:
        values = (
            "model",
            np.array([[[100, 30, 10, 1]], [[80, 40, 10, 1]]], dtype=np.int64),
            np.ones(2, dtype=np.int64),
            np.array([[[0, 1], [2, 3], [0, 2]]], dtype=np.int32),
            np.array([0], dtype=np.int64),
            np.array([np.nan]),
        )
        assert client.submit(
            *values,
            planning_round=1,
            snapshot_sequence=2,
            sample_sequence=3,
            stats_schema_epoch=4,
            key_digest="11" * 32,
        )
        assert not client.submit(
            *values,
            planning_round=2,
            snapshot_sequence=3,
            sample_sequence=4,
            stats_schema_epoch=4,
            key_digest="22" * 32,
        )
        deadline = time.monotonic() + 5
        plan = None
        while plan is None and time.monotonic() < deadline:
            plan = client.poll()
            time.sleep(0.01)
        assert plan is not None
        assert (plan.planning_round, plan.snapshot_sequence, plan.sample_sequence) == (1, 2, 3)
    finally:
        client.close()
