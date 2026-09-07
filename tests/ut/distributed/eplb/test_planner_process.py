import threading
from multiprocessing import Pipe

import numpy as np

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.planner_process import PlannerServer
from vllm_ascend.distributed.eplb.planner_protocol import (
    PlannerRegistration,
    WireOp,
    encode_registration,
    receive_frame,
    send_frame,
)
from vllm_ascend.distributed.eplb.planner_shared_memory import SharedSnapshotBuffer, SharedSnapshotShape
from vllm_ascend.distributed.eplb.planner_wire import PlanRequest, decode_plan, encode_request
from vllm_ascend.distributed.eplb.policy.stair import config_digest
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology


def test_planner_server_reads_shared_snapshot_and_returns_typed_plan():
    parent, child = Pipe()
    shared = SharedSnapshotBuffer.create(SharedSnapshotShape(2, 1, 4, 3, 2))
    config = StairConfig(
        sample_size=2,
        imbalance_threshold=1.0,
        hysteresis_enabled=False,
        min_relative_score_improvement=0.0,
        p95_regression_tolerance=1.0,
    )
    registration = PlannerRegistration("model", shared.spec, RankTopology((0, 0, 1), (0, 1, 2)), config)
    server = PlannerServer(child)
    thread = threading.Thread(target=server.run)
    thread.start()
    try:
        send_frame(parent, WireOp.INITIALIZE, encode_registration(registration))
        assert receive_frame(parent) == (WireOp.ACK, b"model")
        slot, sequence, digest = shared.write(
            np.array([[[100, 30, 10, 1]], [[80, 40, 10, 1]]], dtype=np.int64),
            np.ones(2, dtype=np.int64),
            np.array([[[0, 1], [2, 3], [0, 2]]], dtype=np.int32),
            np.array([5], dtype=np.int64),
            np.array([np.nan]),
        )
        request = PlanRequest("model", slot, 2, sequence, 1, 2, 3, 4, digest, "11" * 32, config_digest(config))
        send_frame(parent, WireOp.PLAN, encode_request(request))
        operation, payload = receive_frame(parent)
        assert operation == WireOp.RESULT
        plan = decode_plan(payload, (3, 2))
        assert plan.model_id == "model"
        assert plan.config_digest == config_digest(config)
        send_frame(parent, WireOp.SHUTDOWN)
        assert receive_frame(parent) == (WireOp.ACK, b"shutdown")
    finally:
        thread.join(timeout=5)
        parent.close()
        shared.close()
