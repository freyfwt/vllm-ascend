from multiprocessing import Pipe

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.planner_protocol import (
    PlannerRegistration,
    WireOp,
    decode_registration,
    encode_registration,
    receive_frame,
    send_frame,
)
from vllm_ascend.distributed.eplb.planner_shared_memory import SharedSnapshotBuffer, SharedSnapshotShape
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology


def test_binary_frame_round_trip_without_object_pickling():
    first, second = Pipe()
    send_frame(first, WireOp.PLAN, b"payload")
    assert receive_frame(second) == (WireOp.PLAN, b"payload")
    first.close()
    second.close()


def test_registration_round_trip():
    shared = SharedSnapshotBuffer.create(SharedSnapshotShape(4, 2, 3, 2, 2))
    registration = PlannerRegistration(
        "model",
        shared.spec,
        RankTopology((0, 1), (1, 0)),
        StairConfig(sample_size=4, experimental_flash_tree_width=2),
    )
    try:
        decoded = decode_registration(encode_registration(registration))
        assert decoded.model_id == "model"
        assert decoded.shared == registration.shared
        assert decoded.topology == registration.topology
        assert decoded.config.sample_size == 4
        assert decoded.config.experimental_flash_tree_width == 2
    finally:
        shared.close()
