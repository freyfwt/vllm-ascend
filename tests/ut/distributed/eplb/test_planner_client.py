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
