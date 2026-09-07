import numpy as np

from vllm_ascend.distributed.eplb.planner_wire import (
    PlanRequest,
    decode_plan,
    decode_request,
    encode_plan,
    encode_request,
)
from vllm_ascend.distributed.eplb.policy.stair_types import BalanceScore, LayerPlan, RebalancePlan


def test_plan_request_round_trip():
    request = PlanRequest("model", 1, 2, 3, 4, 5, 6, 7, "11" * 32, "22" * 32, "33" * 32)
    assert decode_request(encode_request(request)) == request


def test_plan_result_round_trip_and_digest():
    placement = np.array([[0, 1], [2, 3]], dtype=np.int32)
    layer = LayerPlan(
        0,
        placement.copy(),
        placement.copy(),
        np.array([[0, 0], [1, 1]], dtype=np.int32),
        np.array([[0, 1], [0, 1]], dtype=np.int32),
        8,
        BalanceScore(1.2, 1.3, 1.4),
        BalanceScore(1.0, 1.1, 1.2),
    )
    plan = RebalancePlan("model", 1, 2, 3, 4, "44" * 32, "55" * 32, (layer,))
    decoded = decode_plan(encode_plan(plan), (2, 2))
    assert decoded.digest() == plan.digest()
    np.testing.assert_array_equal(decoded.layers[0].source_slot, layer.source_slot)
