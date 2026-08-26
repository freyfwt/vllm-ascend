# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.distributed.eplb import transfer_plan as transfer_plan_module
from vllm_ascend.distributed.eplb.policy.stair_types import (
    StairBalanceScore,
    StairLayerPlan,
    StairSourceMode,
    StairTopology,
)
from vllm_ascend.distributed.eplb.transfer_plan import (
    _decode_executor_sources,
    _source_capacities,
    build_transfer_plan,
    compute_layer_expert_bytes,
    get_active_source_ordering,
    source_ordering_context,
    transfer_layer_with_plan,
)


def _crossed_source_example():
    current = np.array(
        [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
        ],
        dtype=np.int64,
    )
    candidate = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 3],
        ],
        dtype=np.int64,
    )
    topology = StairTopology.from_rank_to_node((0, 1, 1, 0))
    return current, candidate, topology


def test_source_capacities_match_upstream_balanced_fanout():
    assert _source_capacities(3, 8) == (3, 3, 2)
    assert _decode_executor_sources((2, 4, 6), (1, 3, 5, 7, 8, 9, 10, 11)) == {
        1: 2,
        3: 2,
        5: 4,
        7: 4,
        8: 6,
        9: 6,
        10: 2,
        11: 4,
    }


def test_transfer_plan_reorders_sources_to_avoid_cross_node_bytes(monkeypatch):
    current, candidate, topology = _crossed_source_example()
    monkeypatch.setattr(transfer_plan_module, "_SOURCE_ORDERING_PATCH_ENABLED", True)

    plan = build_transfer_plan(current, candidate, topology, expert_bytes=1024)

    assert plan.source_mode is StairSourceMode.PLUGIN_ORDERED
    assert plan.cost.cross_node_bytes == 0
    assert plan.executor_default_cost.cross_node_bytes == 2 * 1024
    assert plan.cost.weighted_bytes < plan.executor_default_cost.weighted_bytes
    assert _decode_executor_sources(plan.send_order(0), plan.recv_order(0)) == {
        2: 1,
        3: 0,
    }


def test_transfer_plan_falls_back_when_ordering_patch_is_disabled(monkeypatch):
    current, candidate, topology = _crossed_source_example()
    monkeypatch.setattr(transfer_plan_module, "_SOURCE_ORDERING_PATCH_ENABLED", False)

    plan = build_transfer_plan(current, candidate, topology, expert_bytes=1024)

    assert plan.source_mode is StairSourceMode.EXECUTOR_DEFAULT
    assert plan.cost.cross_node_bytes == 2 * 1024


def test_source_ordering_context_is_exception_safe(monkeypatch):
    current, candidate, topology = _crossed_source_example()
    monkeypatch.setattr(transfer_plan_module, "_SOURCE_ORDERING_PATCH_ENABLED", True)
    plan = build_transfer_plan(current, candidate, topology, expert_bytes=1)

    with pytest.raises(RuntimeError, match="stop"), source_ordering_context(plan):
        assert get_active_source_ordering() is plan
        raise RuntimeError("stop")

    assert get_active_source_ordering() is None


def test_compute_layer_expert_bytes_includes_every_weight_view():
    weights = [
        [
            torch.empty((2, 3), dtype=torch.float32),
            torch.empty((2, 5), dtype=torch.int8),
        ]
    ]

    assert compute_layer_expert_bytes(weights) == (17,)


def test_compute_layer_expert_bytes_supports_per_expert_tensor_lists():
    class ExpertTensorList(list[torch.Tensor]):
        @property
        def shape(self) -> torch.Size:
            return torch.Size((len(self), *self[0].shape))

    weights = [
        [
            ExpertTensorList(
                [
                    torch.empty(3, dtype=torch.float32),
                    torch.empty(3, dtype=torch.float32),
                ]
            ),
            torch.empty((2, 5), dtype=torch.int8),
        ]
    ]

    assert compute_layer_expert_bytes(weights) == (17,)


def test_transfer_layer_activates_ordering_only_around_upstream_call(monkeypatch):
    current, candidate, topology = _crossed_source_example()
    monkeypatch.setattr(transfer_plan_module, "_SOURCE_ORDERING_PATCH_ENABLED", True)
    transfer_plan = build_transfer_plan(current, candidate, topology, expert_bytes=1)
    score = StairBalanceScore(1.1, 1.1, 1.1)
    layer_plan = StairLayerPlan(
        layer_idx=0,
        placement=candidate,
        current_score=score,
        candidate_score=score,
        balance_gain=0.1,
        utility=0.1,
        transfer_cost=transfer_plan.budget_cost,
        transfer_plan=transfer_plan,
    )
    metadata = object()

    def upstream_transfer(**kwargs):
        del kwargs
        assert get_active_source_ordering() is transfer_plan
        return metadata

    monkeypatch.setattr(transfer_plan_module._rebalance_execute, "transfer_layer", upstream_transfer)

    result, source_mode = transfer_layer_with_plan(
        layer_plan=layer_plan,
        old_layer_indices=torch.from_numpy(current.reshape(-1)),
        new_layer_indices=torch.from_numpy(candidate.reshape(-1)),
        ep_group=SimpleNamespace(size=lambda: 4),
    )

    assert result is metadata
    assert source_mode is StairSourceMode.PLUGIN_ORDERED
    assert get_active_source_ordering() is None
