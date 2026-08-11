# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

import vllm_ascend.distributed.eplb_policy as eplb_policy
from vllm_ascend.distributed.eplb_policy import (
    FlashLBEplbPolicyAdapter,
    StairEplbPolicyAdapter,
    SwiftEplbPolicyAdapter,
    _expand_logical_load_to_slots,
    _expand_logical_load_window_to_slots,
    _reject_invalid_placement_layers,
)
from vllm_ascend.eplb.core.policy.policy_stair import StairEplbPolicy, compute_balance_score
from vllm_ascend.eplb.core.policy.policy_swift_balancer import SwiftBalanceEplb


def test_expand_logical_load_to_slots_preserves_logical_load():
    logical_load = torch.tensor([[10, 20, 30, 40]], dtype=torch.int32)
    placement = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long)

    slot_load = _expand_logical_load_to_slots(logical_load, placement)
    reconstructed = torch.zeros_like(logical_load, dtype=slot_load.dtype)
    reconstructed.scatter_add_(1, placement, slot_load)

    torch.testing.assert_close(reconstructed, logical_load.to(torch.float64))


def test_expand_logical_load_window_to_slots_preserves_each_sample():
    logical_load_window = torch.tensor(
        [[[10, 20, 30, 40]], [[50, 60, 70, 80]]],
        dtype=torch.int32,
    )
    placement = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long)

    slot_load_window = _expand_logical_load_window_to_slots(logical_load_window, placement)
    reconstructed = torch.zeros_like(logical_load_window, dtype=slot_load_window.dtype)
    reconstructed.scatter_add_(
        2,
        placement.unsqueeze(0).expand(logical_load_window.shape[0], -1, -1),
        slot_load_window,
    )

    torch.testing.assert_close(reconstructed, logical_load_window.to(torch.float64))


def test_flashlb_adapter_preserves_v2_contract_and_only_applies_priority_layers(monkeypatch):
    class FakeFlashLB:
        def rebalance_experts(self, placement, workload):
            assert placement.shape == (2, 2, 3)
            assert workload.shape == (3, 2, 2, 3)
            proposal = placement.clone()
            proposal[0] = torch.tensor([[0, 1, 3], [2, 0, 1]])
            proposal[1] = torch.tensor([[1, 0, 2], [3, 0, 1]])
            return True, [0], proposal

    logical_load_window = torch.ones((3, 2, 4), dtype=torch.int32)
    placement = torch.tensor(
        [[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 0, 1]],
        dtype=torch.long,
    )
    monkeypatch.setattr(FlashLBEplbPolicyAdapter, "_policy", FakeFlashLB())

    result = FlashLBEplbPolicyAdapter.rebalance_experts(
        logical_load_window,
        num_replicas=6,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=placement,
    )

    assert FlashLBEplbPolicyAdapter.uses_expert_load_time_series
    assert result.shape == placement.shape
    assert result.dtype == torch.long
    assert result.device.type == "cpu"
    assert result.is_contiguous()
    torch.testing.assert_close(result[0], torch.tensor([0, 1, 3, 2, 0, 1]))
    torch.testing.assert_close(result[1], placement[1])


def test_reject_invalid_placement_layers_matches_v1_worker_constraints():
    old_placement = torch.tensor(
        [[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 0, 1]],
        dtype=torch.long,
    )
    proposed_placement = torch.tensor(
        [
            [0, 1, 3, 2, 0, 1],
            [0, 2, 1, 3, 0, 0],
        ],
        dtype=torch.long,
    )

    rejected = _reject_invalid_placement_layers(
        old_placement,
        proposed_placement,
        num_ranks=2,
        num_logical_experts=4,
    )

    assert rejected == [1]
    torch.testing.assert_close(proposed_placement[0], torch.tensor([0, 1, 3, 2, 0, 1]))
    torch.testing.assert_close(proposed_placement[1], old_placement[1])


def test_swift_adapter_preserves_inputs_and_v2_output_contract():
    logical_load = torch.tensor([[1, 1, 100, 1], [1, 80, 1, 1]], dtype=torch.int32)
    placement = torch.tensor(
        [[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 0, 1]],
        dtype=torch.long,
    )
    original_load = logical_load.clone()
    original_placement = placement.clone()

    result = SwiftEplbPolicyAdapter.rebalance_experts(
        logical_load,
        num_replicas=6,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=placement,
    )

    assert result.shape == (2, 6)
    assert result.dtype == torch.long
    assert result.device.type == "cpu"
    assert result.is_contiguous()
    torch.testing.assert_close(logical_load, original_load)
    torch.testing.assert_close(placement, original_placement)


def test_swift_adapter_keeps_balanced_placement():
    logical_load = torch.ones((2, 4), dtype=torch.int32)
    placement = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]],
        dtype=torch.long,
    )

    result = SwiftEplbPolicyAdapter.rebalance_experts(
        logical_load,
        num_replicas=4,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=placement,
    )

    torch.testing.assert_close(result, placement)


def test_swift_adapter_matches_direct_legacy_policy():
    logical_load = torch.tensor([[1, 1, 100, 1], [1, 80, 1, 1]], dtype=torch.int32)
    placement = torch.tensor(
        [[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 0, 1]],
        dtype=torch.long,
    )
    slot_load = _expand_logical_load_to_slots(logical_load, placement)
    legacy_policy = SwiftBalanceEplb()
    legacy_policy.num_die_per_host = 2
    _, _, legacy_result = legacy_policy.rebalance_experts(
        placement.reshape(2, 2, 3),
        slot_load.reshape(2, 2, 3),
        is_node_redundant=False,
    )

    adapter_result = SwiftEplbPolicyAdapter.rebalance_experts(
        logical_load,
        num_replicas=6,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=placement,
    )

    torch.testing.assert_close(adapter_result, torch.tensor(legacy_result).reshape(2, 6))


def test_swift_adapter_keeps_four_redundant_experts_and_is_deterministic():
    logical_load = torch.tensor([[100, 1, 80, 1, 60, 1, 40, 1]], dtype=torch.int32)
    placement = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]], dtype=torch.long)
    expected_total_replicas = placement.shape[1]

    results = [
        SwiftEplbPolicyAdapter.rebalance_experts(
            logical_load,
            num_replicas=expected_total_replicas,
            num_groups=1,
            num_nodes=1,
            num_ranks=2,
            old_global_expert_indices=placement,
        )
        for _ in range(2)
    ]

    torch.testing.assert_close(results[0], results[1])
    replica_counts = torch.zeros_like(logical_load, dtype=torch.long)
    replica_counts.scatter_add_(1, results[0], torch.ones_like(results[0]))
    assert results[0].numel() - logical_load.shape[1] == 4
    assert bool((replica_counts >= 1).all())


def test_swift_adapter_rejects_missing_or_invalid_placement():
    logical_load = torch.ones((1, 4), dtype=torch.int32)

    for placement, error in [
        (None, "requires the current"),
        (torch.tensor([[0, 1, 2, 4]]), "invalid logical expert"),
        (torch.tensor([[0, 0, 1, 2]]), "at least one physical replica"),
    ]:
        try:
            SwiftEplbPolicyAdapter.rebalance_experts(
                logical_load,
                num_replicas=4,
                num_groups=1,
                num_nodes=1,
                num_ranks=2,
                old_global_expert_indices=placement,
            )
        except ValueError as exc:
            assert error in str(exc)
        else:
            raise AssertionError("Expected invalid Swift placement to fail.")


def test_stair_adapter_preserves_inputs_contract_and_improves_balance(monkeypatch):
    logical_load_window = torch.tensor(
        [
            [[1, 1, 100, 1]],
            [[1, 1, 120, 1]],
            [[1, 1, 80, 1]],
            [[1, 1, 110, 1]],
        ],
        dtype=torch.int32,
    )
    placement = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long)
    original_load = logical_load_window.clone()
    original_placement = placement.clone()
    monkeypatch.setattr(StairEplbPolicyAdapter, "_policy", StairEplbPolicy())

    result = StairEplbPolicyAdapter.rebalance_experts(
        logical_load_window,
        num_replicas=6,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=placement,
    )

    assert StairEplbPolicyAdapter.uses_expert_load_time_series
    assert result.shape == placement.shape
    assert result.dtype == torch.long
    assert result.device.type == "cpu"
    assert result.is_contiguous()
    torch.testing.assert_close(logical_load_window, original_load)
    torch.testing.assert_close(placement, original_placement)
    old_score = compute_balance_score(
        logical_load_window[:, 0].numpy(),
        placement.reshape(2, 3).numpy(),
    )
    new_score = compute_balance_score(
        logical_load_window[:, 0].numpy(),
        result.reshape(2, 3).numpy(),
    )
    assert new_score < old_score
    for rank in result.reshape(2, 3):
        assert torch.unique(rank).numel() == rank.numel()


def test_stair_adapter_filters_swift_candidates_with_the_full_time_series(monkeypatch):
    logical_load_window = torch.tensor(
        [
            [[100, 90, 1, 1], [100, 1, 90, 1]],
            [[90, 100, 1, 1], [90, 1, 100, 1]],
        ],
        dtype=torch.int32,
    )
    placement = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]],
        dtype=torch.long,
    )
    swift_candidate = torch.tensor(
        [[0, 2, 1, 3], [0, 2, 1, 3]],
        dtype=torch.long,
    )
    observed_logical_load = None

    def fake_calculate_swift_placement(logical_load, old_placement, num_nodes, num_ranks):
        nonlocal observed_logical_load
        observed_logical_load = logical_load.clone()
        return swift_candidate.clone()

    monkeypatch.setattr(eplb_policy, "_calculate_swift_placement", fake_calculate_swift_placement)
    monkeypatch.setattr(StairEplbPolicyAdapter, "_policy", StairEplbPolicy())

    result = StairEplbPolicyAdapter.rebalance_experts(
        logical_load_window,
        num_replicas=4,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=placement,
    )

    torch.testing.assert_close(observed_logical_load, logical_load_window.sum(dim=0))
    torch.testing.assert_close(result[0], swift_candidate[0])
    torch.testing.assert_close(result[1], placement[1])


def test_stair_adapter_rejects_missing_or_invalid_placement(monkeypatch):
    logical_load_window = torch.ones((2, 1, 4), dtype=torch.int32)
    monkeypatch.setattr(StairEplbPolicyAdapter, "_policy", StairEplbPolicy())

    for placement, error in [
        (None, "requires the current"),
        (torch.tensor([[0, 1, 2, 4]]), "invalid logical expert"),
        (torch.tensor([[0, 0, 1, 2]]), "at least one physical replica"),
    ]:
        try:
            StairEplbPolicyAdapter.rebalance_experts(
                logical_load_window,
                num_replicas=4,
                num_groups=1,
                num_nodes=1,
                num_ranks=2,
                old_global_expert_indices=placement,
            )
        except ValueError as exc:
            assert error in str(exc)
        else:
            raise AssertionError("Expected invalid STAIR placement to fail.")
