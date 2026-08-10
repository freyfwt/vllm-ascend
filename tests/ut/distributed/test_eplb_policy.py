# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

from vllm_ascend.distributed.eplb_policy import (
    SwiftEplbPolicyAdapter,
    _expand_logical_load_to_slots,
)
from vllm_ascend.eplb.core.policy.policy_swift_balancer import SwiftBalanceEplb


def test_expand_logical_load_to_slots_preserves_logical_load():
    logical_load = torch.tensor([[10, 20, 30, 40]], dtype=torch.int32)
    placement = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long)

    slot_load = _expand_logical_load_to_slots(logical_load, placement)
    reconstructed = torch.zeros_like(logical_load, dtype=slot_load.dtype)
    reconstructed.scatter_add_(1, placement, slot_load)

    torch.testing.assert_close(reconstructed, logical_load.to(torch.float64))


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
