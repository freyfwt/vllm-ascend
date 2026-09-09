# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend.eplb.core.policy import policy_flashlb
from vllm_ascend.eplb.core.policy.policy_flashlb import FlashLB, compute_score


@pytest.fixture
def policy():
    result = FlashLB()
    result.true_update = True  # Keep EP64 thresholds without a distributed group.
    return result


@pytest.fixture
def search(monkeypatch):
    result = Mock()
    monkeypatch.setattr(policy_flashlb, "FlashTree", result)
    return result


def test_ep64_replica_skew_is_not_hidden_from_update_gate(policy, search):
    placement = torch.tensor([[[4 * r, 4 * r + 1, 4 * r + 2, 4 * r + 3, 4 * ((r + 32) % 64)] for r in range(64)]])
    loads = torch.tensor([[[[960 if r < 32 else 0, 96, 96, 96, 960 if r < 32 else 0] for r in range(64)]]])
    policy.average_to_peak_history[0] = 1.0
    search.return_value.optimize_balanceness.return_value = (
        placement[0].numpy(),
        np.bincount(placement[0].numpy().ravel()),
        1.0,
    )

    changed, priority, updated = policy.rebalance_experts(placement, loads)

    # The old gate saw PAR 1.0 after averaging replicas. Physical PAR is 1.769.
    search.assert_called_once()
    assert policy.average_to_peak_history[0] == pytest.approx(1248 / 2208)
    # Searching is required, but returning the same placement is not an update.
    assert not changed
    assert priority.size == 0
    np.testing.assert_array_equal(updated, placement.numpy())


def test_observed_balance_is_not_overridden_by_replica_estimate(policy, search):
    placement = torch.tensor([[[0, 1], [2, 0]]])
    loads = torch.tensor([[[[90, 10], [90, 10]]]])
    policy.average_to_peak_history[0] = 1.0
    search.return_value.optimize_balanceness.return_value = (placement[0].numpy(), np.array([2, 1, 1]), 1.0)

    changed, _, updated = policy.rebalance_experts(placement, loads)

    search.assert_not_called()
    assert not changed
    np.testing.assert_array_equal(updated, placement.numpy())
    assert policy.average_to_peak_history[0] == 1.0


def test_history_uses_inverse_mean_step_par_and_ignores_empty_samples(policy, search):
    placement = torch.tensor([[[0, 1], [2, 3]]])
    loads = torch.tensor([[[[100, 0], [0, 0]]], [[[100, 0], [100, 0]]], [[[0, 0], [0, 0]]]])
    search.return_value.optimize_balanceness.return_value = (placement[0].numpy(), np.ones(4, dtype=np.int32), 1.0)

    policy.rebalance_experts(placement, loads)

    # Mean step PAR = (2 + 1) / 2, not PAR of the accumulated window (4/3).
    assert policy.average_to_peak_history[0] == pytest.approx(2 / 3)


def test_candidate_is_rescored_before_accepting_compressed_window_gain(policy, search):
    placement = torch.tensor([[[0, 1], [2, 3]]])
    loads = torch.tensor([[[[90, 10], [70, 10]]], [[[10, 90], [10, 70]]]])
    candidate = np.array([[0, 2], [1, 3]])
    policy.sample_size = 1
    search.return_value.optimize_balanceness.return_value = (candidate, np.ones(4, dtype=np.int32), 1.0)

    changed, priority, updated = policy.rebalance_experts(placement, loads)

    # The candidate balances the summed window (180,180), but worsens each
    # step from (100,80) to (160,20) or (20,160). It must not be returned.
    assert search.call_args.args[0].shape[0] == 1
    assert not changed
    assert priority.size == 0
    np.testing.assert_array_equal(updated, placement.numpy())


def test_equal_score_candidate_does_not_gain_from_float32_rounding(policy, search):
    placement = torch.tensor([[[0, 1], [2, 3]]])
    loads = torch.tensor([[[[60, 50], [50, 40]]]])
    search.return_value.optimize_balanceness.return_value = (
        np.array([[0, 2], [1, 3]]),
        np.ones(4, dtype=np.int32),
        1.1,
    )

    changed, priority, updated = policy.rebalance_experts(placement, loads)

    assert not changed
    assert priority.size == 0
    np.testing.assert_array_equal(updated, placement.numpy())


@pytest.mark.parametrize("upper_bound,selected", [(1, [0]), (-1, [0, 1])])
def test_only_selected_layers_are_returned_to_worker(policy, search, upper_bound, selected):
    placement = torch.tensor([[[0, 1], [2, 3]], [[0, 1], [2, 3]]])
    loads = torch.tensor([[[[100, 90], [10, 0]], [[100, 60], [30, 10]]]])
    candidate = np.array([[0, 2], [1, 3]])
    policy.update_layers_upper_bound = upper_bound
    search.return_value.optimize_balanceness.side_effect = [
        (candidate.copy(), np.ones(4, dtype=np.int32), 1.1),
        (candidate.copy(), np.ones(4, dtype=np.int32), 1.3),
    ]

    changed, priority, updated = policy.rebalance_experts(placement, loads)

    expected = placement.numpy().copy()
    expected[selected] = candidate
    assert changed
    np.testing.assert_array_equal(priority, selected)
    np.testing.assert_array_equal(updated, expected)
    # History is observed load, not the predicted load after a proposed move.
    assert policy.average_to_peak_history[0] == pytest.approx(100 / 190)
    assert policy.average_to_peak_history[1] == pytest.approx(100 / 160)


@pytest.mark.parametrize("invalid", [False, True])
def test_unchanged_or_invalid_candidate_has_no_gain(policy, search, invalid):
    placement = torch.tensor([[[0, 1], [2, 3]]])
    loads = torch.tensor([[[[100, 90], [10, 0]]]])
    candidate = placement[0].numpy().copy()
    if invalid:
        candidate[0, 0] = -1
    search.return_value.optimize_balanceness.return_value = (candidate, np.ones(4, dtype=np.int32), 1.0)

    changed, priority, updated = policy.rebalance_experts(placement, loads)

    assert not changed
    assert priority.size == 0
    np.testing.assert_array_equal(updated, placement.numpy())
    np.testing.assert_array_equal(policy.current_deployed_replicas[0], np.ones(4))


def test_empty_window_does_not_trigger_first_update(policy, search):
    placement = torch.tensor([[[0, 1], [2, 3]]])
    search.return_value.optimize_balanceness.return_value = (placement[0].numpy(), np.ones(4, dtype=np.int32), 1.0)

    changed, priority, updated = policy.rebalance_experts(placement, torch.zeros((2, 1, 2, 2), dtype=torch.int64))

    search.assert_not_called()
    assert not changed
    assert priority.size == 0
    assert 0 not in policy.average_to_peak_history
    np.testing.assert_array_equal(updated, placement.numpy())


@pytest.mark.parametrize("hotness,expected", [([[1, 0], [0, 0], [1, 1]], 1.5), ([[0, 0]], 1.0)])
def test_candidate_score_ignores_empty_samples_without_smoothing(hotness, expected):
    assert compute_score(np.array(hotness), np.ones(2, dtype=np.int32), np.array([[0], [1]])) == expected


def test_real_search_balances_alternating_rank_overload(policy):
    placement = torch.tensor([[[0, 1], [2, 3]]])
    loads = torch.tensor([[[[90, 80], [20, 10]]], [[[10, 20], [80, 90]]]])

    changed, priority, updated = policy.rebalance_experts(placement, loads)

    # Both ranks accumulate 200 tokens, but every sample has PAR 1.7.
    assert changed
    np.testing.assert_array_equal(priority, [0])
    assert policy.average_to_peak_history[0] == pytest.approx(1 / 1.7)
    candidate_score = compute_score(loads.numpy().reshape(2, 4), np.ones(4, dtype=np.int32), updated[0])
    assert candidate_score < 1.7
