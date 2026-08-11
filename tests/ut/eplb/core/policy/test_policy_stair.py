# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np
import pytest

from vllm_ascend.eplb.core.policy import policy_stair
from vllm_ascend.eplb.core.policy.policy_stair import (
    StairEplbPolicy,
    _compute_statistics,
    _refine_placement,
    _replace_surplus_replicas,
    allocate_replicas,
    compute_balance_score,
)


def _assert_valid_placement(
    placement: np.ndarray,
    num_experts: int,
    num_ranks: int,
) -> None:
    slots_per_rank = placement.shape[1] // num_ranks
    for layer in placement.reshape(placement.shape[0], num_ranks, slots_per_rank):
        counts = np.bincount(layer.reshape(-1), minlength=num_experts)
        assert np.all(counts >= 1)
        assert np.all(counts <= num_ranks)
        for rank in layer:
            assert np.unique(rank).size == rank.size


def _assert_kept_experts_stay_in_their_slots(
    old_placement: np.ndarray,
    new_placement: np.ndarray,
    num_ranks: int,
) -> None:
    slots_per_rank = old_placement.shape[1] // num_ranks
    old_by_rank = old_placement.reshape(old_placement.shape[0], num_ranks, slots_per_rank)
    new_by_rank = new_placement.reshape(new_placement.shape[0], num_ranks, slots_per_rank)
    for old_layer, new_layer in zip(old_by_rank, new_by_rank):
        for old_rank, new_rank in zip(old_layer, new_layer):
            kept = np.isin(new_rank, old_rank)
            np.testing.assert_array_equal(new_rank[kept], old_rank[kept])


def test_compute_balance_score_uses_peak_to_average_ratio():
    expert_load = np.array([[100, 1, 1, 1]], dtype=np.float64)
    placement = np.array([[0, 1], [2, 3]], dtype=np.int64)

    score = compute_balance_score(expert_load, placement)

    assert score == pytest.approx(101 / 51.5)


def test_allocate_replicas_caps_each_expert_at_num_ranks():
    replicas = allocate_replicas(
        mean=np.array([100.0, 1.0]),
        variance=np.zeros(2),
        num_replicas=4,
        num_ranks=2,
    )

    np.testing.assert_array_equal(replicas, np.array([2, 2]))


def test_stair_keeps_balanced_and_zero_load_placements_unchanged():
    placement = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)
    balanced_load = np.ones((4, 2, 4), dtype=np.float64)
    zero_load = np.zeros_like(balanced_load)

    policy = StairEplbPolicy()
    balanced_result = policy.rebalance_experts(balanced_load, placement, num_ranks=2)
    zero_result = policy.rebalance_experts(zero_load, placement, num_ranks=2)

    np.testing.assert_array_equal(balanced_result, placement)
    np.testing.assert_array_equal(zero_result, placement)


def test_stair_improves_balance_without_redundant_experts():
    placement = np.array([[0, 1, 2, 3]], dtype=np.int64)
    expert_load = np.tile(np.array([100, 90, 1, 1], dtype=np.float64), (8, 1))[:, None, :]

    result = StairEplbPolicy().rebalance_experts(expert_load, placement, num_ranks=2)

    current_score = compute_balance_score(expert_load[:, 0], placement.reshape(2, 2))
    result_score = compute_balance_score(expert_load[:, 0], result.reshape(2, 2))
    assert result_score < current_score
    _assert_valid_placement(result, num_experts=4, num_ranks=2)


def test_stair_reallocates_redundant_experts_and_preserves_local_slots():
    placement = np.array([[0, 1, 2, 3, 0, 1]], dtype=np.int64)
    samples = np.array(
        [
            [1, 1, 100, 80],
            [1, 1, 120, 60],
            [1, 1, 80, 100],
            [1, 1, 110, 70],
        ],
        dtype=np.float64,
    )
    expert_load = samples[:, None, :]
    original_load = expert_load.copy()
    original_placement = placement.copy()

    result = StairEplbPolicy().rebalance_experts(expert_load, placement, num_ranks=2)

    current_score = compute_balance_score(samples, placement.reshape(2, 3))
    result_score = compute_balance_score(samples, result.reshape(2, 3))
    assert result_score < current_score
    _assert_valid_placement(result, num_experts=4, num_ranks=2)
    _assert_kept_experts_stay_in_their_slots(placement, result, num_ranks=2)
    np.testing.assert_array_equal(expert_load, original_load)
    np.testing.assert_array_equal(placement, original_placement)


def test_stair_restores_flashlb_hysteresis_and_absolute_thresholds():
    policy = StairEplbPolicy()
    policy.average_to_peak_history[0] = 1.0

    assert not policy._needs_flash_update(0, current_score=1 / 0.96, num_ranks=2)
    assert policy._needs_flash_update(0, current_score=1 / 0.94, num_ranks=2)

    policy.average_to_peak_history[0] = 0.92
    assert policy._needs_flash_update(0, current_score=1 / 0.89, num_ranks=2)


def test_stair_commits_flashlb_history_only_for_changed_layers():
    policy = StairEplbPolicy()
    expert_load = np.tile(np.array([[100, 100, 1, 1], [1, 1, 100, 100]]), (4, 1, 1))
    current = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)
    committed = current.copy()
    committed[0] = np.array([0, 2, 1, 3])

    policy.commit(expert_load, current, committed, num_ranks=2)

    assert set(policy.average_to_peak_history) == {0}
    expected_score = compute_balance_score(expert_load[:, 0], committed[0].reshape(2, 2))
    assert policy.average_to_peak_history[0] == pytest.approx(1 / expected_score)


def test_stair_restores_swift_layer_imbalance_gate(monkeypatch):
    placement = np.array([[0, 1, 2, 3]], dtype=np.int64)
    expert_load = np.array([[[80.0, 20.9, 79.0, 20.0]]])
    monkeypatch.setattr(policy_stair, "SWIFT_GLOBAL_IMPROVEMENT_RATIO", 0.0)
    monkeypatch.setattr(policy_stair, "SWIFT_MIN_SWAP_IMPROVEMENT_RATIO", 0.0)

    gated = StairEplbPolicy().rebalance_experts(expert_load, placement, num_ranks=2)
    monkeypatch.setattr(policy_stair, "SWIFT_IMBALANCE_THRESHOLD", 1.0)
    ungated = StairEplbPolicy().rebalance_experts(expert_load, placement, num_ranks=2)

    np.testing.assert_array_equal(gated, placement)
    assert np.any(ungated != placement)


def test_stair_restores_swift_minimum_swap_improvement(monkeypatch):
    placement = np.array([[0, 1, 2, 3]], dtype=np.int64)
    expert_load = np.array([[[80.0, 21.5, 79.5, 19.0]]])
    monkeypatch.setattr(policy_stair, "SWIFT_GLOBAL_IMPROVEMENT_RATIO", 0.0)

    gated = StairEplbPolicy().rebalance_experts(expert_load, placement, num_ranks=2)
    monkeypatch.setattr(policy_stair, "SWIFT_MIN_SWAP_IMPROVEMENT_RATIO", 0.0)
    ungated = StairEplbPolicy().rebalance_experts(expert_load, placement, num_ranks=2)

    np.testing.assert_array_equal(gated, placement)
    assert np.any(ungated != placement)


def test_stair_restores_swift_rank_pair_swap_limit(monkeypatch):
    placement = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    expert_load = np.array([[100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0]])
    target_replicas = np.ones(8, dtype=np.int64)

    limited = _refine_placement(
        expert_load,
        placement,
        placement,
        target_replicas,
        np.zeros((2, 2), dtype=np.int64),
    )
    monkeypatch.setattr(policy_stair, "SWIFT_MAX_COMMUNICATIONS_PER_RANK_PAIR", 100)
    unlimited = _refine_placement(
        expert_load,
        placement,
        placement,
        target_replicas,
        np.zeros((2, 2), dtype=np.int64),
    )

    assert np.count_nonzero(limited != placement) == 2
    assert np.count_nonzero(unlimited != placement) == 4
    assert compute_balance_score(expert_load, unlimited) < compute_balance_score(expert_load, limited)


def test_stair_restores_swift_rank_pair_replacement_limit(monkeypatch):
    placement = np.array([[0, 1, 2, 3], [4, 5, 0, 1]], dtype=np.int64)
    target_replicas = np.array([1, 1, 2, 2, 1, 1], dtype=np.int64)
    expert_load = np.array([[1.0, 1.0, 100.0, 90.0, 1.0, 1.0]])
    mean, variance, covariance = _compute_statistics(expert_load)

    limited = _replace_surplus_replicas(
        expert_load,
        placement,
        target_replicas,
        mean,
        variance,
        covariance,
    )
    monkeypatch.setattr(policy_stair, "SWIFT_MAX_COMMUNICATIONS_PER_RANK_PAIR", 2)
    unlimited = _replace_surplus_replicas(
        expert_load,
        placement,
        target_replicas,
        mean,
        variance,
        covariance,
    )

    assert limited is None
    assert unlimited is not None
    np.testing.assert_array_equal(
        np.bincount(unlimited[0].reshape(-1), minlength=6),
        target_replicas,
    )


def test_stair_restores_swift_global_five_percent_gate(monkeypatch):
    placement = np.array([[0, 1, 2, 3]], dtype=np.int64)
    expert_load = np.array([[[80.0, 25.0, 75.0, 20.0]]])

    gated = StairEplbPolicy().rebalance_experts(expert_load, placement, num_ranks=2)
    monkeypatch.setattr(policy_stair, "SWIFT_GLOBAL_IMPROVEMENT_RATIO", 0.0)
    ungated = StairEplbPolicy().rebalance_experts(expert_load, placement, num_ranks=2)

    np.testing.assert_array_equal(gated, placement)
    assert np.any(ungated != placement)


@pytest.mark.parametrize("seed", range(10))
def test_stair_randomized_outputs_are_valid_improving_and_deterministic(seed: int):
    rng = np.random.default_rng(seed)
    num_layers = 3
    num_experts = 8
    num_ranks = 4
    slots_per_rank = 3
    base_placement = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 0],
            [1, 2, 3],
        ],
        dtype=np.int64,
    ).reshape(-1)
    placement = np.tile(base_placement, (num_layers, 1))
    expert_load = rng.integers(1, 200, size=(12, num_layers, num_experts)).astype(np.float64)

    first = StairEplbPolicy().rebalance_experts(expert_load, placement, num_ranks=num_ranks)
    second = StairEplbPolicy().rebalance_experts(expert_load, placement, num_ranks=num_ranks)

    np.testing.assert_array_equal(first, second)
    _assert_valid_placement(first, num_experts=num_experts, num_ranks=num_ranks)
    _assert_kept_experts_stay_in_their_slots(placement, first, num_ranks=num_ranks)
    for layer_id in range(num_layers):
        old_score = compute_balance_score(
            expert_load[:, layer_id],
            placement[layer_id].reshape(num_ranks, slots_per_rank),
        )
        new_score = compute_balance_score(
            expert_load[:, layer_id],
            first[layer_id].reshape(num_ranks, slots_per_rank),
        )
        assert new_score <= old_score + 1e-6


@pytest.mark.parametrize(
    ("expert_load", "placement", "error"),
    [
        (
            np.ones((2, 1, 4)),
            np.array([[0, 1, 2, 4]]),
            "invalid logical expert",
        ),
        (
            np.ones((2, 1, 4)),
            np.array([[0, 0, 1, 2]]),
            "at least one physical replica",
        ),
        (
            np.ones((2, 1, 3)),
            np.array([[0, 0, 1, 2]]),
            "two replicas on the same rank",
        ),
        (
            np.array([[[1.0, -1.0, 1.0, 1.0]]]),
            np.array([[0, 1, 2, 3]]),
            "finite, non-negative",
        ),
    ],
)
def test_stair_rejects_invalid_inputs(
    expert_load: np.ndarray,
    placement: np.ndarray,
    error: str,
):
    with pytest.raises(ValueError, match=error):
        StairEplbPolicy().rebalance_experts(expert_load, placement, num_ranks=2)
