# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np
import pytest

from vllm_ascend.eplb.core.policy.policy_stair import (
    StairEplbPolicy,
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

    balanced_result = StairEplbPolicy.rebalance_experts(balanced_load, placement, num_ranks=2)
    zero_result = StairEplbPolicy.rebalance_experts(zero_load, placement, num_ranks=2)

    np.testing.assert_array_equal(balanced_result, placement)
    np.testing.assert_array_equal(zero_result, placement)


def test_stair_improves_balance_without_redundant_experts():
    placement = np.array([[0, 1, 2, 3]], dtype=np.int64)
    expert_load = np.tile(np.array([100, 90, 1, 1], dtype=np.float64), (8, 1))[:, None, :]

    result = StairEplbPolicy.rebalance_experts(expert_load, placement, num_ranks=2)

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

    result = StairEplbPolicy.rebalance_experts(expert_load, placement, num_ranks=2)

    current_score = compute_balance_score(samples, placement.reshape(2, 3))
    result_score = compute_balance_score(samples, result.reshape(2, 3))
    assert result_score < current_score
    _assert_valid_placement(result, num_experts=4, num_ranks=2)
    _assert_kept_experts_stay_in_their_slots(placement, result, num_ranks=2)
    np.testing.assert_array_equal(expert_load, original_load)
    np.testing.assert_array_equal(placement, original_placement)


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

    first = StairEplbPolicy.rebalance_experts(expert_load, placement, num_ranks=num_ranks)
    second = StairEplbPolicy.rebalance_experts(expert_load, placement, num_ranks=num_ranks)

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
        StairEplbPolicy.rebalance_experts(expert_load, placement, num_ranks=2)
