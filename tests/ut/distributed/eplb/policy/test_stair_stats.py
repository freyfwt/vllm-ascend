# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np
import pytest

from vllm_ascend.distributed.eplb.policy import stair_constants as constants
from vllm_ascend.distributed.eplb.policy.stair_stats import (
    StairLoadStats,
    compute_balance_gain,
    compute_balance_metrics,
)


def test_balance_metrics_preserve_mean_p95_and_max():
    load_window = np.array(
        [
            [10, 10, 10, 10],
            [100, 100, 1, 1],
        ],
        dtype=np.float64,
    )
    placement = np.array([[0, 1], [2, 3]], dtype=np.int64)

    score = compute_balance_metrics(load_window, placement)

    assert 1 < score.mean_imbalance < score.p95_imbalance < score.max_imbalance


def test_balance_gain_combines_mean_and_tail(monkeypatch):
    current = compute_balance_metrics(
        np.array([[100, 100, 1, 1]], dtype=np.float64),
        np.array([[0, 1], [2, 3]], dtype=np.int64),
    )
    candidate = compute_balance_metrics(
        np.array([[100, 100, 1, 1]], dtype=np.float64),
        np.array([[0, 2], [1, 3]], dtype=np.int64),
    )
    monkeypatch.setattr(constants, "MEAN_SCORE_WEIGHT", 0.25)
    monkeypatch.setattr(constants, "P95_SCORE_WEIGHT", 0.75)

    gain = compute_balance_gain(current, candidate)

    expected = 0.25 * (current.mean_imbalance - candidate.mean_imbalance) + 0.75 * (
        current.p95_imbalance - candidate.p95_imbalance
    )
    assert gain == pytest.approx(expected)


def test_load_stats_use_full_window_moments_and_ewma():
    stats = StairLoadStats()
    first_window = np.array([[[0, 2]], [[2, 0]]], dtype=np.float64)
    second_window = np.array([[[3, 3]], [[3, 3]]], dtype=np.float64)

    stats.update(first_window, (1, 2, constants.STAIR_TUNING_VERSION))
    np.testing.assert_allclose(stats.risk_load(), [[2, 2]])
    stats.update(second_window, (1, 2, constants.STAIR_TUNING_VERSION))

    expected_mean = np.array([[1.5, 1.5]])
    expected_second_moment = np.array([[3.75, 3.75]])
    expected_risk = expected_mean + np.sqrt(expected_second_moment - np.square(expected_mean))
    np.testing.assert_allclose(stats.ewma_mean, expected_mean)
    np.testing.assert_allclose(stats.risk_load(), expected_risk)
    assert stats.sample_count == 4


def test_load_stats_reset_on_shape_or_tuning_key_change():
    stats = StairLoadStats()
    stats.update(np.ones((2, 1, 2)), (1, 2, 1))
    stats.note_candidate(0, improved=False)

    stats.update(np.full((1, 1, 2), 5.0), (1, 2, 2))

    np.testing.assert_allclose(stats.ewma_mean, [[5, 5]])
    np.testing.assert_array_equal(stats.stagnation_cycles, [0])
    assert stats.sample_count == 1
