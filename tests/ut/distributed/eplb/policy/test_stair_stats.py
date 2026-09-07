import numpy as np

from vllm_ascend.distributed.eplb.policy.stair_stats import (
    compress_samples,
    placement_score,
    risk_weights,
    weighted_moments,
)


def test_compression_keeps_remainder_and_step_weights():
    values = np.arange(20, dtype=np.float64).reshape(5, 2, 2)
    compressed, weights = compress_samples(values, 3)
    np.testing.assert_array_equal(weights, [1, 2, 2])
    np.testing.assert_allclose(compressed[0], values[0])
    np.testing.assert_allclose(compressed[1], values[1:3].mean(axis=0))
    np.testing.assert_allclose(compressed[2], values[3:5].mean(axis=0))


def test_weighted_moments_use_sample_denominator_and_optional_covariance():
    samples = np.array([[1.0, 3.0], [3.0, 7.0]])
    mean, variance = weighted_moments(samples, np.array([1, 1]), covariance=False)
    np.testing.assert_allclose(mean, [2, 5])
    np.testing.assert_allclose(variance, [2, 8])
    _, covariance = weighted_moments(samples, np.array([1, 1]), covariance=True)
    np.testing.assert_allclose(covariance, [[2, 4], [4, 8]])
    np.testing.assert_allclose(risk_weights(mean, variance, 1), mean + np.sqrt(variance))


def test_score_is_step_weighted_mean_with_tail_metrics():
    samples = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
    placement = np.array([[0], [1]])
    score = placement_score(samples, np.array([1, 1, 8]), placement)
    assert score.mean == 1.2
    assert score.p95 == 2.0
    assert score.maximum == 2.0


def test_zero_load_score_is_balanced():
    score = placement_score(np.zeros((2, 2)), np.ones(2, dtype=np.int64), np.array([[0], [1]]))
    assert (score.mean, score.p95, score.maximum) == (1.0, 1.0, 1.0)
