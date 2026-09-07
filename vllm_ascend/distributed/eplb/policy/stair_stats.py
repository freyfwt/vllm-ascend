# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Deterministic weighted statistics for STAIR."""

import numpy as np
import numpy.typing as npt

from vllm_ascend.distributed.eplb.policy.stair_types import BalanceScore


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def compress_samples(samples: np.ndarray, sample_size: int) -> tuple[FloatArray, IntArray]:
    """Compress chronological steps into non-empty contiguous weighted bins."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] == 0 or sample_size < 1:
        raise ValueError("STAIR samples require a non-empty time axis and positive sample_size")
    bins = min(values.shape[0], sample_size)
    boundaries = np.arange(bins + 1, dtype=np.int64) * values.shape[0] // bins
    weights = np.diff(boundaries)
    compressed = np.stack(
        [values[start:end].mean(axis=0, dtype=np.float64) for start, end in zip(boundaries[:-1], boundaries[1:])]
    )
    return compressed, weights


def weighted_moments(
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    covariance: bool,
) -> tuple[FloatArray, FloatArray]:
    """Return float64 mean and sample variance/covariance in fixed order."""
    values = np.asarray(samples, dtype=np.float64)
    counts = np.asarray(weights, dtype=np.int64)
    if values.ndim != 2 or counts.shape != (values.shape[0],) or np.any(counts <= 0):
        raise ValueError("STAIR weighted moments require [samples, experts] and positive bin lengths")
    total = int(counts.sum())
    mean = np.sum(values * counts[:, None], axis=0, dtype=np.float64) / total
    centered = values - mean
    if total == 1:
        shape = (values.shape[1], values.shape[1]) if covariance else (values.shape[1],)
        return mean, np.zeros(shape, dtype=np.float64)
    if covariance:
        moments = (centered * counts[:, None]).T @ centered / (total - 1)
        moments = (moments + moments.T) * 0.5
    else:
        moments = np.sum(centered * centered * counts[:, None], axis=0, dtype=np.float64) / (total - 1)
    return mean, moments


def risk_weights(mean: np.ndarray, variance: np.ndarray, z_score: float) -> FloatArray:
    diagonal = np.diag(variance) if variance.ndim == 2 else variance
    result = np.asarray(mean, dtype=np.float64) + z_score * np.sqrt(np.maximum(diagonal, 0.0))
    if not np.all(np.isfinite(result)):
        raise ValueError("STAIR risk weights must be finite")
    return result


def replica_counts(placement: np.ndarray, num_experts: int) -> IntArray:
    counts = np.bincount(np.asarray(placement).reshape(-1), minlength=num_experts).astype(np.int64)
    if np.any(counts == 0):
        raise ValueError("STAIR placement must contain every logical expert")
    return counts


def rank_loads(samples: np.ndarray, placement: np.ndarray) -> FloatArray:
    values = np.asarray(samples, dtype=np.float64)
    layout = np.asarray(placement, dtype=np.int64)
    if values.ndim != 2 or layout.ndim != 2:
        raise ValueError("STAIR scoring expects [samples, experts] and [ranks, slots]")
    counts = replica_counts(layout, values.shape[1])
    loads = np.empty((values.shape[0], layout.shape[0]), dtype=np.float64)
    for rank, experts in enumerate(layout):
        loads[:, rank] = np.sum(values[:, experts] / counts[experts], axis=1)
    return loads


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=np.int64)[order])
    target = max(1, int(np.ceil(quantile * int(cumulative[-1]))))
    return float(sorted_values[np.searchsorted(cumulative, target, side="left")])


def placement_score(samples: np.ndarray, weights: np.ndarray, placement: np.ndarray) -> BalanceScore:
    loads = rank_loads(samples, placement)
    totals = loads.sum(axis=1)
    imbalance = np.ones(loads.shape[0], dtype=np.float64)
    active = totals > 0
    imbalance[active] = loads[active].max(axis=1) / (totals[active] / loads.shape[1])
    counts = np.asarray(weights, dtype=np.int64)
    if counts.shape != imbalance.shape or np.any(counts <= 0):
        raise ValueError("STAIR score weights must match samples and be positive")
    return BalanceScore(
        mean=float(np.sum(imbalance * counts, dtype=np.float64) / counts.sum()),
        p95=weighted_quantile(imbalance, counts, 0.95),
        maximum=float(imbalance.max()),
    )
