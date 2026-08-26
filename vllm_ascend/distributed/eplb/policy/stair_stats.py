# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Temporal load statistics and balance metrics for STAIR EPLB."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from vllm_ascend.distributed.eplb.policy import stair_constants as constants
from vllm_ascend.distributed.eplb.policy.stair_types import StairBalanceScore


def replica_counts(placement: np.ndarray, num_experts: int) -> npt.NDArray[np.int64]:
    """Return the number of physical replicas for every logical expert."""
    return np.bincount(placement.reshape(-1), minlength=num_experts).astype(np.int64, copy=False)


def rank_loads(
    expert_load: np.ndarray,
    placement: np.ndarray,
    num_experts: int,
) -> npt.NDArray[np.float64]:
    """Project a load time series onto ranks for one layer placement."""
    counts = replica_counts(placement, num_experts)
    if np.any(counts == 0):
        raise ValueError("Every logical expert must have at least one physical replica.")

    loads = np.zeros((expert_load.shape[0], placement.shape[0]), dtype=np.float64)
    for rank_id, rank in enumerate(placement):
        loads[:, rank_id] = np.sum(expert_load[:, rank] / counts[rank], axis=1)
    return loads


def compute_balance_metrics(expert_load: np.ndarray, placement: np.ndarray) -> StairBalanceScore:
    """Return mean, p95, and maximum peak-to-average imbalance."""
    expert_load = np.asarray(expert_load, dtype=np.float64)
    placement = np.asarray(placement, dtype=np.int64)
    if expert_load.ndim != 2:
        raise ValueError(f"expert_load must have shape [window, experts], got {expert_load.shape}.")
    if placement.ndim != 2:
        raise ValueError(f"placement must have shape [ranks, slots], got {placement.shape}.")
    if expert_load.shape[0] == 0 or expert_load.shape[1] == 0:
        raise ValueError("expert_load window and expert dimensions must be nonzero.")
    if np.any(placement < 0) or np.any(placement >= expert_load.shape[1]):
        raise ValueError("placement contains an invalid logical expert index.")

    loads = rank_loads(expert_load, placement, expert_load.shape[1])
    total_load = np.sum(loads, axis=1)
    imbalance = np.ones(loads.shape[0], dtype=np.float64)
    nonzero = total_load > 0
    if np.any(nonzero):
        average_load = total_load[nonzero] / loads.shape[1]
        imbalance[nonzero] = np.max(loads[nonzero], axis=1) / average_load
    return StairBalanceScore(
        mean_imbalance=float(np.mean(imbalance)),
        p95_imbalance=float(np.percentile(imbalance, 95)),
        max_imbalance=float(np.max(imbalance)),
    )


def compute_balance_gain(current: StairBalanceScore, candidate: StairBalanceScore) -> float:
    """Combine average and tail improvements using repository constants."""
    return constants.MEAN_SCORE_WEIGHT * (
        current.mean_imbalance - candidate.mean_imbalance
    ) + constants.P95_SCORE_WEIGHT * (current.p95_imbalance - candidate.p95_imbalance)


@dataclass
class StairLoadStats:
    """Policy-owned EWMA state used to construct risk-aware expert load."""

    shape_key: tuple[int, ...] | None = None
    sample_count: int = 0
    ewma_mean: npt.NDArray[np.float64] | None = None
    ewma_second_moment: npt.NDArray[np.float64] | None = None
    stagnation_cycles: npt.NDArray[np.int64] | None = None
    _initialized_layers: npt.NDArray[np.bool_] | None = field(default=None, repr=False)

    def reset(self) -> None:
        """Discard statistics after a topology, shape, or tuning change."""
        self.shape_key = None
        self.sample_count = 0
        self.ewma_mean = None
        self.ewma_second_moment = None
        self.stagnation_cycles = None
        self._initialized_layers = None

    def _ensure_compatible(self, load_window: np.ndarray, shape_key: tuple[int, ...]) -> None:
        value_shape = load_window.shape[1:]
        if self.shape_key == shape_key and self.ewma_mean is not None and self.ewma_mean.shape == value_shape:
            return
        self.reset()
        self.shape_key = shape_key
        self.ewma_mean = np.zeros(value_shape, dtype=np.float64)
        self.ewma_second_moment = np.zeros(value_shape, dtype=np.float64)
        self.stagnation_cycles = np.zeros(value_shape[0], dtype=np.int64)
        self._initialized_layers = np.zeros(value_shape[0], dtype=np.bool_)

    def update(self, load_window: np.ndarray, shape_key: tuple[int, ...]) -> None:
        """Blend all samples in the latest time window into EWMA moments."""
        load_window = np.asarray(load_window, dtype=np.float64)
        if load_window.ndim != 3 or load_window.shape[0] == 0:
            raise ValueError("STAIR load statistics require [window, layers, experts] samples.")
        self._ensure_compatible(load_window, shape_key)
        assert self.ewma_mean is not None
        assert self.ewma_second_moment is not None
        assert self._initialized_layers is not None

        window_mean = np.mean(load_window, axis=0)
        window_second_moment = np.mean(np.square(load_window), axis=0)
        initialized = self._initialized_layers
        if np.any(initialized):
            self.ewma_mean[initialized] = (1 - constants.EWMA_ALPHA) * self.ewma_mean[
                initialized
            ] + constants.EWMA_ALPHA * window_mean[initialized]
            self.ewma_second_moment[initialized] = (1 - constants.EWMA_ALPHA) * self.ewma_second_moment[
                initialized
            ] + constants.EWMA_ALPHA * window_second_moment[initialized]
        if np.any(~initialized):
            self.ewma_mean[~initialized] = window_mean[~initialized]
            self.ewma_second_moment[~initialized] = window_second_moment[~initialized]
        initialized[:] = True
        self.sample_count += load_window.shape[0]

    def risk_load(self) -> npt.NDArray[np.float64]:
        """Return EWMA mean plus a variance risk margin for candidate search."""
        if self.ewma_mean is None or self.ewma_second_moment is None:
            raise RuntimeError("STAIR load statistics have not been initialized.")
        variance = np.maximum(self.ewma_second_moment - np.square(self.ewma_mean), 0.0)
        return self.ewma_mean + constants.RISK_Z * np.sqrt(variance)

    def reset_layers(self, layer_ids: list[int]) -> None:
        """Invalidate layers whose committed placement was not observed."""
        if self._initialized_layers is None or self.stagnation_cycles is None:
            return
        for layer_id in layer_ids:
            if 0 <= layer_id < self._initialized_layers.size:
                self._initialized_layers[layer_id] = False
                self.stagnation_cycles[layer_id] = 0

    def note_candidate(self, layer_idx: int, improved: bool) -> None:
        """Track cycles without a useful greedy candidate for bounded search."""
        if self.stagnation_cycles is None:
            return
        self.stagnation_cycles[layer_idx] = 0 if improved else self.stagnation_cycles[layer_idx] + 1

    def should_search(
        self,
        layer_idx: int,
        current_score: StairBalanceScore,
        greedy_gain: float,
    ) -> bool:
        """Return whether a persistently difficult layer may use bounded search."""
        if not constants.ENABLE_MINI_FLASH_TREE or self.stagnation_cycles is None:
            return False
        if self.stagnation_cycles[layer_idx] < constants.SEARCH_STAGNATION_CYCLES:
            return False
        excess_imbalance = max(current_score.p95_imbalance - 1.0, constants.BALANCE_EPSILON)
        return (
            current_score.p95_imbalance >= constants.IMBALANCE_THRESHOLD
            and greedy_gain <= excess_imbalance * constants.SEARCH_MAX_GREEDY_GAIN_RATIO
        )
