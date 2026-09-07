# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Pure six-stage STAIR rebalance planner."""

import numpy as np

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.policy.stair_candidate import config_digest, passes_hysteresis, plan_layer
from vllm_ascend.distributed.eplb.policy.stair_stats import compress_samples, placement_score
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology, RebalancePlan, validate_placement


def plan_rebalance(
    logical_load: np.ndarray,
    placements: np.ndarray,
    placement_epochs: np.ndarray,
    accepted_scores: tuple[float | None, ...],
    topology: RankTopology,
    config: StairConfig,
    *,
    sample_weights: np.ndarray | None = None,
    model_id: str,
    planning_round: int,
    snapshot_sequence: int,
    sample_sequence: int,
    stats_schema_epoch: int,
) -> RebalancePlan:
    """Plan every layer that passes the load, balance, and hysteresis gates."""
    raw = np.asarray(logical_load)
    old = np.asarray(placements, dtype=np.int64)
    if raw.ndim != 3 or old.ndim != 3 or raw.shape[1] != old.shape[0]:
        raise ValueError("STAIR expects [steps,layers,experts] load and [layers,ranks,slots] placements")
    if sample_weights is None:
        compressed, weights = compress_samples(raw, config.sample_size)
    else:
        compressed = np.asarray(raw, dtype=np.float64)
        weights = np.asarray(sample_weights, dtype=np.int64)
        if weights.shape != (raw.shape[0],) or np.any(weights <= 0):
            raise ValueError("STAIR sample weights must match compressed bins and be positive")
    eligible: list[tuple[float, float, int]] = []
    for layer_idx in range(old.shape[0]):
        validate_placement(old[layer_idx], raw.shape[2])
        if np.sum(compressed[:, layer_idx], dtype=np.float64) == 0:
            continue
        score = placement_score(compressed[:, layer_idx], weights, old[layer_idx])
        anchor = accepted_scores[layer_idx]
        if score.mean > config.imbalance_threshold and passes_hysteresis(score.mean, anchor, config):
            deterioration = 0.0 if anchor is None else score.mean / anchor - 1.0
            eligible.append((-score.mean, -deterioration, layer_idx))
    layers = []
    for _, _, layer_idx in sorted(eligible):
        layer = plan_layer(
            layer_idx,
            compressed[:, layer_idx],
            weights,
            old[layer_idx],
            int(placement_epochs[layer_idx]),
            topology,
            config,
        )
        if layer is not None:
            layers.append(layer)
    return RebalancePlan(
        model_id,
        planning_round,
        snapshot_sequence,
        sample_sequence,
        stats_schema_epoch,
        config_digest(config),
        topology.digest(),
        tuple(layers),
    )
