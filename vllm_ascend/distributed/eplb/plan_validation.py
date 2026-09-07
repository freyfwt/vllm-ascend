# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Execution-side validation for untrusted STAIR planner results."""

from collections import Counter

import numpy as np

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.policy.stair_candidate import config_digest
from vllm_ascend.distributed.eplb.policy.stair_stats import placement_score
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology, RebalancePlan, validate_placement


def validate_plan(
    plan: RebalancePlan,
    placements: np.ndarray,
    placement_epochs: np.ndarray,
    topology: RankTopology,
    config: StairConfig,
    *,
    model_id: str,
    current_planning_round: int,
    snapshot_sequence: int,
    sample_sequence: int,
    stats_schema_epoch: int,
    logical_load: np.ndarray | None = None,
    sample_weights: np.ndarray | None = None,
) -> None:
    """Reject a plan before any transfer when one invariant has changed."""
    expected_identity = (
        model_id,
        snapshot_sequence,
        sample_sequence,
        stats_schema_epoch,
        config_digest(config),
        topology.digest(),
    )
    actual_identity = (
        plan.model_id,
        plan.snapshot_sequence,
        plan.sample_sequence,
        plan.stats_schema_epoch,
        plan.config_digest,
        plan.topology_digest,
    )
    if actual_identity != expected_identity:
        raise ValueError("STAIR plan identity is stale or inconsistent")
    age = current_planning_round - plan.planning_round
    if age < 0 or age > config.max_plan_age_intervals:
        raise ValueError(f"STAIR plan age {age} is outside the accepted range")

    current = np.asarray(placements, dtype=np.int64)
    epochs = np.asarray(placement_epochs, dtype=np.int64)
    seen_layers: set[int] = set()
    for layer in plan.layers:
        layer_idx = layer.layer_idx
        if layer_idx in seen_layers or not 0 <= layer_idx < current.shape[0]:
            raise ValueError(f"STAIR plan has invalid or duplicate layer {layer_idx}")
        seen_layers.add(layer_idx)
        if layer.base_placement_epoch != epochs[layer_idx]:
            raise ValueError(f"STAIR layer {layer_idx} placement epoch is stale")
        if not np.array_equal(layer.old_placement, current[layer_idx]):
            raise ValueError(f"STAIR layer {layer_idx} old placement changed")
        validate_placement(layer.old_placement, int(current.max()) + 1)
        validate_placement(layer.new_placement, int(current.max()) + 1)
        if layer.old_placement.size != layer.new_placement.size:
            raise ValueError("STAIR cannot change the number of physical slots")
        ranks, slots = layer.old_placement.shape
        if (
            np.any(layer.source_rank < 0)
            or np.any(layer.source_rank >= ranks)
            or np.any(layer.source_slot < 0)
            or np.any(layer.source_slot >= slots)
        ):
            raise ValueError("STAIR plan contains an invalid source rank or slot")
        pair_usage: Counter[tuple[int, int]] = Counter()
        for dst in range(ranks):
            old_row = layer.old_placement[dst]
            new_row = layer.new_placement[dst]
            for old_slot, expert in enumerate(old_row):
                if expert in new_row and new_row[old_slot] != expert:
                    raise ValueError("STAIR rank-local stable slot alignment was not preserved")
            for dst_slot, expert in enumerate(new_row):
                src = int(layer.source_rank[dst, dst_slot])
                src_slot = int(layer.source_slot[dst, dst_slot])
                if layer.old_placement[src, src_slot] != expert:
                    raise ValueError("STAIR source does not own the requested expert")
                if expert in old_row:
                    expected_slot = int(np.flatnonzero(old_row == expert)[0])
                    if src != dst or src_slot != expected_slot:
                        raise ValueError("STAIR retained expert must use its local source")
                elif src != dst:
                    pair_usage[(src, dst)] += 1
        if pair_usage and max(pair_usage.values()) > config.max_expert_transfers_per_rank_pair:
            raise ValueError("STAIR directed rank-pair transfer cap was exceeded")
        scores = (
            layer.current_score.mean,
            layer.current_score.p95,
            layer.current_score.maximum,
            layer.candidate_score.mean,
            layer.candidate_score.p95,
            layer.candidate_score.maximum,
        )
        if not np.all(np.isfinite(scores)):
            raise ValueError("STAIR plan score contains NaN or infinity")
        if logical_load is not None and sample_weights is not None:
            samples = np.asarray(logical_load, dtype=np.float64)[:, layer_idx]
            current_score = placement_score(samples, sample_weights, layer.old_placement)
            candidate_score = placement_score(samples, sample_weights, layer.new_placement)
            reported = np.asarray(scores)
            recomputed = np.asarray(
                (
                    current_score.mean,
                    current_score.p95,
                    current_score.maximum,
                    candidate_score.mean,
                    candidate_score.p95,
                    candidate_score.maximum,
                )
            )
            if not np.allclose(reported, recomputed, rtol=1e-12, atol=1e-12):
                raise ValueError("STAIR planner score does not match the submitted snapshot")
        relative_gain = (layer.current_score.mean - layer.candidate_score.mean) / layer.current_score.mean
        if (
            relative_gain < config.min_relative_score_improvement
            or layer.current_score.mean - layer.candidate_score.mean < config.min_absolute_score_improvement
            or layer.candidate_score.p95 > layer.current_score.p95 * (1 + config.p95_regression_tolerance)
        ):
            raise ValueError("STAIR plan does not satisfy admission thresholds")
