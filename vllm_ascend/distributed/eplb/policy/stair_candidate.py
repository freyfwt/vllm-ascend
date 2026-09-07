# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Candidate generation and admission for one STAIR layer."""

import dataclasses
import hashlib
import json

import numpy as np

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.policy.stair_placement import PlacementResult, constrained_lpt, unconstrained_lpt
from vllm_ascend.distributed.eplb.policy.stair_search import replica_candidates
from vllm_ascend.distributed.eplb.policy.stair_stats import placement_score, risk_weights, weighted_moments
from vllm_ascend.distributed.eplb.policy.stair_types import BalanceScore, LayerPlan, RankTopology


def config_digest(config: StairConfig) -> str:
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def passes_hysteresis(current_score: float, anchor: float | None, config: StairConfig) -> bool:
    if not config.hysteresis_enabled or anchor is None:
        return True
    current_balance = 1.0 / current_score
    anchor_balance = 1.0 / anchor
    return current_balance / anchor_balance <= config.hysteresis_relative or current_balance <= config.hysteresis_absolute


def _candidate_key(result: PlacementResult) -> tuple:
    return (
        result.cross_node_transfers,
        result.intra_node_transfers,
        result.cross_node_transfers + result.intra_node_transfers,
        tuple(result.placement.ravel()),
        tuple(result.source_rank.ravel()),
        tuple(result.source_slot.ravel()),
    )


def plan_layer(
    layer_idx: int,
    samples: np.ndarray,
    weights: np.ndarray,
    old_placement: np.ndarray,
    epoch: int,
    topology: RankTopology,
    config: StairConfig,
) -> LayerPlan | None:
    current = placement_score(samples, weights, old_placement)
    mean, moments = weighted_moments(samples, weights, covariance=config.use_covariance)
    risk = risk_weights(mean, moments, config.z_score)

    def screening(replicas: np.ndarray) -> float:
        placement = unconstrained_lpt(mean, moments, replicas, old_placement.shape[0], config.z_score)
        return placement_score(samples, weights, placement).mean

    replica_vectors = replica_candidates(
        risk,
        old_placement.size,
        old_placement.shape[0],
        depth=config.experimental_flash_tree_depth,
        width=config.experimental_flash_tree_width,
        max_candidates=config.experimental_max_candidates_per_layer,
        screening_score=screening,
    )
    candidates: list[tuple[float, PlacementResult, BalanceScore]] = []
    for replicas in replica_vectors:
        result = constrained_lpt(
            mean,
            moments,
            replicas,
            old_placement,
            topology,
            z_score=config.z_score,
            pair_cap=config.max_expert_transfers_per_rank_pair,
            max_backtracks=config.experimental_lpt_max_backtracks,
        )
        if result is None:
            continue
        score = placement_score(samples, weights, result.placement)
        relative_gain = (current.mean - score.mean) / current.mean
        absolute_gain = current.mean - score.mean
        if (
            relative_gain >= config.min_relative_score_improvement
            and absolute_gain >= config.min_absolute_score_improvement
            and score.p95 <= current.p95 * (1 + config.p95_regression_tolerance)
        ):
            candidates.append((score.mean, result, score))
    if not candidates:
        return None
    minimum = min(score for score, _, _ in candidates)
    tied = [item for item in candidates if item[0] <= minimum + config.experimental_score_tie_tolerance]
    _, result, candidate_score = min(tied, key=lambda item: _candidate_key(item[1]))
    return LayerPlan(
        layer_idx,
        np.asarray(old_placement, dtype=np.int64).copy(),
        result.placement,
        result.source_rank,
        result.source_slot,
        epoch,
        current,
        candidate_score,
    )
