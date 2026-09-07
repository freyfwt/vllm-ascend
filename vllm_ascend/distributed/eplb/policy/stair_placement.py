# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Risk-aware LPT placement with a source-feasibility oracle."""

from dataclasses import dataclass

import numpy as np

from vllm_ascend.distributed.eplb.policy.stair_source import align_slots, assign_sources
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology


@dataclass(frozen=True)
class PlacementResult:
    placement: np.ndarray
    source_rank: np.ndarray
    source_slot: np.ndarray
    cross_node_transfers: int
    intra_node_transfers: int


def _post_insert_risk(
    experts: set[int],
    candidate: int,
    mean: np.ndarray,
    moments: np.ndarray,
    replicas: np.ndarray,
    z_score: float,
) -> float:
    selected = sorted((*experts, candidate))
    counts = replicas[selected].astype(np.float64)
    rank_mean = float(np.sum(mean[selected] / counts, dtype=np.float64))
    if moments.ndim == 1:
        variance = float(np.sum(moments[selected] / np.square(counts), dtype=np.float64))
    else:
        covariance = moments[np.ix_(selected, selected)]
        variance = float(np.sum(covariance / np.outer(counts, counts), dtype=np.float64))
    risk = rank_mean + z_score * np.sqrt(max(variance, 0.0))
    if not np.isfinite(risk):
        raise ValueError("STAIR rank risk must be finite")
    return risk


def _copies(mean: np.ndarray, moments: np.ndarray, replicas: np.ndarray, z_score: float) -> list[int]:
    diagonal = np.diag(moments) if moments.ndim == 2 else moments
    risk = mean + z_score * np.sqrt(np.maximum(diagonal, 0.0))
    copies = [(float(risk[expert] / replicas[expert]), expert, ordinal) for expert in range(len(mean)) for ordinal in range(replicas[expert])]
    copies.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [expert for _, expert, _ in copies]


def unconstrained_lpt(
    mean: np.ndarray,
    moments: np.ndarray,
    replicas: np.ndarray,
    num_ranks: int,
    z_score: float,
) -> np.ndarray:
    """Build the deterministic source-agnostic placement used for screening."""
    total_slots = int(np.sum(replicas))
    if total_slots % num_ranks:
        raise ValueError("STAIR physical slots must divide evenly across ranks")
    slots_per_rank = total_slots // num_ranks
    ranks = [set() for _ in range(num_ranks)]
    for expert in _copies(mean, moments, replicas, z_score):
        candidates = [rank for rank in range(num_ranks) if len(ranks[rank]) < slots_per_rank and expert not in ranks[rank]]
        if not candidates:
            raise ValueError("STAIR replica vector has no duplicate-free LPT placement")
        rank = min(
            candidates,
            key=lambda item: (_post_insert_risk(ranks[item], expert, mean, moments, replicas, z_score), item),
        )
        ranks[rank].add(expert)
    return np.asarray([sorted(experts) for experts in ranks], dtype=np.int64)


def constrained_lpt(
    mean: np.ndarray,
    moments: np.ndarray,
    replicas: np.ndarray,
    old_placement: np.ndarray,
    topology: RankTopology,
    *,
    z_score: float,
    pair_cap: int,
    max_backtracks: int,
) -> PlacementResult | None:
    """Place replicas with bounded DFS and exact partial source matching."""
    old = np.asarray(old_placement, dtype=np.int64)
    if int(np.sum(replicas)) != old.size or topology.num_ranks != old.shape[0]:
        raise ValueError("STAIR replica count, placement, and topology sizes disagree")
    ranks = [set() for _ in range(old.shape[0])]
    ordered_copies = _copies(mean, moments, replicas, z_score)
    backtracks = 0

    def search(index: int) -> dict[tuple[int, int], tuple[int, int]] | None:
        nonlocal backtracks
        if index == len(ordered_copies):
            return assign_sources(old, ranks, topology, pair_cap)
        expert = ordered_copies[index]
        candidates = [rank for rank in range(old.shape[0]) if len(ranks[rank]) < old.shape[1] and expert not in ranks[rank]]
        candidates.sort(
            key=lambda rank: (_post_insert_risk(ranks[rank], expert, mean, moments, replicas, z_score), rank)
        )
        for rank in candidates:
            ranks[rank].add(expert)
            feasible = assign_sources(old, ranks, topology, pair_cap) is not None
            result = search(index + 1) if feasible else None
            if result is not None:
                return result
            ranks[rank].remove(expert)
            if feasible:
                backtracks += 1
                if backtracks > max_backtracks:
                    return None
        return None

    sources = search(0)
    if sources is None:
        return None
    placement, source_rank, source_slot = align_slots(old, ranks, sources)
    cross = intra = 0
    for dst in range(old.shape[0]):
        for slot, src in enumerate(source_rank[dst]):
            if src != dst:
                if topology.same_node(int(src), dst):
                    intra += 1
                else:
                    cross += 1
    return PlacementResult(placement, source_rank, source_slot, cross, intra)
