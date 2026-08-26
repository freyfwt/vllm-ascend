# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Topology-aware deterministic placement search for STAIR EPLB."""

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vllm_ascend.distributed.eplb.policy import stair_constants as constants
from vllm_ascend.distributed.eplb.policy.stair_stats import compute_balance_metrics, replica_counts
from vllm_ascend.distributed.eplb.policy.stair_types import StairTopology


def validate_layer_placement(placement: np.ndarray, num_experts: int, num_ranks: int) -> None:
    """Validate the placement invariants required by the vLLM executor."""
    if placement.ndim != 2 or placement.shape[0] != num_ranks:
        raise ValueError("placement must have shape [num_ranks, slots_per_rank].")
    if np.any(placement < 0) or np.any(placement >= num_experts):
        raise ValueError("placement contains an invalid logical expert index.")
    counts = replica_counts(placement, num_experts)
    if np.any(counts == 0):
        raise ValueError("Every logical expert must have at least one physical replica.")
    if np.any(counts > num_ranks):
        raise ValueError("A logical expert cannot have more replicas than ranks.")
    for rank in placement:
        if np.unique(rank).size != rank.size:
            raise ValueError("A logical expert cannot have two replicas on the same rank.")


def allocate_replica_counts(
    risk_load: np.ndarray,
    num_replicas: int,
    num_ranks: int,
) -> npt.NDArray[np.int64]:
    """Allocate redundant slots to the hottest per-replica risk load."""
    num_experts = risk_load.size
    counts: npt.NDArray[np.int64] = np.ones(num_experts, dtype=np.int64)
    per_replica_load: npt.NDArray[np.float64] = risk_load.astype(np.float64, copy=True)
    for _ in range(num_replicas - num_experts):
        allocated = False
        for expert_id in np.argsort(per_replica_load, kind="stable")[::-1]:
            if counts[expert_id] >= num_ranks:
                continue
            counts[expert_id] += 1
            per_replica_load[expert_id] = risk_load[expert_id] / counts[expert_id]
            allocated = True
            break
        if not allocated:
            raise ValueError("The requested replica count cannot be placed without duplicate experts on a rank.")
    return counts


@dataclass
class _PlacementWorkspace:
    assignments: npt.NDArray[np.int64]
    empty_slots: list[list[int]]
    remaining_counts: npt.NDArray[np.int64]
    rank_loads: npt.NDArray[np.float64]
    node_loads: npt.NDArray[np.float64]
    source_catalog: tuple[tuple[int, ...], ...]


def _build_source_catalog(current: np.ndarray, num_experts: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(rank for rank in range(current.shape[0]) if expert_id in current[rank])
        for expert_id in range(num_experts)
    )


def _reserve_resident_replicas(
    current: np.ndarray,
    target_counts: np.ndarray,
    risk_load: np.ndarray,
    topology: StairTopology,
) -> _PlacementWorkspace:
    """Create a workspace whose assignment costs favor resident copies."""
    num_ranks, slots_per_rank = current.shape
    num_experts = target_counts.size
    assignments = np.full_like(current, -1)
    rank_loads = np.zeros(num_ranks, dtype=np.float64)
    node_loads: npt.NDArray[np.float64] = np.zeros(topology.num_nodes, dtype=np.float64)
    empty_slots = [list(range(slots_per_rank)) for _ in range(num_ranks)]
    return _PlacementWorkspace(
        assignments=assignments,
        empty_slots=empty_slots,
        remaining_counts=target_counts.copy(),
        rank_loads=rank_loads,
        node_loads=node_loads,
        source_catalog=_build_source_catalog(current, num_experts),
    )


def _transfer_tier(
    source_ranks: tuple[int, ...],
    dst_rank: int,
    topology: StairTopology,
) -> int:
    if dst_rank in source_ranks:
        return 0
    if any(topology.same_node(src_rank, dst_rank) for src_rank in source_ranks):
        return 1
    return 2


def _assign_replica_ranks(
    workspace: _PlacementWorkspace,
    target_counts: np.ndarray,
    risk_load: np.ndarray,
    topology: StairTopology,
) -> npt.NDArray[np.int64] | None:
    """Assign missing replicas to low-load ranks with topology-local sources."""
    target_load = risk_load / target_counts
    ranks_per_node = np.bincount(topology.rank_to_node, minlength=topology.num_nodes)
    while np.any(workspace.remaining_counts > 0):
        candidates_by_expert = {
            expert_id: [
                rank
                for rank, slots in enumerate(workspace.empty_slots)
                if slots and expert_id not in workspace.assignments[rank]
            ]
            for expert_id in np.flatnonzero(workspace.remaining_counts > 0)
        }
        if any(
            len(candidates) < workspace.remaining_counts[expert_id]
            for expert_id, candidates in candidates_by_expert.items()
        ):
            return None
        expert_id = min(
            candidates_by_expert,
            key=lambda candidate_expert: (
                len(candidates_by_expert[candidate_expert]) - workspace.remaining_counts[candidate_expert],
                -target_load[candidate_expert],
                candidate_expert,
            ),
        )
        candidates = candidates_by_expert[expert_id]

        def candidate_key(
            rank: int,
            current_expert_id: int = expert_id,
        ) -> tuple[float, float, float, int]:
            node_id = topology.rank_to_node[rank]
            transfer_tier = _transfer_tier(workspace.source_catalog[current_expert_id], rank, topology)
            transfer_cost = (
                constants.INTRA_NODE_COST_MULTIPLIER if transfer_tier <= 1 else constants.CROSS_NODE_COST_MULTIPLIER
            )
            normalized_node_load = workspace.node_loads[node_id] / ranks_per_node[node_id]
            return transfer_cost, normalized_node_load, workspace.rank_loads[rank], rank

        rank = min(candidates, key=candidate_key)
        slot = workspace.empty_slots[rank].pop(0)
        workspace.assignments[rank, slot] = expert_id
        workspace.rank_loads[rank] += target_load[expert_id]
        workspace.node_loads[topology.rank_to_node[rank]] += target_load[expert_id]
        workspace.remaining_counts[expert_id] -= 1
    return workspace.assignments


def _improve_by_swaps(
    candidate: np.ndarray,
    risk_load: np.ndarray,
    topology: StairTopology,
) -> np.ndarray:
    """Bounded local search that prices cross-node swaps conservatively."""
    improved = candidate.copy()
    counts = replica_counts(improved, risk_load.size)
    per_replica_load = risk_load / counts
    rank_load = np.sum(per_replica_load[improved], axis=1)
    minimum_gain = risk_load.sum() / improved.shape[0] * constants.SWAP_IMPROVEMENT_RATIO
    for _ in range(constants.MAX_SWAP_ATTEMPTS):
        best: tuple[float, int, int, int, int] | None = None
        hot_ranks = np.argsort(rank_load, kind="stable")[::-1]
        for hot_rank_value in hot_ranks:
            hot_rank = int(hot_rank_value)
            for cold_rank_value in np.argsort(rank_load, kind="stable"):
                cold_rank = int(cold_rank_value)
                if cold_rank == hot_rank or rank_load[cold_rank] >= rank_load[hot_rank]:
                    continue
                cost_multiplier = (
                    constants.INTRA_NODE_COST_MULTIPLIER
                    if topology.same_node(hot_rank, cold_rank)
                    else constants.CROSS_NODE_COST_MULTIPLIER
                )
                for hot_slot, hot_expert in enumerate(improved[hot_rank]):
                    if hot_expert in improved[cold_rank]:
                        continue
                    for cold_slot, cold_expert in enumerate(improved[cold_rank]):
                        if cold_expert in improved[hot_rank]:
                            continue
                        hot_after = rank_load[hot_rank] - per_replica_load[hot_expert] + per_replica_load[cold_expert]
                        cold_after = rank_load[cold_rank] - per_replica_load[cold_expert] + per_replica_load[hot_expert]
                        gain = rank_load[hot_rank] - max(hot_after, cold_after)
                        if gain <= minimum_gain * cost_multiplier:
                            continue
                        value = (gain / cost_multiplier, -hot_rank, -cold_rank, -hot_slot, -cold_slot)
                        if best is None or value > best:
                            best = value
        if best is None:
            break
        _, neg_hot_rank, neg_cold_rank, neg_hot_slot, neg_cold_slot = best
        hot_rank, cold_rank = -neg_hot_rank, -neg_cold_rank
        hot_slot, cold_slot = -neg_hot_slot, -neg_cold_slot
        hot_expert = improved[hot_rank, hot_slot]
        cold_expert = improved[cold_rank, cold_slot]
        improved[hot_rank, hot_slot], improved[cold_rank, cold_slot] = cold_expert, hot_expert
        rank_load[hot_rank] += per_replica_load[cold_expert] - per_replica_load[hot_expert]
        rank_load[cold_rank] += per_replica_load[hot_expert] - per_replica_load[cold_expert]
    return improved


def _build_rank_cost_matrix(current: np.ndarray, desired: np.ndarray) -> npt.NDArray[np.float64]:
    """Cost desired rows by experts that are not resident on a physical rank."""
    cost = np.zeros((desired.shape[0], current.shape[0]), dtype=np.float64)
    for desired_idx, desired_row in enumerate(desired):
        desired_set = set(int(expert) for expert in desired_row)
        for current_idx, current_row in enumerate(current):
            cost[desired_idx, current_idx] = len(desired_set - set(int(expert) for expert in current_row))
    return cost


def _solve_linear_assignment(
    cost: np.ndarray,
    deadline: float | None = None,
) -> npt.NDArray[np.int64] | None:
    """Solve a square assignment deterministically with O(R^3) memory-bounded work."""
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or not np.all(np.isfinite(cost)):
        return None
    size = cost.shape[0]
    row_potential = np.zeros(size + 1, dtype=np.float64)
    col_potential = np.zeros(size + 1, dtype=np.float64)
    matching = np.zeros(size + 1, dtype=np.int64)
    path = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        matching[0] = row
        column = 0
        minimum = np.full(size + 1, np.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=np.bool_)
        while True:
            used[column] = True
            matched_row = matching[column]
            delta = np.inf
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                reduced_cost = (
                    cost[matched_row - 1, candidate_column - 1]
                    - row_potential[matched_row]
                    - col_potential[candidate_column]
                )
                if reduced_cost < minimum[candidate_column]:
                    minimum[candidate_column] = reduced_cost
                    path[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            if not np.isfinite(delta):
                return None
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    row_potential[matching[candidate_column]] += delta
                    col_potential[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matching[column] == 0:
                break
        while True:
            previous_column = path[column]
            matching[column] = matching[previous_column]
            column = previous_column
            if column == 0:
                break
    assignment = np.empty(size, dtype=np.int64)
    for column in range(1, size + 1):
        assignment[matching[column] - 1] = column - 1
    return assignment


def _match_equivalent_ranks(
    current: np.ndarray,
    desired: np.ndarray,
    topology: StairTopology,
    deadline: float | None = None,
) -> np.ndarray:
    """Reorder only topology-equivalent rows to reduce remote expert moves."""
    matched = desired.copy()
    for group in topology.equivalent_rank_groups:
        if len(group) <= 1:
            continue
        group_indices = np.asarray(group, dtype=np.int64)
        current_group = current[group_indices]
        desired_group = desired[group_indices]
        assignment = _solve_linear_assignment(
            _build_rank_cost_matrix(current_group, desired_group),
            deadline,
        )
        if assignment is None:
            continue
        for desired_idx, physical_idx in enumerate(assignment):
            matched[group_indices[physical_idx]] = desired_group[desired_idx]
    return matched


def align_local_slots(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Keep resident experts in their original slot before stable filling."""
    aligned = np.full_like(current, -1)
    for rank_idx in range(current.shape[0]):
        desired_experts = set(int(expert) for expert in desired[rank_idx])
        for slot_idx, expert_id in enumerate(current[rank_idx]):
            if int(expert_id) in desired_experts:
                aligned[rank_idx, slot_idx] = expert_id
                desired_experts.remove(int(expert_id))
        replacements = iter(sorted(desired_experts))
        for slot_idx in np.flatnonzero(aligned[rank_idx] < 0):
            aligned[rank_idx, slot_idx] = next(replacements)
    return aligned


def build_greedy_layer_candidate(
    risk_load: np.ndarray,
    current: np.ndarray,
    topology: StairTopology,
    *,
    deadline: float | None = None,
) -> np.ndarray:
    """Build one complete topology-aware candidate or return current."""
    num_ranks = current.shape[0]
    num_experts = risk_load.size
    validate_layer_placement(current, num_experts, num_ranks)
    target_counts = allocate_replica_counts(risk_load, current.size, num_ranks)
    workspace = _reserve_resident_replicas(current, target_counts, risk_load, topology)
    assigned = _assign_replica_ranks(workspace, target_counts, risk_load, topology)
    if assigned is None:
        return current.copy()
    desired = _improve_by_swaps(assigned, risk_load, topology)
    matched = _match_equivalent_ranks(current, desired, topology, deadline)
    candidate = align_local_slots(current, matched)
    validate_layer_placement(candidate, num_experts, num_ranks)
    current_score = compute_balance_metrics(risk_load[None, :], current).mean_imbalance
    candidate_score = compute_balance_metrics(risk_load[None, :], candidate).mean_imbalance
    return candidate if candidate_score < current_score else current.copy()


def build_greedy_candidate(
    risk_load: np.ndarray,
    current_placement: np.ndarray,
    topology: StairTopology,
    *,
    deadline: float | None = None,
) -> np.ndarray:
    """Build topology-aware candidates for all MoE layers."""
    num_layers = risk_load.shape[0]
    num_ranks = len(topology.rank_to_node)
    slots_per_rank = current_placement.shape[1] // num_ranks
    current_by_rank = current_placement.reshape(num_layers, num_ranks, slots_per_rank)
    candidate = current_by_rank.copy()
    for layer_idx in range(num_layers):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        current_score = compute_balance_metrics(
            risk_load[layer_idx : layer_idx + 1],
            current_by_rank[layer_idx],
        ).mean_imbalance
        if current_score < constants.IMBALANCE_THRESHOLD:
            continue
        candidate[layer_idx] = build_greedy_layer_candidate(
            risk_load[layer_idx],
            current_by_rank[layer_idx],
            topology,
            deadline=deadline,
        )
    return candidate.reshape(current_placement.shape)
