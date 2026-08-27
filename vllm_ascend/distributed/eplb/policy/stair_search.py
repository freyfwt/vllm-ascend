# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Topology-aware deterministic placement search for STAIR EPLB."""

import time
from collections import Counter
from collections.abc import Callable
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
    replaceable_slots: list[list[int]]
    replaceable_positions: tuple[tuple[int, int], ...]
    rank_loads: npt.NDArray[np.float64]
    node_loads: npt.NDArray[np.float64]
    source_catalog: tuple[tuple[int, ...], ...]
    pair_counts: Counter[tuple[int, int]]


@dataclass(frozen=True)
class SearchCandidate:
    """One bounded replica-count candidate evaluated by the policy."""

    placement: np.ndarray
    replica_counts: np.ndarray
    objective: float
    path_key: tuple[int, ...]


@dataclass(frozen=True)
class MiniFlashTreeResult:
    candidate: SearchCandidate
    evaluated_candidates: int
    elapsed_ms: float
    timed_out: bool


def _build_source_catalog(current: np.ndarray, num_experts: int) -> tuple[tuple[int, ...], ...]:
    sources: list[list[int]] = [[] for _ in range(num_experts)]
    for rank_idx, rank_experts in enumerate(current):
        for expert_id in rank_experts:
            sources[int(expert_id)].append(rank_idx)
    return tuple(tuple(ranks) for ranks in sources)


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _reserve_resident_replicas(
    current: np.ndarray,
    target_counts: np.ndarray,
    risk_load: np.ndarray,
    topology: StairTopology,
) -> _PlacementWorkspace:
    """Lock one stable resident copy and expose only redundant slots."""
    num_ranks, slots_per_rank = current.shape
    num_experts = target_counts.size
    assignments = current.copy()
    replaceable_slots: list[list[int]] = [[] for _ in range(num_ranks)]
    primary_positions: set[tuple[int, int]] = set()
    seen: set[int] = set()
    for slot_idx in range(slots_per_rank):
        for rank_idx in range(num_ranks):
            expert_id = int(current[rank_idx, slot_idx])
            if expert_id in seen:
                replaceable_slots[rank_idx].append(slot_idx)
            else:
                seen.add(expert_id)
                primary_positions.add((rank_idx, slot_idx))
    if len(seen) != num_experts:
        raise ValueError("Every logical expert must have at least one physical replica.")

    replaceable_positions = tuple(
        (rank_idx, slot_idx) for rank_idx, slots in enumerate(replaceable_slots) for slot_idx in slots
    )
    if len(replaceable_positions) != int(target_counts.sum()) - num_experts:
        raise ValueError("STAIR target counts do not match the available redundant slots.")
    for rank_idx, slot_idx in replaceable_positions:
        assignments[rank_idx, slot_idx] = -1

    target_load = risk_load / target_counts
    rank_loads = np.zeros(num_ranks, dtype=np.float64)
    node_loads: npt.NDArray[np.float64] = np.zeros(topology.num_nodes, dtype=np.float64)
    for rank_idx, slot_idx in primary_positions:
        expert_id = int(assignments[rank_idx, slot_idx])
        rank_loads[rank_idx] += target_load[expert_id]
        node_loads[topology.rank_to_node[rank_idx]] += target_load[expert_id]
    return _PlacementWorkspace(
        assignments=assignments,
        replaceable_slots=replaceable_slots,
        replaceable_positions=replaceable_positions,
        rank_loads=rank_loads,
        node_loads=node_loads,
        source_catalog=_build_source_catalog(current, num_experts),
        pair_counts=Counter(),
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
    deadline: float | None = None,
) -> npt.NDArray[np.int64] | None:
    """Place only redundant replicas with topology and pair budgets."""
    target_load = risk_load / target_counts
    ranks_per_node = np.bincount(topology.rank_to_node, minlength=topology.num_nodes)
    replicas_to_place = [expert_id for expert_id, count in enumerate(target_counts) for _ in range(int(count) - 1)]
    replicas_to_place.sort(key=lambda expert_id: (-target_load[expert_id], expert_id))
    assigned_counts: npt.NDArray[np.int64] = np.ones(target_counts.size, dtype=np.int64)

    for requested_expert in replicas_to_place:
        if _deadline_expired(deadline):
            return None
        fallback_experts = sorted(
            (expert_id for expert_id in range(target_counts.size) if expert_id != requested_expert),
            key=lambda expert_id: (
                assigned_counts[expert_id] >= target_counts[expert_id],
                -risk_load[expert_id] / (assigned_counts[expert_id] + 1),
                expert_id,
            ),
        )
        chosen: tuple[int, int, int] | None = None
        for expert_id in (requested_expert, *fallback_experts):
            candidates: list[tuple[tuple[float, float, float, int, int], int, int]] = []
            for rank, slots in enumerate(workspace.replaceable_slots):
                if not slots or expert_id in workspace.assignments[rank]:
                    continue
                source_options = []
                for source_rank in workspace.source_catalog[expert_id]:
                    pair_count = workspace.pair_counts[(source_rank, rank)]
                    if source_rank != rank and pair_count >= constants.MAX_TRANSFERS_PER_RANK_PAIR:
                        continue
                    source_options.append(
                        (
                            _transfer_tier((source_rank,), rank, topology),
                            pair_count,
                            source_rank,
                        )
                    )
                if not source_options:
                    continue
                transfer_tier, pair_count, source_rank = min(source_options)
                node_id = topology.rank_to_node[rank]
                normalized_node_load = workspace.node_loads[node_id] / ranks_per_node[node_id]
                key = (
                    float(transfer_tier),
                    normalized_node_load,
                    workspace.rank_loads[rank],
                    pair_count,
                    rank,
                )
                candidates.append((key, rank, source_rank))
            if candidates:
                _, rank, source_rank = min(candidates, key=lambda item: item[0])
                chosen = expert_id, rank, source_rank
                break
        if chosen is None:
            return None
        expert_id, rank, source_rank = chosen
        slot = workspace.replaceable_slots[rank].pop(0)
        workspace.assignments[rank, slot] = expert_id
        assigned_counts[expert_id] += 1
        per_replica_load = risk_load / assigned_counts
        workspace.rank_loads.fill(0.0)
        workspace.node_loads.fill(0.0)
        for assigned_rank, rank_experts in enumerate(workspace.assignments):
            valid_experts = rank_experts[rank_experts >= 0]
            workspace.rank_loads[assigned_rank] = per_replica_load[valid_experts].sum()
            workspace.node_loads[topology.rank_to_node[assigned_rank]] += workspace.rank_loads[assigned_rank]
        if source_rank != rank:
            workspace.pair_counts[(source_rank, rank)] += 1
    return workspace.assignments


def _improve_by_swaps(
    candidate: np.ndarray,
    risk_load: np.ndarray,
    topology: StairTopology,
    protected_positions: tuple[tuple[int, int], ...] | None = None,
    reserved_pair_counts: Counter[tuple[int, int]] | None = None,
    deadline: float | None = None,
) -> np.ndarray:
    """Relocate redundant capacity with a bounded hot/cold rank search."""
    improved = candidate.copy()
    protected = set(protected_positions or ())
    if not protected:
        return improved
    counts = replica_counts(improved, risk_load.size)
    per_replica_load = risk_load / counts
    rank_load = np.sum(per_replica_load[improved], axis=1)
    pair_counts = Counter(reserved_pair_counts or {})
    minimum_gain = risk_load.sum() / improved.shape[0] * constants.SWAP_IMPROVEMENT_RATIO
    rank_experts = [set(int(expert) for expert in rank) for rank in improved]
    attempt_limit = min(constants.MAX_SWAP_ATTEMPTS, len(protected) * 2)
    for _ in range(attempt_limit):
        if _deadline_expired(deadline):
            return candidate.copy()
        best: tuple[float, int, int, int, int] | None = None
        hot_rank = int(np.argmax(rank_load))
        for cold_rank_value in np.argsort(rank_load, kind="stable"):
            cold_rank = int(cold_rank_value)
            if cold_rank == hot_rank or rank_load[cold_rank] >= rank_load[hot_rank]:
                continue
            if (
                pair_counts[(hot_rank, cold_rank)] >= constants.MAX_TRANSFERS_PER_RANK_PAIR
                or pair_counts[(cold_rank, hot_rank)] >= constants.MAX_TRANSFERS_PER_RANK_PAIR
            ):
                continue
            cost_multiplier = (
                constants.INTRA_NODE_COST_MULTIPLIER
                if topology.same_node(hot_rank, cold_rank)
                else constants.CROSS_NODE_COST_MULTIPLIER
            )
            hot_slots = [
                int(slot)
                for slot in np.argsort(per_replica_load[improved[hot_rank]], kind="stable")[::-1]
                if (hot_rank, int(slot)) not in protected
            ][: constants.MAX_SWAP_SLOT_CANDIDATES]
            cold_slots = [
                int(slot)
                for slot in np.argsort(per_replica_load[improved[cold_rank]], kind="stable")
                if (cold_rank, int(slot)) not in protected
            ][: constants.MAX_SWAP_SLOT_CANDIDATES]
            for hot_slot in hot_slots:
                hot_expert_value = improved[hot_rank, hot_slot]
                hot_expert = int(hot_expert_value)
                if hot_expert in rank_experts[cold_rank]:
                    continue
                for cold_slot in cold_slots:
                    cold_expert_value = improved[cold_rank, cold_slot]
                    cold_expert = int(cold_expert_value)
                    if cold_expert in rank_experts[hot_rank]:
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
        hot_expert = int(improved[hot_rank, hot_slot])
        cold_expert = int(improved[cold_rank, cold_slot])
        improved[hot_rank, hot_slot], improved[cold_rank, cold_slot] = cold_expert, hot_expert
        pair_counts[(hot_rank, cold_rank)] += 1
        pair_counts[(cold_rank, hot_rank)] += 1
        rank_experts[hot_rank].remove(hot_expert)
        rank_experts[hot_rank].add(cold_expert)
        rank_experts[cold_rank].remove(cold_expert)
        rank_experts[cold_rank].add(hot_expert)
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
    return build_candidate_from_replica_counts(
        risk_load,
        current,
        topology,
        target_counts,
        deadline=deadline,
    )


def build_candidate_from_replica_counts(
    risk_load: np.ndarray,
    current: np.ndarray,
    topology: StairTopology,
    target_counts: np.ndarray,
    *,
    deadline: float | None = None,
) -> np.ndarray:
    """Reuse the production placement backend for an explicit count vector."""
    num_ranks = current.shape[0]
    num_experts = risk_load.size
    if (
        target_counts.shape != (num_experts,)
        or int(target_counts.sum()) != current.size
        or np.any(target_counts < 1)
        or np.any(target_counts > num_ranks)
    ):
        raise ValueError("STAIR search replica counts violate placement bounds.")
    if _deadline_expired(deadline):
        return current.copy()
    workspace = _reserve_resident_replicas(current, target_counts, risk_load, topology)
    assigned = _assign_replica_ranks(
        workspace,
        target_counts,
        risk_load,
        topology,
        deadline,
    )
    if assigned is None or _deadline_expired(deadline):
        return current.copy()
    desired = _improve_by_swaps(
        assigned,
        risk_load,
        topology,
        workspace.replaceable_positions,
        workspace.pair_counts,
        deadline,
    )
    if _deadline_expired(deadline):
        return current.copy()
    matched = _match_equivalent_ranks(current, desired, topology, deadline)
    if _deadline_expired(deadline):
        return current.copy()
    candidate = align_local_slots(current, matched)
    validate_layer_placement(candidate, num_experts, num_ranks)
    current_score = compute_balance_metrics(risk_load[None, :], current).mean_imbalance
    candidate_score = compute_balance_metrics(risk_load[None, :], candidate).mean_imbalance
    return candidate if candidate_score < current_score else current.copy()


def _replica_count_neighbors(
    counts: np.ndarray,
    risk_load: np.ndarray,
    num_ranks: int,
    width: int,
) -> tuple[np.ndarray, ...]:
    """Move one redundant copy along the highest marginal-gain edges."""
    neighbors: list[tuple[float, int, int, np.ndarray]] = []
    for donor in np.flatnonzero(counts > 1):
        donor_penalty = risk_load[donor] / (counts[donor] - 1) - risk_load[donor] / counts[donor]
        for receiver in np.flatnonzero(counts < num_ranks):
            if receiver == donor:
                continue
            receiver_gain = risk_load[receiver] / counts[receiver] - risk_load[receiver] / (counts[receiver] + 1)
            neighbor = counts.copy()
            neighbor[donor] -= 1
            neighbor[receiver] += 1
            neighbors.append((float(receiver_gain - donor_penalty), int(receiver), int(donor), neighbor))
    neighbors.sort(key=lambda item: (-item[0], item[1], item[2]))
    unique: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for _gain, _receiver, _donor, neighbor in neighbors:
        key = tuple(int(value) for value in neighbor)
        if key in seen:
            continue
        seen.add(key)
        unique.append(neighbor)
        if len(unique) >= width:
            break
    return tuple(unique)


class MiniFlashTreeSearch:
    """Small deterministic beam search around the greedy count vector."""

    def search(
        self,
        root: SearchCandidate,
        risk_load: np.ndarray,
        num_ranks: int,
        evaluator: Callable[[np.ndarray, float], SearchCandidate | None],
        deadline: float,
    ) -> MiniFlashTreeResult:
        start = time.perf_counter()
        best = root
        beam = [root]
        visited = {root.path_key}
        evaluated = 1
        timed_out = False
        for _depth in range(constants.SEARCH_DEPTH):
            next_beam: list[SearchCandidate] = []
            for node in beam:
                for neighbor_counts in _replica_count_neighbors(
                    node.replica_counts,
                    risk_load,
                    num_ranks,
                    constants.SEARCH_WIDTH,
                ):
                    if evaluated >= constants.MAX_CANDIDATES_PER_LAYER or time.perf_counter() >= deadline:
                        timed_out = time.perf_counter() >= deadline
                        break
                    path_key = tuple(int(value) for value in neighbor_counts)
                    if path_key in visited:
                        continue
                    visited.add(path_key)
                    evaluation = evaluator(neighbor_counts, deadline)
                    evaluated += 1
                    if time.perf_counter() >= deadline:
                        timed_out = True
                        break
                    if evaluation is None:
                        continue
                    next_beam.append(evaluation)
                    if (evaluation.objective, tuple(-value for value in evaluation.path_key)) > (
                        best.objective,
                        tuple(-value for value in best.path_key),
                    ):
                        best = evaluation
                if evaluated >= constants.MAX_CANDIDATES_PER_LAYER or timed_out:
                    break
            if not next_beam or timed_out:
                break
            next_beam.sort(key=lambda item: (-item.objective, item.path_key))
            beam = next_beam[: constants.SEARCH_WIDTH]
        return MiniFlashTreeResult(
            candidate=root if timed_out else best,
            evaluated_candidates=evaluated,
            elapsed_ms=(time.perf_counter() - start) * 1000,
            timed_out=timed_out,
        )


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
    current_scores = [
        compute_balance_metrics(
            risk_load[layer_idx : layer_idx + 1],
            current_by_rank[layer_idx],
        ).mean_imbalance
        for layer_idx in range(num_layers)
    ]
    layer_order = sorted(
        range(num_layers),
        key=lambda layer_idx: (-current_scores[layer_idx], layer_idx),
    )
    for layer_idx in layer_order:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if current_scores[layer_idx] < constants.IMBALANCE_THRESHOLD:
            continue
        candidate[layer_idx] = build_greedy_layer_candidate(
            risk_load[layer_idx],
            current_by_rank[layer_idx],
            topology,
            deadline=deadline,
        )
    return candidate.reshape(current_placement.shape)
