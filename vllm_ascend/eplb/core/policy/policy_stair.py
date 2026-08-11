# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Statistical Temporal-Aware Incremental Rebalancing policy."""

from collections import deque

import numpy as np

BALANCE_EPSILON = 1e-6
MAX_REFINEMENT_STEPS = 100
Z_SCORE = 0.674
SWIFT_IMBALANCE_THRESHOLD = 1.01
SWIFT_MIN_SWAP_IMPROVEMENT_RATIO = 0.01
SWIFT_MAX_COMMUNICATIONS_PER_RANK_PAIR = 1
SWIFT_GLOBAL_IMPROVEMENT_RATIO = 0.05
FLASH_UPDATE_THRESHOLD_RATIO = 0.9
FLASH_UPDATE_THRESHOLD_VALUE = 0.85
FLASH_SMALL_WORLD_SIZE = 32
FLASH_SMALL_WORLD_UPDATE_THRESHOLD_RATIO = 0.95
FLASH_SMALL_WORLD_UPDATE_THRESHOLD_VALUE = 0.9


def _replica_counts(placement: np.ndarray, num_experts: int) -> np.ndarray:
    valid_experts = placement[placement >= 0]
    return np.bincount(valid_experts, minlength=num_experts).astype(np.int64, copy=False)


def _rank_loads(expert_load: np.ndarray, placement: np.ndarray, num_experts: int) -> np.ndarray:
    replica_counts = _replica_counts(placement, num_experts)
    if np.any(replica_counts == 0):
        raise ValueError("Every logical expert must have at least one physical replica.")

    rank_loads = np.zeros((expert_load.shape[0], placement.shape[0]), dtype=np.float64)
    for rank_id, rank in enumerate(placement):
        expert_ids = rank[rank >= 0]
        if expert_ids.size:
            rank_loads[:, rank_id] = np.sum(
                expert_load[:, expert_ids] / replica_counts[expert_ids],
                axis=1,
            )
    return rank_loads


def _score_rank_loads(rank_loads: np.ndarray) -> float:
    total_load = np.sum(rank_loads, axis=1)
    scores = np.ones(rank_loads.shape[0], dtype=np.float64)
    nonzero = total_load > 0
    if np.any(nonzero):
        average_load = total_load[nonzero] / rank_loads.shape[1]
        scores[nonzero] = np.max(rank_loads[nonzero], axis=1) / average_load
    return float(np.mean(scores))


def compute_balance_score(expert_load: np.ndarray, placement: np.ndarray) -> float:
    """Return the mean peak-to-average rank load for one MoE layer."""
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
    return _score_rank_loads(_rank_loads(expert_load, placement, expert_load.shape[1]))


def _aggregate_peak_and_average(
    expert_load: np.ndarray,
    placement: np.ndarray,
) -> tuple[float, float]:
    rank_loads = np.sum(
        _rank_loads(expert_load, placement, expert_load.shape[1]),
        axis=0,
    )
    return float(np.max(rank_loads)), float(np.mean(rank_loads))


def _compute_statistics(expert_load: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(expert_load, axis=0)
    if expert_load.shape[0] == 1:
        covariance = np.zeros((expert_load.shape[1], expert_load.shape[1]), dtype=np.float64)
    else:
        centered = expert_load - mean
        covariance = centered.T @ centered / (expert_load.shape[0] - 1)
    variance = np.maximum(np.diag(covariance), 0.0)
    return mean, variance, covariance


def allocate_replicas(
    mean: np.ndarray,
    variance: np.ndarray,
    num_replicas: int,
    num_ranks: int,
) -> np.ndarray:
    """Allocate replicas by per-copy statistical risk with a rank-count cap."""
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    if mean.ndim != 1 or variance.shape != mean.shape:
        raise ValueError("mean and variance must be one-dimensional arrays with the same shape.")
    if num_ranks <= 0:
        raise ValueError("num_ranks must be positive.")
    num_experts = mean.shape[0]
    if num_replicas < num_experts:
        raise ValueError("num_replicas must cover every logical expert.")
    if num_replicas > num_experts * num_ranks:
        raise ValueError("num_replicas exceeds the no-duplicate-per-rank capacity.")

    risk = mean + Z_SCORE * np.sqrt(np.maximum(variance, 0.0))
    replica_counts = np.ones(num_experts, dtype=np.int64)
    for _ in range(num_replicas - num_experts):
        unit_risk = np.full(num_experts, -np.inf, dtype=np.float64)
        eligible = replica_counts < num_ranks
        unit_risk[eligible] = risk[eligible] / replica_counts[eligible]
        expert_id = int(np.argmax(unit_risk))
        if not np.isfinite(unit_risk[expert_id]):
            raise ValueError("Unable to allocate all replicas under rank uniqueness constraints.")
        replica_counts[expert_id] += 1
    return replica_counts


def _validate_layer_placement(
    placement: np.ndarray,
    num_experts: int,
    num_ranks: int,
    target_replicas: np.ndarray | None = None,
) -> None:
    if placement.ndim != 2 or placement.shape[0] != num_ranks:
        raise ValueError("placement must have shape [num_ranks, slots_per_rank].")
    if np.any(placement < 0) or np.any(placement >= num_experts):
        raise ValueError("placement contains an invalid logical expert index.")
    replica_counts = _replica_counts(placement, num_experts)
    if np.any(replica_counts == 0):
        raise ValueError("Every logical expert must have at least one physical replica.")
    if np.any(replica_counts > num_ranks):
        raise ValueError("A logical expert cannot have more replicas than ranks.")
    for rank in placement:
        if np.unique(rank).size != rank.size:
            raise ValueError("A logical expert cannot have two replicas on the same rank.")
    if target_replicas is not None and not np.array_equal(replica_counts, target_replicas):
        raise ValueError("placement does not match the target replica counts.")


def _has_required_flow(
    graph: list[dict[int, int]],
    source: int,
    sink: int,
    required_flow: int,
) -> bool:
    flow = 0
    while flow < required_flow:
        parent = [-1] * len(graph)
        parent[source] = source
        queue = deque([source])
        while queue and parent[sink] == -1:
            node = queue.popleft()
            for neighbor in sorted(graph[node]):
                if parent[neighbor] == -1 and graph[node][neighbor] > 0:
                    parent[neighbor] = node
                    queue.append(neighbor)
                    if neighbor == sink:
                        break
        if parent[sink] == -1:
            return False

        increment = required_flow - flow
        node = sink
        while node != source:
            previous = parent[node]
            increment = min(increment, graph[previous][node])
            node = previous
        node = sink
        while node != source:
            previous = parent[node]
            graph[previous][node] -= increment
            graph[node][previous] += increment
            node = previous
        flow += increment
    return True


def _can_reach_target(placement: np.ndarray, target_replicas: np.ndarray) -> bool:
    """Check whether legal slot replacements can reach target replica counts."""
    num_ranks, slots_per_rank = placement.shape
    num_experts = target_replicas.shape[0]
    replica_counts = _replica_counts(placement, num_experts)
    deficits = np.maximum(target_replicas - replica_counts, 0)
    surpluses = np.maximum(replica_counts - target_replicas, 0)
    required_flow = int(np.sum(deficits))
    if required_flow != int(np.sum(surpluses)):
        return False
    if required_flow == 0:
        return True

    source = 0
    deficit_offset = 1
    slot_offset = deficit_offset + num_experts
    surplus_offset = slot_offset + placement.size
    sink = surplus_offset + num_experts
    graph: list[dict[int, int]] = [dict() for _ in range(sink + 1)]

    def add_edge(src: int, dst: int, capacity: int) -> None:
        graph[src][dst] = capacity
        graph[dst].setdefault(src, 0)

    for expert_id, deficit in enumerate(deficits):
        if deficit > 0:
            add_edge(source, deficit_offset + expert_id, int(deficit))

    for rank_id, rank in enumerate(placement):
        missing_experts = [
            expert_id for expert_id, deficit in enumerate(deficits) if deficit > 0 and expert_id not in rank
        ]
        for slot_id, surplus_expert in enumerate(rank):
            if surpluses[surplus_expert] <= 0:
                continue
            slot_node = slot_offset + rank_id * slots_per_rank + slot_id
            add_edge(slot_node, surplus_offset + int(surplus_expert), 1)
            for missing_expert in missing_experts:
                add_edge(deficit_offset + missing_expert, slot_node, 1)

    for expert_id, surplus in enumerate(surpluses):
        if surplus > 0:
            add_edge(surplus_offset + expert_id, sink, int(surplus))
    return _has_required_flow(graph, source, sink, required_flow)


def _rank_risk(
    rank: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    target_replicas: np.ndarray,
) -> float:
    expert_ids = rank[rank >= 0]
    if expert_ids.size == 0:
        return 0.0
    scales = 1.0 / target_replicas[expert_ids]
    rank_mean = float(np.sum(mean[expert_ids] * scales))
    rank_covariance = covariance[np.ix_(expert_ids, expert_ids)]
    rank_variance = float(scales @ rank_covariance @ scales)
    return rank_mean + Z_SCORE * np.sqrt(max(rank_variance, 0.0))


def _replace_surplus_replicas(
    expert_load: np.ndarray,
    placement: np.ndarray,
    target_replicas: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    proposal = placement.copy()
    num_experts = target_replicas.shape[0]
    risk = mean + Z_SCORE * np.sqrt(np.maximum(variance, 0.0))
    num_ranks = placement.shape[0]
    num_com_between_rank = np.zeros((num_ranks, num_ranks), dtype=np.int64)

    if not _can_reach_target(proposal, target_replicas):
        return None

    while True:
        replica_counts = _replica_counts(proposal, num_experts)
        deficits = target_replicas - replica_counts
        if not np.any(deficits > 0):
            return proposal, num_com_between_rank

        unit_risk = np.full(num_experts, -np.inf, dtype=np.float64)
        missing = deficits > 0
        unit_risk[missing] = risk[missing] / target_replicas[missing]
        expert_id = int(np.argmax(unit_risk))

        best_candidate: np.ndarray | None = None
        best_score = np.inf
        best_risk = np.inf
        best_position: tuple[int, int] | None = None
        best_send_rank: int | None = None
        for rank_id, rank in enumerate(proposal):
            if expert_id in rank:
                continue
            send_ranks = np.flatnonzero(np.any(proposal == expert_id, axis=1))
            send_ranks = send_ranks[num_com_between_rank[send_ranks, rank_id] < SWIFT_MAX_COMMUNICATIONS_PER_RANK_PAIR]
            if send_ranks.size == 0:
                continue
            send_rank = int(send_ranks[0])
            for slot_id, surplus_expert in enumerate(rank):
                if replica_counts[surplus_expert] <= target_replicas[surplus_expert]:
                    continue
                candidate = proposal.copy()
                candidate[rank_id, slot_id] = expert_id
                if not _can_reach_target(candidate, target_replicas):
                    continue
                score = compute_balance_score(expert_load, candidate)
                candidate_risk = _rank_risk(
                    candidate[rank_id],
                    mean,
                    covariance,
                    target_replicas,
                )
                position = (rank_id, slot_id)
                is_better = score < best_score - BALANCE_EPSILON
                if abs(score - best_score) <= BALANCE_EPSILON:
                    is_better = candidate_risk < best_risk - BALANCE_EPSILON or (
                        abs(candidate_risk - best_risk) <= BALANCE_EPSILON
                        and (best_position is None or position < best_position)
                    )
                if is_better:
                    best_candidate = candidate
                    best_score = score
                    best_risk = candidate_risk
                    best_position = position
                    best_send_rank = send_rank

        if best_candidate is None or best_position is None or best_send_rank is None:
            return None
        num_com_between_rank[best_send_rank, best_position[0]] += 1
        proposal = best_candidate


def _changed_slots(placement: np.ndarray, original_placement: np.ndarray) -> int:
    return int(np.count_nonzero(placement != original_placement))


def _refine_placement(
    expert_load: np.ndarray,
    placement: np.ndarray,
    original_placement: np.ndarray,
    target_replicas: np.ndarray,
    num_com_between_rank: np.ndarray,
) -> np.ndarray:
    proposal = placement.copy()
    unit_load = expert_load / target_replicas
    rank_loads = np.zeros((expert_load.shape[0], proposal.shape[0]), dtype=np.float64)
    for rank_id, rank in enumerate(proposal):
        rank_loads[:, rank_id] = np.sum(unit_load[:, rank], axis=1)
    current_score = _score_rank_loads(rank_loads)
    total_load = np.sum(rank_loads, axis=1)
    nonzero = total_load > 0
    average_load = np.zeros_like(total_load)
    average_load[nonzero] = total_load[nonzero] / proposal.shape[0]

    for _ in range(MAX_REFINEMENT_STEPS):
        aggregate_rank_loads = np.sum(rank_loads, axis=0)
        hottest_rank = int(np.argmax(aggregate_rank_loads))
        aggregate_peak = float(aggregate_rank_loads[hottest_rank])
        swap_threshold = float(np.mean(aggregate_rank_loads)) * SWIFT_MIN_SWAP_IMPROVEMENT_RATIO
        best_swap: tuple[int, int, int, int] | None = None
        best_score = current_score
        best_changed_slots = np.iinfo(np.int64).max
        source_experts = proposal[hottest_rank]
        source_units = unit_load[:, source_experts]
        current_changed_slots = _changed_slots(proposal, original_placement)

        for target_rank in range(proposal.shape[0]):
            if target_rank == hottest_rank:
                continue
            if (
                num_com_between_rank[target_rank, hottest_rank] >= SWIFT_MAX_COMMUNICATIONS_PER_RANK_PAIR
                or num_com_between_rank[hottest_rank, target_rank] >= SWIFT_MAX_COMMUNICATIONS_PER_RANK_PAIR
            ):
                continue
            target_experts = proposal[target_rank]
            target_units = unit_load[:, target_experts]

            # A swap is legal only when neither incoming expert is already on
            # the destination rank. Evaluate all legal slot pairs together.
            valid_sources = ~np.isin(source_experts, target_experts)
            valid_targets = ~np.isin(target_experts, source_experts)
            valid_swaps = valid_sources[:, None] & valid_targets[None, :]
            if not np.any(valid_swaps):
                continue

            delta = target_units[:, None, :] - source_units[:, :, None]
            if proposal.shape[0] == 2:
                # For two ranks, peak / average is exactly
                # 1 + abs(rank_0 - rank_1) / total. This avoids materializing
                # both candidate rank-load tensors for every slot pair.
                candidate_difference = (
                    rank_loads[:, hottest_rank, None, None] - rank_loads[:, target_rank, None, None] + 2.0 * delta
                )
                candidate_scores = np.ones(candidate_difference.shape[1:], dtype=np.float64)
                if np.any(nonzero):
                    candidate_scores += (
                        np.sum(
                            np.abs(candidate_difference[nonzero]) / total_load[nonzero, None, None],
                            axis=0,
                        )
                        / expert_load.shape[0]
                    )
            else:
                source_rank_loads = rank_loads[:, hottest_rank, None, None] + delta
                target_rank_loads = rank_loads[:, target_rank, None, None] - delta
                candidate_peak = np.maximum(source_rank_loads, target_rank_loads)
                other_ranks = [
                    rank_id for rank_id in range(proposal.shape[0]) if rank_id not in (hottest_rank, target_rank)
                ]
                other_peak = np.max(rank_loads[:, other_ranks], axis=1)
                candidate_peak = np.maximum(candidate_peak, other_peak[:, None, None])
                candidate_ratios = np.ones_like(candidate_peak)
                candidate_ratios[nonzero] = candidate_peak[nonzero] / average_load[nonzero, None, None]
                candidate_scores = np.mean(candidate_ratios, axis=0)
            aggregate_delta = np.sum(delta, axis=0)
            source_aggregate_loads = aggregate_rank_loads[hottest_rank] + aggregate_delta
            target_aggregate_loads = aggregate_rank_loads[target_rank] - aggregate_delta
            other_ranks = [
                rank_id for rank_id in range(proposal.shape[0]) if rank_id not in (hottest_rank, target_rank)
            ]
            if other_ranks:
                other_aggregate_peak = float(np.max(aggregate_rank_loads[other_ranks]))
                candidate_aggregate_peaks = np.maximum(
                    np.maximum(source_aggregate_loads, target_aggregate_loads),
                    other_aggregate_peak,
                )
            else:
                candidate_aggregate_peaks = np.maximum(source_aggregate_loads, target_aggregate_loads)
            meets_swap_threshold = aggregate_peak - candidate_aggregate_peaks >= swap_threshold
            candidate_scores[~valid_swaps] = np.inf
            candidate_scores[~meets_swap_threshold] = np.inf
            score = float(np.min(candidate_scores))
            if score >= current_score - BALANCE_EPSILON:
                continue

            source_was_changed = source_experts != original_placement[hottest_rank]
            target_was_changed = target_experts != original_placement[target_rank]
            changed_slots = (
                current_changed_slots
                - source_was_changed[:, None].astype(np.int64)
                - target_was_changed[None, :].astype(np.int64)
                + (target_experts[None, :] != original_placement[hottest_rank, :, None]).astype(np.int64)
                + (source_experts[:, None] != original_placement[target_rank, None, :]).astype(np.int64)
            )
            score_ties = candidate_scores <= score + BALANCE_EPSILON
            tied_changed_slots = np.where(score_ties, changed_slots, np.iinfo(np.int64).max)
            candidate_changed_slots = int(np.min(tied_changed_slots))
            slot_ties = score_ties & (changed_slots == candidate_changed_slots)
            source_slot, target_slot = np.argwhere(slot_ties)[0]
            swap = (hottest_rank, int(source_slot), target_rank, int(target_slot))
            if score < best_score - BALANCE_EPSILON or (
                abs(score - best_score) <= BALANCE_EPSILON
                and (
                    candidate_changed_slots < best_changed_slots
                    or (candidate_changed_slots == best_changed_slots and (best_swap is None or swap < best_swap))
                )
            ):
                best_swap = swap
                best_score = score
                best_changed_slots = candidate_changed_slots

        if best_swap is None:
            break
        source_rank, source_slot, target_rank, target_slot = best_swap
        source_expert = int(proposal[source_rank, source_slot])
        target_expert = int(proposal[target_rank, target_slot])
        delta = unit_load[:, target_expert] - unit_load[:, source_expert]
        rank_loads[:, source_rank] += delta
        rank_loads[:, target_rank] -= delta
        proposal[source_rank, source_slot] = target_expert
        proposal[target_rank, target_slot] = source_expert
        num_com_between_rank[source_rank, target_rank] += 1
        num_com_between_rank[target_rank, source_rank] += 1
        current_score = best_score
    return proposal


def _align_local_slots(original_placement: np.ndarray, placement: np.ndarray) -> np.ndarray:
    aligned = np.full_like(placement, -1)
    for rank_id, (original_rank, new_rank) in enumerate(zip(original_placement, placement)):
        remaining_experts = list(new_rank)
        for slot_id, expert_id in enumerate(original_rank):
            if expert_id in remaining_experts:
                aligned[rank_id, slot_id] = expert_id
                remaining_experts.remove(expert_id)
        remaining_slots = np.flatnonzero(aligned[rank_id] < 0)
        for slot_id, expert_id in zip(remaining_slots, remaining_experts):
            aligned[rank_id, slot_id] = expert_id
    return aligned


class StairEplbPolicy:
    """Pure CPU implementation of STAIR expert placement."""

    def __init__(self) -> None:
        self.average_to_peak_history: dict[int, float] = {}
        self._topology: tuple[int, int, int, int] | None = None
        self._expected_placement: np.ndarray | None = None

    def _prepare_history(
        self,
        expert_load: np.ndarray,
        current_placement: np.ndarray,
        num_ranks: int,
    ) -> None:
        topology = (
            expert_load.shape[1],
            expert_load.shape[2],
            current_placement.shape[1],
            num_ranks,
        )
        if self._topology != topology:
            self.average_to_peak_history.clear()
            self._topology = topology
            self._expected_placement = None
        elif self._expected_placement is not None and not np.array_equal(
            current_placement,
            self._expected_placement,
        ):
            self.average_to_peak_history.clear()
            self._expected_placement = None

    def _needs_flash_update(
        self,
        layer_id: int,
        current_score: float,
        num_ranks: int,
    ) -> bool:
        past_ratio = self.average_to_peak_history.get(layer_id)
        if past_ratio is None:
            return True
        if num_ranks < FLASH_SMALL_WORLD_SIZE:
            threshold_ratio = FLASH_SMALL_WORLD_UPDATE_THRESHOLD_RATIO
            threshold_value = FLASH_SMALL_WORLD_UPDATE_THRESHOLD_VALUE
        else:
            threshold_ratio = FLASH_UPDATE_THRESHOLD_RATIO
            threshold_value = FLASH_UPDATE_THRESHOLD_VALUE
        current_ratio = 1.0 / current_score
        return current_ratio < past_ratio * threshold_ratio or current_ratio < threshold_value

    def rebalance_experts(
        self,
        expert_load: np.ndarray,
        current_placement: np.ndarray,
        num_ranks: int,
    ) -> np.ndarray:
        expert_load = np.asarray(expert_load, dtype=np.float64)
        current_placement = np.asarray(current_placement, dtype=np.int64)
        if expert_load.ndim != 3:
            raise ValueError(f"expert_load must have shape [window, layers, experts], got {expert_load.shape}.")
        if current_placement.ndim != 2:
            raise ValueError(f"current_placement must have shape [layers, replicas], got {current_placement.shape}.")
        if expert_load.shape[0] == 0 or expert_load.shape[2] == 0:
            raise ValueError("expert_load window and expert dimensions must be nonzero.")
        if current_placement.shape[0] != expert_load.shape[1]:
            raise ValueError("expert_load and current_placement must have the same number of layers.")
        if num_ranks <= 0 or current_placement.shape[1] % num_ranks != 0:
            raise ValueError("The number of physical replicas must be divisible by num_ranks.")
        if not np.all(np.isfinite(expert_load)) or np.any(expert_load < 0):
            raise ValueError("expert_load must contain finite, non-negative values.")

        num_experts = expert_load.shape[2]
        slots_per_rank = current_placement.shape[1] // num_ranks
        current_by_rank = current_placement.reshape(current_placement.shape[0], num_ranks, slots_per_rank)
        for layer_placement in current_by_rank:
            _validate_layer_placement(layer_placement, num_experts, num_ranks)
        self._prepare_history(expert_load, current_placement, num_ranks)

        result = current_by_rank.copy()
        if not np.any(expert_load):
            return result.reshape(current_placement.shape).copy()

        current_global_peak = 0.0
        candidate_global_peak = 0.0
        for layer_id in range(expert_load.shape[1]):
            layer_load = expert_load[:, layer_id, :]
            original_placement = current_by_rank[layer_id]
            current_score = compute_balance_score(layer_load, original_placement)
            current_peak, current_average = _aggregate_peak_and_average(layer_load, original_placement)
            current_global_peak += current_peak
            candidate_global_peak += current_peak
            aggregate_imbalance = current_peak / current_average if current_average > 0 else 1.0
            if aggregate_imbalance < SWIFT_IMBALANCE_THRESHOLD or not self._needs_flash_update(
                layer_id,
                current_score,
                num_ranks,
            ):
                continue

            mean, variance, covariance = _compute_statistics(layer_load)
            target_replicas = allocate_replicas(
                mean,
                variance,
                current_placement.shape[1],
                num_ranks,
            )
            replacement = _replace_surplus_replicas(
                layer_load,
                original_placement,
                target_replicas,
                mean,
                variance,
                covariance,
            )
            if replacement is None:
                continue
            proposal, num_com_between_rank = replacement
            proposal = _refine_placement(
                layer_load,
                proposal,
                original_placement,
                target_replicas,
                num_com_between_rank,
            )
            proposal = _align_local_slots(original_placement, proposal)
            _validate_layer_placement(
                proposal,
                num_experts,
                num_ranks,
                target_replicas,
            )
            candidate_score = compute_balance_score(layer_load, proposal)
            if current_score - candidate_score > BALANCE_EPSILON:
                result[layer_id] = proposal
                candidate_peak, _ = _aggregate_peak_and_average(layer_load, proposal)
                candidate_global_peak += candidate_peak - current_peak

        minimum_global_improvement = current_global_peak * SWIFT_GLOBAL_IMPROVEMENT_RATIO
        if current_global_peak - candidate_global_peak <= minimum_global_improvement:
            result = current_by_rank.copy()

        return result.reshape(current_placement.shape).copy()

    def commit(
        self,
        expert_load: np.ndarray,
        current_placement: np.ndarray,
        committed_placement: np.ndarray,
        num_ranks: int,
    ) -> None:
        """Record FlashLB hysteresis state only for placements actually committed."""
        expert_load = np.asarray(expert_load, dtype=np.float64)
        current_placement = np.asarray(current_placement, dtype=np.int64)
        committed_placement = np.asarray(committed_placement, dtype=np.int64)
        if current_placement.shape != committed_placement.shape:
            raise ValueError("Current and committed placements must have the same shape.")
        self._prepare_history(expert_load, current_placement, num_ranks)
        slots_per_rank = current_placement.shape[1] // num_ranks
        for layer_id in np.flatnonzero(np.any(committed_placement != current_placement, axis=1)):
            placement = committed_placement[layer_id].reshape(num_ranks, slots_per_rank)
            score = compute_balance_score(expert_load[:, layer_id, :], placement)
            self.average_to_peak_history[int(layer_id)] = 1.0 / score
        self._expected_placement = committed_placement.copy()
