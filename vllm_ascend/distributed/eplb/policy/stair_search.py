# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Deterministic FlashTree-style replica search for STAIR."""

from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt


ReplicaVector = npt.NDArray[np.int64]


def capped_min_max(
    risk: np.ndarray,
    replicas: np.ndarray,
    slots: int,
    num_ranks: int,
    experts: Iterable[int] | None = None,
) -> ReplicaVector | None:
    """Assign slots to the largest risk-per-replica value without rank duplicates."""
    weights = np.asarray(risk, dtype=np.float64)
    result = np.asarray(replicas, dtype=np.int64).copy()
    allowed = tuple(range(weights.size)) if experts is None else tuple(experts)
    if weights.shape != result.shape or not np.all(np.isfinite(weights)) or slots < 0:
        raise ValueError("Invalid STAIR replica allocation input")
    for _ in range(slots):
        candidates = [expert for expert in allowed if result[expert] < num_ranks]
        if not candidates:
            return None
        expert = min(candidates, key=lambda item: (-weights[item] / result[item], item))
        result[expert] += 1
    return result


def _budgets(center: int, lower: int, upper: int, width: int) -> list[int]:
    values: list[int] = []
    for distance in range(width + 1):
        candidates = (center,) if distance == 0 else (center + distance, center - distance)
        for value in candidates:
            if lower <= value <= upper and value not in values:
                values.append(value)
    return values


def replica_candidates(
    risk: np.ndarray,
    total_slots: int,
    num_ranks: int,
    *,
    depth: int,
    width: int,
    max_candidates: int,
    screening_score: Callable[[ReplicaVector], float],
) -> list[ReplicaVector]:
    """Return bounded FlashTree-style full replica vectors in stable order."""
    weights = np.asarray(risk, dtype=np.float64)
    experts = weights.size
    if experts == 0 or total_slots < experts or total_slots > experts * num_ranks:
        raise ValueError("STAIR requires E <= physical slots <= E * ranks")
    order = sorted(range(experts), key=lambda expert: (-weights[expert], expert))
    group_size = (experts + min(depth, experts) - 1) // min(depth, experts)
    groups = [tuple(order[start : start + group_size]) for start in range(0, experts, group_size)]
    initial = np.ones(experts, dtype=np.int64)
    beam: list[tuple[ReplicaVector, int]] = [(initial, total_slots - experts)]

    for group_idx, group in enumerate(groups[:-1]):
        later = tuple(expert for remaining in groups[group_idx + 1 :] for expert in remaining)
        expanded: list[tuple[ReplicaVector, int, ReplicaVector]] = []
        for replicas, remaining_slots in beam:
            baseline = capped_min_max(weights, replicas, remaining_slots, num_ranks, (*group, *later))
            if baseline is None:
                continue
            center = int(np.sum(baseline[list(group)] - replicas[list(group)]))
            current_capacity = sum(num_ranks - replicas[expert] for expert in group)
            later_capacity = sum(num_ranks - replicas[expert] for expert in later)
            lower = max(0, remaining_slots - later_capacity)
            upper = min(remaining_slots, current_capacity)
            for budget in _budgets(center, lower, upper, width):
                partial = capped_min_max(weights, replicas, budget, num_ranks, group)
                if partial is None:
                    continue
                remainder = remaining_slots - budget
                full = capped_min_max(weights, partial, remainder, num_ranks, later)
                if full is not None:
                    expanded.append((partial, remainder, full))

        unique: dict[bytes, tuple[ReplicaVector, int, ReplicaVector]] = {}
        for partial, remainder, full in expanded:
            unique.setdefault(np.asarray(partial, dtype="<i4").tobytes(), (partial, remainder, full))
        ranked = sorted(
            unique.values(),
            key=lambda item: (screening_score(item[2]), tuple(item[2]), tuple(item[0])),
        )
        beam = [(partial, remainder) for partial, remainder, _ in ranked[:max_candidates]]

    final_group = groups[-1]
    complete: dict[bytes, ReplicaVector] = {}
    for replicas, remaining_slots in beam:
        candidate = capped_min_max(weights, replicas, remaining_slots, num_ranks, final_group)
        if candidate is not None:
            complete.setdefault(np.asarray(candidate, dtype="<i4").tobytes(), candidate)
    return sorted(complete.values(), key=lambda item: (screening_score(item), tuple(item)))[:max_candidates]
