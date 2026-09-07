# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Deterministic source matching and rank-local slot alignment."""

from collections import deque

import numpy as np

from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology

Demand = tuple[int, int]  # (destination rank, logical expert)


def _topology_cost(topology: RankTopology, src: int, dst: int, demand_count: int) -> int:
    return 1 if topology.same_node(src, dst) else demand_count + 1


def _minimum_cost(
    demands: list[Demand],
    owners: dict[int, tuple[int, ...]],
    capacity: dict[tuple[int, int], int],
    topology: RankTopology,
    scale: int,
) -> int | None:
    """Solve the pair-capacitated bipartite flow using stable augmenting paths."""
    if not demands:
        return 0
    pairs = sorted(
        {
            (src, dst)
            for dst, expert in demands
            for src in owners[expert]
            if capacity.get((src, dst), 0) > 0
        }
    )
    pair_node = {pair: len(demands) + 1 + index for index, pair in enumerate(pairs)}
    sink = len(demands) + len(pairs) + 1
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, cap: int, cost: int) -> None:
        graph[start].append([end, len(graph[end]), cap, cost])
        graph[end].append([start, len(graph[start]) - 1, 0, -cost])

    for index, (dst, expert) in enumerate(demands, 1):
        add_edge(0, index, 1, 0)
        for src in owners[expert]:
            if (src, dst) in pair_node:
                add_edge(index, pair_node[(src, dst)], 1, _topology_cost(topology, src, dst, scale))
    for pair, node in pair_node.items():
        add_edge(node, sink, capacity[pair], 0)

    total = 0
    for _ in demands:
        distance = [10**18] * len(graph)
        parent: list[tuple[int, int] | None] = [None] * len(graph)
        distance[0] = 0
        queue, queued = deque([0]), {0}
        while queue:
            node = queue.popleft()
            queued.discard(node)
            for edge_idx, edge in enumerate(graph[node]):
                target, _, cap, cost = edge
                if cap and distance[node] + cost < distance[target]:
                    distance[target] = distance[node] + cost
                    parent[target] = (node, edge_idx)
                    if target not in queued:
                        queue.append(target)
                        queued.add(target)
        if parent[sink] is None:
            return None
        total += distance[sink]
        node = sink
        while node:
            previous, edge_idx = parent[node]  # type: ignore[misc]
            edge = graph[previous][edge_idx]
            edge[2] -= 1
            graph[node][edge[1]][2] += 1
            node = previous
    return total


def assign_sources(
    old_placement: np.ndarray,
    destination_experts: list[set[int]],
    topology: RankTopology,
    pair_cap: int,
) -> dict[Demand, tuple[int, int]] | None:
    """Minimize cross-node then intra-node moves, then source ranks lexically."""
    old = np.asarray(old_placement, dtype=np.int64)
    owners: dict[int, list[tuple[int, int]]] = {}
    demands: list[Demand] = []
    for rank, experts in enumerate(old):
        for slot, expert in enumerate(experts):
            owners.setdefault(int(expert), []).append((rank, slot))
    for dst, experts in enumerate(destination_experts):
        demands.extend((dst, expert) for expert in sorted(experts) if expert not in old[dst])
    demands.sort()
    owner_ranks = {expert: tuple(rank for rank, _ in locations) for expert, locations in owners.items()}
    slots = {(expert, rank): slot for expert, locations in owners.items() for rank, slot in locations}
    capacity = {(src, dst): pair_cap for src in range(old.shape[0]) for dst in range(old.shape[0]) if src != dst}
    target = _minimum_cost(demands, owner_ranks, capacity, topology, len(demands))
    if target is None:
        return None

    assignment: dict[Demand, tuple[int, int]] = {}
    for index, demand in enumerate(demands):
        dst, expert = demand
        for src in owner_ranks[expert]:
            pair = (src, dst)
            if capacity.get(pair, 0) == 0:
                continue
            cost = _topology_cost(topology, src, dst, len(demands))
            capacity[pair] -= 1
            future = _minimum_cost(demands[index + 1 :], owner_ranks, capacity, topology, len(demands))
            if future is not None and cost + future == target:
                assignment[demand] = (src, slots[(expert, src)])
                target -= cost
                break
            capacity[pair] += 1
        else:
            return None
    return assignment


def align_slots(
    old_placement: np.ndarray,
    rank_experts: list[set[int]],
    sources: dict[Demand, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep local experts in their old slots, then fill holes by expert id."""
    old = np.asarray(old_placement, dtype=np.int64)
    new = np.full_like(old, -1)
    source_rank = np.full_like(old, -1)
    source_slot = np.full_like(old, -1)
    for rank, desired in enumerate(rank_experts):
        for slot, expert in enumerate(old[rank]):
            if int(expert) in desired:
                new[rank, slot] = expert
                source_rank[rank, slot], source_slot[rank, slot] = rank, slot
        empty = iter(np.flatnonzero(new[rank] < 0))
        for expert in sorted(desired - set(old[rank])):
            slot = int(next(empty))
            new[rank, slot] = expert
            source_rank[rank, slot], source_slot[rank, slot] = sources[(rank, expert)]
    return new, source_rank, source_slot
