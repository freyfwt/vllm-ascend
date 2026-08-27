# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Executor-compatible source planning for STAIR expert transfers."""

from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch
from vllm.distributed.eplb import rebalance_execute as _rebalance_execute

from vllm_ascend.distributed.eplb.policy import stair_constants as constants
from vllm_ascend.distributed.eplb.policy.stair_search import _solve_linear_assignment
from vllm_ascend.distributed.eplb.policy.stair_types import (
    StairLayerPlan,
    StairSourceMode,
    StairTopology,
    StairTransfer,
    StairTransferCost,
    StairTransferKind,
    StairTransferPlan,
)

_ACTIVE_SOURCE_ORDERING: ContextVar[StairTransferPlan | None] = ContextVar(
    "vllm_ascend_stair_source_ordering",
    default=None,
)
_SOURCE_ORDERING_PATCH_ENABLED = False
_INFEASIBLE_COST = 1e15


class _ExpertWeightView(Protocol):
    @property
    def shape(self) -> torch.Size: ...

    def __getitem__(self, index: int) -> torch.Tensor: ...


def set_source_ordering_patch_enabled(enabled: bool) -> None:
    global _SOURCE_ORDERING_PATCH_ENABLED
    _SOURCE_ORDERING_PATCH_ENABLED = enabled


def get_active_source_ordering() -> StairTransferPlan | None:
    return _ACTIVE_SOURCE_ORDERING.get()


@contextmanager
def source_ordering_context(transfer_plan: StairTransferPlan) -> Iterator[None]:
    """Expose immutable ordering only for one executor transfer call."""
    if transfer_plan.source_mode is not StairSourceMode.PLUGIN_ORDERED:
        with nullcontext():
            yield
        return
    token = _ACTIVE_SOURCE_ORDERING.set(transfer_plan)
    try:
        yield
    finally:
        _ACTIVE_SOURCE_ORDERING.reset(token)


def compute_layer_expert_bytes(
    expert_weights: Sequence[Sequence[_ExpertWeightView]],
) -> tuple[int, ...]:
    """Return bytes occupied by one local physical expert in every layer."""
    result: list[int] = []
    for layer_idx, layer_weights in enumerate(expert_weights):
        if not layer_weights:
            raise ValueError(f"EPLB layer {layer_idx} has no transferable expert weights.")
        local_experts = layer_weights[0].shape[0]
        if local_experts <= 0:
            raise ValueError(f"EPLB layer {layer_idx} has no local physical experts.")
        expert_bytes = 0
        for tensor in layer_weights:
            if tensor.shape[0] != local_experts:
                raise ValueError(f"EPLB layer {layer_idx} weight views disagree on local expert count.")
            expert_tensor = tensor[0]
            expert_bytes += expert_tensor.numel() * expert_tensor.element_size()
        result.append(expert_bytes)
    return tuple(result)


def _source_capacities(source_count: int, destination_count: int) -> tuple[int, ...]:
    if source_count <= 0:
        return ()
    base, remainder = divmod(destination_count, source_count)
    return tuple(base + int(source_idx < remainder) for source_idx in range(source_count))


def _decode_executor_sources(
    send_ranks: tuple[int, ...],
    recv_ranks: tuple[int, ...],
) -> dict[int, int]:
    """Reproduce upstream move_to_buffer's balanced fan-out exactly."""
    if not send_ranks or not recv_ranks:
        return {}
    base = len(recv_ranks) // len(send_ranks)
    remainder_start = len(send_ranks) * base
    result: dict[int, int] = {}
    for sender_pos, source_rank in enumerate(send_ranks):
        begin = sender_pos * base
        for dst_rank in recv_ranks[begin : begin + base]:
            result[dst_rank] = source_rank
        remainder_position = remainder_start + sender_pos
        if remainder_position < len(recv_ranks):
            result[recv_ranks[remainder_position]] = source_rank
    return result


def _classify_transfer(src_rank: int, dst_rank: int, topology: StairTopology) -> StairTransferKind:
    if topology.is_flat_fallback:
        return StairTransferKind.REMOTE_UNKNOWN
    if topology.same_node(src_rank, dst_rank):
        return StairTransferKind.INTRA_NODE
    return StairTransferKind.CROSS_NODE


def _make_remote_transfers(
    source_by_expert_destination: dict[int, dict[int, int]],
    candidate: np.ndarray,
    topology: StairTopology,
    expert_bytes: int,
) -> tuple[StairTransfer, ...]:
    transfers: list[StairTransfer] = []
    for expert_id in sorted(source_by_expert_destination):
        for dst_rank, src_rank in sorted(source_by_expert_destination[expert_id].items()):
            destination_slots = np.flatnonzero(candidate[dst_rank] == expert_id)
            if destination_slots.size != 1:
                raise ValueError("STAIR transfer destination must contain exactly one expert copy.")
            transfers.append(
                StairTransfer(
                    expert_id=expert_id,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    dst_slot=int(destination_slots[0]),
                    kind=_classify_transfer(src_rank, dst_rank, topology),
                    bytes=expert_bytes,
                )
            )
    return tuple(transfers)


def _compute_transfer_cost(
    current: np.ndarray,
    candidate: np.ndarray,
    remote_transfers: tuple[StairTransfer, ...],
    expert_bytes: int,
) -> StairTransferCost:
    remote_by_destination = {(item.expert_id, item.dst_rank) for item in remote_transfers}
    local_copy_count = 0
    for dst_rank in range(candidate.shape[0]):
        for dst_slot, expert_id in enumerate(candidate[dst_rank]):
            if current[dst_rank, dst_slot] == expert_id:
                continue
            if (int(expert_id), dst_rank) not in remote_by_destination:
                local_copy_count += 1

    kind_counts = Counter(item.kind for item in remote_transfers)
    pair_counts = Counter((item.src_rank, item.dst_rank) for item in remote_transfers)
    sender_counts = Counter(item.src_rank for item in remote_transfers)
    intra_count = kind_counts[StairTransferKind.INTRA_NODE]
    cross_count = kind_counts[StairTransferKind.CROSS_NODE]
    unknown_count = kind_counts[StairTransferKind.REMOTE_UNKNOWN]
    local_bytes = local_copy_count * expert_bytes
    intra_bytes = intra_count * expert_bytes
    cross_bytes = cross_count * expert_bytes
    unknown_bytes = unknown_count * expert_bytes
    total_bytes = local_bytes + intra_bytes + cross_bytes + unknown_bytes
    weighted_bytes = (
        local_bytes
        + constants.INTRA_NODE_COST_MULTIPLIER * intra_bytes
        + constants.CROSS_NODE_COST_MULTIPLIER * (cross_bytes + unknown_bytes)
    )
    return StairTransferCost(
        expert_transfers=local_copy_count + len(remote_transfers),
        total_bytes=total_bytes,
        cross_node_bytes=cross_bytes + unknown_bytes,
        weighted_bytes=float(weighted_bytes),
        max_rank_pair_transfers=max(pair_counts.values(), default=0),
        local_copy_count=local_copy_count,
        local_copy_bytes=local_bytes,
        intra_node_count=intra_count,
        intra_node_bytes=intra_bytes,
        cross_node_count=cross_count,
        unknown_remote_count=unknown_count,
        unknown_remote_bytes=unknown_bytes,
        max_sender_transfers=max(sender_counts.values(), default=0),
    )


def _default_source_assignments(
    current: np.ndarray,
    candidate: np.ndarray,
) -> tuple[
    dict[int, dict[int, int]],
    dict[int, tuple[int, ...]],
    dict[int, tuple[int, ...]],
]:
    changed_positions = candidate != current
    if not np.any(changed_positions):
        return {}, {}, {}
    expert_ids = np.unique(
        np.concatenate(
            (current[changed_positions], candidate[changed_positions]),
        )
    )
    send_map, recv_map = _rebalance_execute.get_ep_ranks_with_experts_batch(
        expert_ids,
        current.shape[1],
        current.reshape(-1),
        candidate.reshape(-1),
    )
    assignments: dict[int, dict[int, int]] = {}
    send_orders: dict[int, tuple[int, ...]] = {}
    recv_orders: dict[int, tuple[int, ...]] = {}
    for expert_id in sorted(int(value) for value in expert_ids):
        send_ranks = tuple(send_map[expert_id])
        recv_ranks = tuple(recv_map[expert_id])
        send_orders[expert_id] = send_ranks
        recv_orders[expert_id] = recv_ranks
        assignments[expert_id] = _decode_executor_sources(send_ranks, recv_ranks)
    return assignments, send_orders, recv_orders


def _choose_expert_sources(
    send_ranks: tuple[int, ...],
    recv_ranks: tuple[int, ...],
    topology: StairTopology,
    pair_counts: Counter[tuple[int, int]],
) -> tuple[dict[int, int], tuple[int, ...], tuple[int, ...]] | None:
    if not recv_ranks:
        return {}, send_ranks, recv_ranks
    if not send_ranks:
        return None
    base, remainder = divmod(len(recv_ranks), len(send_ranks))
    source_slots: list[tuple[int, bool]] = []
    for source_rank in send_ranks:
        source_slots.extend((source_rank, False) for _ in range(base))
        if remainder:
            source_slots.append((source_rank, True))
    size = len(source_slots)
    cost: npt.NDArray[np.float64] = np.zeros((size, size), dtype=np.float64)
    for row_idx in range(size):
        if row_idx < len(recv_ranks):
            dst_rank = recv_ranks[row_idx]
            for column_idx, (source_rank, _optional) in enumerate(source_slots):
                if pair_counts[(source_rank, dst_rank)] >= constants.MAX_TRANSFERS_PER_RANK_PAIR:
                    cost[row_idx, column_idx] = _INFEASIBLE_COST
                    continue
                multiplier = (
                    constants.INTRA_NODE_COST_MULTIPLIER
                    if topology.same_node(source_rank, dst_rank)
                    else constants.CROSS_NODE_COST_MULTIPLIER
                )
                cost[row_idx, column_idx] = multiplier + source_rank * 1e-6 + dst_rank * 1e-9
        else:
            for column_idx, (_source_rank, optional) in enumerate(source_slots):
                cost[row_idx, column_idx] = 0.0 if optional else _INFEASIBLE_COST
    assignment = _solve_linear_assignment(cost)
    if assignment is None:
        return None
    source_by_destination: dict[int, int] = {}
    assigned_by_source: defaultdict[int, list[int]] = defaultdict(list)
    for row_idx, column_idx in enumerate(assignment[: len(recv_ranks)]):
        if cost[row_idx, column_idx] >= _INFEASIBLE_COST:
            return None
        source_rank = source_slots[column_idx][0]
        dst_rank = recv_ranks[row_idx]
        source_by_destination[dst_rank] = source_rank
        assigned_by_source[source_rank].append(dst_rank)
    extra_sources = sorted(source for source in send_ranks if len(assigned_by_source[source]) == base + 1)
    if len(extra_sources) != remainder:
        return None
    ordered_sources = tuple(extra_sources + sorted(source for source in send_ranks if source not in extra_sources))
    prefix_destinations: list[int] = []
    remainder_destinations: list[int] = []
    for source_rank in ordered_sources:
        destinations = sorted(assigned_by_source[source_rank])
        prefix_destinations.extend(destinations[:base])
        if len(destinations) > base:
            remainder_destinations.append(destinations[base])
    ordered_destinations = tuple(prefix_destinations + remainder_destinations)
    if _decode_executor_sources(ordered_sources, ordered_destinations) != source_by_destination:
        return None
    return source_by_destination, ordered_sources, ordered_destinations


def _choose_low_cost_sources(
    default_send_orders: dict[int, tuple[int, ...]],
    default_recv_orders: dict[int, tuple[int, ...]],
    topology: StairTopology,
) -> (
    tuple[
        dict[int, dict[int, int]],
        dict[int, tuple[int, ...]],
        dict[int, tuple[int, ...]],
    ]
    | None
):
    pair_counts: Counter[tuple[int, int]] = Counter()
    assignments: dict[int, dict[int, int]] = {}
    send_orders: dict[int, tuple[int, ...]] = {}
    recv_orders: dict[int, tuple[int, ...]] = {}
    for expert_id in sorted(default_send_orders):
        chosen = _choose_expert_sources(
            default_send_orders[expert_id],
            default_recv_orders[expert_id],
            topology,
            pair_counts,
        )
        if chosen is None:
            return None
        source_by_destination, send_order, recv_order = chosen
        assignments[expert_id] = source_by_destination
        send_orders[expert_id] = send_order
        recv_orders[expert_id] = recv_order
        pair_counts.update((source_rank, dst_rank) for dst_rank, source_rank in source_by_destination.items())
    return assignments, send_orders, recv_orders


def _make_transfer_plan(
    current: np.ndarray,
    candidate: np.ndarray,
    topology: StairTopology,
    expert_bytes: int,
    assignments: dict[int, dict[int, int]],
    send_orders: dict[int, tuple[int, ...]],
    recv_orders: dict[int, tuple[int, ...]],
    source_mode: StairSourceMode,
    executor_default_cost: StairTransferCost | None = None,
) -> StairTransferPlan:
    remote_transfers = _make_remote_transfers(assignments, candidate, topology, expert_bytes)
    cost = _compute_transfer_cost(current, candidate, remote_transfers, expert_bytes)
    source_positions: npt.NDArray[np.int32] = np.full(candidate.size, -1, dtype=np.int32)
    for transfer in remote_transfers:
        source_positions[transfer.dst_rank * candidate.shape[1] + transfer.dst_slot] = transfer.src_rank
    return StairTransferPlan(
        assignments=remote_transfers,
        source_rank_by_position=source_positions,
        send_order_by_expert=tuple(sorted(send_orders.items())),
        recv_order_by_expert=tuple(sorted(recv_orders.items())),
        cost=cost,
        executor_default_cost=executor_default_cost or cost,
        source_mode=source_mode,
    )


def build_transfer_plan(
    current: np.ndarray,
    candidate: np.ndarray,
    topology: StairTopology,
    expert_bytes: int,
) -> StairTransferPlan:
    """Build the cheapest ordering that the current upstream executor consumes."""
    default_assignments, default_send_orders, default_recv_orders = _default_source_assignments(current, candidate)
    default_plan = _make_transfer_plan(
        current,
        candidate,
        topology,
        expert_bytes,
        default_assignments,
        default_send_orders,
        default_recv_orders,
        StairSourceMode.EXECUTOR_DEFAULT,
    )
    if not _SOURCE_ORDERING_PATCH_ENABLED or topology.num_nodes <= 1:
        return default_plan
    if all(len(send_ranks) <= 1 for send_ranks in default_send_orders.values()):
        return default_plan
    chosen = _choose_low_cost_sources(default_send_orders, default_recv_orders, topology)
    if chosen is None:
        return default_plan
    assignments, send_orders, recv_orders = chosen
    plugin_plan = _make_transfer_plan(
        current,
        candidate,
        topology,
        expert_bytes,
        assignments,
        send_orders,
        recv_orders,
        StairSourceMode.PLUGIN_ORDERED,
        default_plan.cost,
    )
    if plugin_plan.cost.weighted_bytes >= default_plan.cost.weighted_bytes:
        return default_plan
    return plugin_plan


def _preflight_source_ordering(
    old_layer_indices: torch.Tensor,
    new_layer_indices: torch.Tensor,
    transfer_plan: StairTransferPlan,
    num_ranks: int,
) -> bool:
    expert_ids = np.asarray([item.expert_id for item in transfer_plan.assignments], dtype=np.int64)
    if expert_ids.size == 0:
        return True
    old_indices = old_layer_indices.detach().cpu().numpy()
    new_indices = new_layer_indices.detach().cpu().numpy()
    if num_ranks <= 0 or old_indices.size % num_ranks != 0:
        return False
    num_local_experts = old_indices.size // num_ranks
    send_map, recv_map = _rebalance_execute.get_ep_ranks_with_experts_batch(
        expert_ids,
        num_local_experts,
        old_indices,
        new_indices,
    )
    for expert_id in np.unique(expert_ids):
        planned_send = transfer_plan.send_order(int(expert_id))
        planned_recv = transfer_plan.recv_order(int(expert_id))
        if planned_send is None or planned_recv is None:
            return False
        if set(planned_send) != set(send_map[int(expert_id)]) or set(planned_recv) != set(recv_map[int(expert_id)]):
            return False
    return True


def transfer_layer_with_plan(
    *,
    layer_plan: StairLayerPlan,
    **transfer_kwargs: Any,
) -> tuple[Any, StairSourceMode]:
    """Run upstream transfer_layer with an all-or-nothing ordering context."""
    transfer_plan = layer_plan.transfer_plan
    if (
        transfer_plan is None
        or transfer_plan.source_mode is StairSourceMode.EXECUTOR_DEFAULT
        or not _SOURCE_ORDERING_PATCH_ENABLED
        or not _preflight_source_ordering(
            transfer_kwargs["old_layer_indices"],
            transfer_kwargs["new_layer_indices"],
            transfer_plan,
            transfer_kwargs["ep_group"].size(),
        )
    ):
        return _rebalance_execute.transfer_layer(**transfer_kwargs), StairSourceMode.EXECUTOR_DEFAULT
    with source_ordering_context(transfer_plan):
        return _rebalance_execute.transfer_layer(**transfer_kwargs), StairSourceMode.PLUGIN_ORDERED
