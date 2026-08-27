# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Pure data types shared by the STAIR planner and executor."""

import hashlib
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch


class StairRejectReason(str, Enum):
    NONE = "none"
    BALANCED = "balanced"
    TEMPORAL_GATE = "temporal_gate"
    NO_BALANCE_GAIN = "no_balance_gain"
    TAIL_REGRESSION = "tail_regression"
    LOW_UTILITY = "low_utility"
    LAYER_BUDGET = "layer_budget"
    CYCLE_BUDGET = "cycle_budget"
    INVALID_TOPOLOGY = "invalid_topology"
    SEARCH_TIMEOUT = "search_timeout"
    SEARCH_NO_GAIN = "search_no_gain"


class StairCandidateKind(str, Enum):
    CURRENT = "current"
    GREEDY = "greedy"
    MINI_FLASH_TREE = "mini_flash_tree"


class StairSourceMode(str, Enum):
    PLUGIN_ORDERED = "plugin_ordered"
    EXECUTOR_DEFAULT = "executor_default"


class StairTransferKind(str, Enum):
    UNCHANGED = "unchanged"
    LOCAL_COPY = "local_copy"
    INTRA_NODE = "intra_node"
    CROSS_NODE = "cross_node"
    REMOTE_UNKNOWN = "remote_unknown"


@dataclass(frozen=True)
class StairTopology:
    """Stable EPLB rank-to-node topology used by the CPU planner."""

    rank_to_node: tuple[int, ...]
    equivalent_rank_groups: tuple[tuple[int, ...], ...]
    num_nodes: int
    is_flat_fallback: bool
    topology_hash: str

    def __post_init__(self) -> None:
        if not self.rank_to_node or self.num_nodes <= 0:
            raise ValueError("STAIR topology must contain at least one rank and node.")
        if len(set(self.rank_to_node)) != self.num_nodes:
            raise ValueError("STAIR topology node count does not match rank_to_node.")
        grouped_ranks = tuple(rank for group in self.equivalent_rank_groups for rank in group)
        if sorted(grouped_ranks) != list(range(len(self.rank_to_node))):
            raise ValueError("STAIR equivalent rank groups must cover every rank exactly once.")

    @classmethod
    def from_rank_to_node(
        cls,
        rank_to_node: tuple[int, ...],
        *,
        is_flat_fallback: bool = False,
    ) -> "StairTopology":
        """Build deterministic node groups and a stable topology digest."""
        if not rank_to_node:
            raise ValueError("STAIR rank_to_node cannot be empty.")
        normalized_nodes: dict[int, int] = {}
        normalized_mapping = tuple(
            normalized_nodes.setdefault(node_id, len(normalized_nodes)) for node_id in rank_to_node
        )
        if is_flat_fallback:
            groups: tuple[tuple[int, ...], ...] = tuple((rank,) for rank in range(len(normalized_mapping)))
        else:
            groups = tuple(
                tuple(rank for rank, rank_node in enumerate(normalized_mapping) if rank_node == node_id)
                for node_id in range(len(normalized_nodes))
            )
        payload = f"{normalized_mapping}:{groups}:{int(is_flat_fallback)}"
        topology_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return cls(
            rank_to_node=normalized_mapping,
            equivalent_rank_groups=groups,
            num_nodes=len(normalized_nodes),
            is_flat_fallback=is_flat_fallback,
            topology_hash=topology_hash,
        )

    @classmethod
    def contiguous(cls, num_ranks: int, num_nodes: int) -> "StairTopology":
        """Build the deterministic fallback used by the public policy contract."""
        if num_ranks <= 0 or num_nodes <= 0 or num_ranks % num_nodes != 0:
            raise ValueError("STAIR requires num_ranks to be divisible by num_nodes.")
        ranks_per_node = num_ranks // num_nodes
        return cls.from_rank_to_node(tuple(rank // ranks_per_node for rank in range(num_ranks)))

    @classmethod
    def flat_fallback(cls, num_ranks: int) -> "StairTopology":
        """Treat every remote rank as unknown and conservatively cross-node."""
        if num_ranks <= 0:
            raise ValueError("STAIR flat fallback requires at least one rank.")
        return cls.from_rank_to_node(tuple(range(num_ranks)), is_flat_fallback=True)

    def same_node(self, src_rank: int, dst_rank: int) -> bool:
        if src_rank == dst_rank:
            return True
        if self.is_flat_fallback:
            return False
        return self.rank_to_node[src_rank] == self.rank_to_node[dst_rank]

    def equivalent_group(self, rank: int) -> tuple[int, ...]:
        for group in self.equivalent_rank_groups:
            if rank in group:
                return group
        raise IndexError(f"Rank {rank} is outside the STAIR topology.")


@dataclass(frozen=True)
class StairBalanceScore:
    mean_imbalance: float
    p95_imbalance: float
    max_imbalance: float


@dataclass(frozen=True)
class StairTransferCost:
    expert_transfers: int = 0
    total_bytes: int = 0
    cross_node_bytes: int = 0
    weighted_bytes: float = 0.0
    max_rank_pair_transfers: int = 0
    local_copy_count: int = 0
    local_copy_bytes: int = 0
    intra_node_count: int = 0
    intra_node_bytes: int = 0
    cross_node_count: int = 0
    unknown_remote_count: int = 0
    unknown_remote_bytes: int = 0
    max_sender_transfers: int = 0

    @classmethod
    def conservative_max(
        cls,
        first: "StairTransferCost",
        second: "StairTransferCost",
    ) -> "StairTransferCost":
        """Return the component-wise budget envelope of two executor modes."""
        values = {
            field_name: max(getattr(first, field_name), getattr(second, field_name))
            for field_name in cls.__dataclass_fields__
        }
        return cls(**values)


@dataclass(frozen=True)
class StairTransfer:
    expert_id: int
    src_rank: int
    dst_rank: int
    dst_slot: int
    kind: StairTransferKind
    bytes: int


@dataclass(frozen=True)
class StairTransferPlan:
    assignments: tuple[StairTransfer, ...]
    source_rank_by_position: np.ndarray
    send_order_by_expert: tuple[tuple[int, tuple[int, ...]], ...]
    recv_order_by_expert: tuple[tuple[int, tuple[int, ...]], ...]
    cost: StairTransferCost
    executor_default_cost: StairTransferCost
    source_mode: StairSourceMode

    def __post_init__(self) -> None:
        if self.source_rank_by_position.ndim != 1:
            raise ValueError("STAIR source rank positions must be one-dimensional.")
        self.source_rank_by_position.setflags(write=False)

    def send_order(self, expert_id: int) -> tuple[int, ...] | None:
        return dict(self.send_order_by_expert).get(expert_id)

    def recv_order(self, expert_id: int) -> tuple[int, ...] | None:
        return dict(self.recv_order_by_expert).get(expert_id)

    @property
    def budget_cost(self) -> StairTransferCost:
        return StairTransferCost.conservative_max(self.cost, self.executor_default_cost)


@dataclass(frozen=True)
class StairLayerPlan:
    layer_idx: int
    placement: np.ndarray
    current_score: StairBalanceScore
    candidate_score: StairBalanceScore
    balance_gain: float
    utility: float
    transfer_cost: StairTransferCost
    transfer_plan: StairTransferPlan | None = None
    candidate_kind: StairCandidateKind = StairCandidateKind.GREEDY
    source_mode: StairSourceMode = StairSourceMode.EXECUTOR_DEFAULT
    accepted: bool = True
    reject_reason: StairRejectReason = StairRejectReason.NONE
    search_candidates: int = 0
    search_elapsed_ms: float = 0.0


@dataclass
class StairBudgetUsage:
    selected_layers: int = 0
    expert_transfers: int = 0
    total_bytes: int = 0
    cross_node_bytes: int = 0

    def can_add(self, layer_plan: StairLayerPlan) -> bool:
        from vllm_ascend.distributed.eplb.policy import stair_constants as constants

        return layer_plan.transfer_cost.max_rank_pair_transfers <= constants.MAX_TRANSFERS_PER_RANK_PAIR

    def add(self, layer_plan: StairLayerPlan) -> None:
        cost = layer_plan.transfer_cost
        self.selected_layers += 1
        self.expert_transfers += cost.expert_transfers
        self.total_bytes += cost.total_bytes
        self.cross_node_bytes += cost.cross_node_bytes


@dataclass(frozen=True)
class StairRebalancePlan:
    new_mapping: torch.Tensor
    selected_layers: tuple[StairLayerPlan, ...]
    rejected_layers: tuple[StairLayerPlan, ...]
    budget_usage: StairBudgetUsage
    planner_elapsed_ms: float
    plan_id: str
    topology_hash: str
    timed_out: bool = False


@dataclass(frozen=True)
class StairExecutionMetrics:
    plan_id: str
    layer_idx: int
    recv_count: int
    actual_remote_bytes: int
    transfer_elapsed_ms: float
    source_mode: StairSourceMode
    committed: bool = False
