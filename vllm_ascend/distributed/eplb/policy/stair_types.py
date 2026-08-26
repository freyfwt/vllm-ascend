# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Pure data types shared by the STAIR planner and executor."""

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


@dataclass(frozen=True)
class StairLayerPlan:
    layer_idx: int
    placement: np.ndarray
    current_score: StairBalanceScore
    candidate_score: StairBalanceScore
    balance_gain: float
    utility: float
    transfer_cost: StairTransferCost
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

        cost = layer_plan.transfer_cost
        return (
            self.selected_layers + 1 <= constants.MAX_LAYERS_PER_CYCLE
            and self.expert_transfers + cost.expert_transfers <= constants.MAX_EXPERT_TRANSFERS_PER_CYCLE
            and self.total_bytes + cost.total_bytes <= constants.MAX_TRANSFER_BYTES_PER_CYCLE
            and self.cross_node_bytes + cost.cross_node_bytes <= constants.MAX_CROSS_NODE_BYTES_PER_CYCLE
        )

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


@dataclass(frozen=True)
class StairExecutionMetrics:
    plan_id: str
    layer_idx: int
    recv_count: int
    actual_remote_bytes: int
    transfer_elapsed_ms: float
    source_mode: StairSourceMode
    committed: bool = False
