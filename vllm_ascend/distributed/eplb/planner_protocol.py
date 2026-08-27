# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Versioned local protocol for the process-isolated STAIR planner."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
import torch

from vllm_ascend.distributed.eplb.policy.stair_types import (
    StairBudgetUsage,
    StairExecutionMetrics,
    StairLayerPlan,
    StairRebalancePlan,
    StairTopology,
)

PLANNER_PROTOCOL_VERSION = 1
PLANNER_HEALTH_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class PlannerModelRegistration:
    """Parent-side description used to create one shared-memory model slot."""

    model_id: str
    load_shape: tuple[int, int, int]
    mapping_shape: tuple[int, int]
    num_replicas: int
    num_groups: int
    num_nodes: int
    num_ranks: int
    topology: StairTopology
    layer_expert_bytes: tuple[int, ...] | None


@dataclass(frozen=True)
class PlannerModelSpec:
    """Child-visible immutable model and shared-memory description."""

    registration: PlannerModelRegistration
    shared_fd: int
    shared_size: int
    slot_stride: int
    load_offset: int
    mapping_offset: int


@dataclass(frozen=True)
class InitializeRequest:
    protocol_version: int
    models: tuple[PlannerModelSpec, ...]


@dataclass(frozen=True)
class AffinityReadyRequest:
    protocol_version: int
    planner_cpus: tuple[int, ...]


@dataclass(frozen=True)
class PlanRequest:
    protocol_version: int
    request_id: int
    model_id: str
    mapping_version: int
    topology_epoch: int
    tuning_version: int
    slot: int
    slot_sequence: int
    input_digest: str


@dataclass(frozen=True)
class PlannerPlanPayload:
    """Tensor-free wire form of a STAIR plan.

    Standard torch tensor pickling invokes torch.load in the receiver and can
    activate device-specific deserializers. The process protocol carries the
    small CPU mapping as explicit int64 bytes instead.
    """

    mapping_shape: tuple[int, int]
    mapping_data: bytes
    selected_layers: tuple[StairLayerPlan, ...]
    rejected_layers: tuple[StairLayerPlan, ...]
    budget_usage: StairBudgetUsage
    planner_elapsed_ms: float
    plan_id: str
    topology_hash: str

    @classmethod
    def from_plan(cls, plan: StairRebalancePlan) -> PlannerPlanPayload:
        mapping = plan.new_mapping.detach().to(device="cpu", dtype=torch.long).contiguous().numpy()
        if mapping.ndim != 2:
            raise ValueError(f"STAIR planner mapping must be two-dimensional, got shape {mapping.shape}.")
        return cls(
            mapping_shape=(int(mapping.shape[0]), int(mapping.shape[1])),
            mapping_data=mapping.tobytes(order="C"),
            selected_layers=plan.selected_layers,
            rejected_layers=plan.rejected_layers,
            budget_usage=plan.budget_usage,
            planner_elapsed_ms=plan.planner_elapsed_ms,
            plan_id=plan.plan_id,
            topology_hash=plan.topology_hash,
        )

    def to_plan(self) -> StairRebalancePlan:
        expected_bytes = int(np.prod(self.mapping_shape, dtype=np.int64)) * np.dtype(np.int64).itemsize
        if len(self.mapping_data) != expected_bytes:
            raise ValueError(
                f"STAIR planner mapping payload has {len(self.mapping_data)} bytes, expected {expected_bytes}."
            )
        mapping = np.frombuffer(self.mapping_data, dtype=np.int64).copy().reshape(self.mapping_shape)
        return StairRebalancePlan(
            new_mapping=torch.from_numpy(mapping),
            selected_layers=self.selected_layers,
            rejected_layers=self.rejected_layers,
            budget_usage=self.budget_usage,
            planner_elapsed_ms=self.planner_elapsed_ms,
            plan_id=self.plan_id,
            topology_hash=self.topology_hash,
        )


@dataclass(frozen=True)
class PlanResponse:
    protocol_version: int
    request_id: int
    model_id: str
    mapping_version: int
    topology_epoch: int
    tuning_version: int
    slot_sequence: int
    plan: PlannerPlanPayload


@dataclass(frozen=True)
class FinalizeRequest:
    protocol_version: int
    request_id: int
    model_id: str
    mapping_version: int
    topology_epoch: int
    tuning_version: int
    committed_layer_ids: tuple[int, ...]
    execution_metrics: tuple[StairExecutionMetrics, ...]
    aborted: bool = False


@dataclass(frozen=True)
class ShutdownRequest:
    protocol_version: int


@dataclass(frozen=True)
class PlannerAck:
    protocol_version: int
    operation: str
    request_id: int | None = None
    model_id: str | None = None


@dataclass(frozen=True)
class PlannerFatalResponse:
    protocol_version: int
    operation: str
    error_type: str
    error_message: str


PlannerRequest = InitializeRequest | AffinityReadyRequest | PlanRequest | FinalizeRequest | ShutdownRequest
PlannerResponse = PlannerAck | PlanResponse | PlannerFatalResponse


def send_protocol_message(connection: Connection, message: Any) -> None:
    """Send a tensor-free message over the inherited local connection."""
    connection.send(message)


def receive_protocol_message(connection: Connection) -> Any:
    return connection.recv()
