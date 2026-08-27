# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Child entry point for the process-isolated STAIR planner."""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection

import torch

from vllm_ascend.distributed.eplb.planner_protocol import (
    PLANNER_PROTOCOL_VERSION,
    AffinityReadyRequest,
    FinalizeRequest,
    InitializeRequest,
    PlannerAck,
    PlannerFatalResponse,
    PlannerPlanPayload,
    PlannerRequest,
    PlanRequest,
    PlanResponse,
    ShutdownRequest,
    receive_protocol_message,
    send_protocol_message,
)
from vllm_ascend.distributed.eplb.planner_shared_memory import StairSharedInputs
from vllm_ascend.distributed.eplb.policy import stair_constants as constants
from vllm_ascend.distributed.eplb.policy.stair import StairEplbPolicy
from vllm_ascend.distributed.eplb.policy.stair_types import StairRebalancePlan

_PR_SET_PDEATHSIG = 1


def _arm_parent_death_signal() -> None:
    """Make the planner die when its worker parent disappears."""
    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent_pid:
        raise RuntimeError("STAIR planner parent exited during startup.")


def _validate_protocol(message: PlannerRequest) -> None:
    if message.protocol_version != PLANNER_PROTOCOL_VERSION:
        raise RuntimeError(
            f"STAIR planner protocol mismatch: parent={message.protocol_version}, child={PLANNER_PROTOCOL_VERSION}."
        )


@dataclass
class _PendingPlan:
    request: PlanRequest
    load: torch.Tensor
    plan: StairRebalancePlan


@dataclass
class _ModelRuntime:
    shared_inputs: StairSharedInputs
    policy: StairEplbPolicy
    last_slot_sequence: int = 0
    pending: _PendingPlan | None = None


class StairPlannerServer:
    """Single-threaded owner of all rank-zero STAIR policy state."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._models: dict[str, _ModelRuntime] = {}
        self._affinity_ready = False

    def _initialize(self, request: InitializeRequest) -> PlannerAck:
        if self._models:
            raise RuntimeError("STAIR planner was initialized more than once.")
        if not request.models:
            raise RuntimeError("STAIR planner requires at least one registered model.")
        for spec in request.models:
            registration = spec.registration
            if registration.model_id in self._models:
                raise RuntimeError(f"Duplicate STAIR planner model id {registration.model_id}.")
            if registration.topology.topology_hash == "":
                raise RuntimeError("STAIR planner topology hash cannot be empty.")
            self._models[registration.model_id] = _ModelRuntime(
                shared_inputs=StairSharedInputs.attach(spec),
                policy=StairEplbPolicy(
                    registration.topology,
                    registration.layer_expert_bytes,
                ),
            )
        return PlannerAck(PLANNER_PROTOCOL_VERSION, "initialize")

    def _mark_affinity_ready(self, request: AffinityReadyRequest) -> PlannerAck:
        if not self._models:
            raise RuntimeError("STAIR planner affinity cannot be enabled before initialization.")
        if not request.planner_cpus:
            raise RuntimeError("STAIR planner requires a non-empty physical-core affinity.")
        get_affinity = getattr(os, "sched_getaffinity", None)
        if get_affinity is None:
            raise RuntimeError("STAIR planner affinity validation requires Linux sched_getaffinity.")
        actual_cpus = tuple(sorted(get_affinity(0)))
        expected_cpus = tuple(sorted(request.planner_cpus))
        if actual_cpus != expected_cpus:
            raise RuntimeError(f"STAIR planner affinity mismatch: expected {expected_cpus}, got {actual_cpus}.")
        self._affinity_ready = True
        return PlannerAck(PLANNER_PROTOCOL_VERSION, "affinity_ready")

    def _plan(self, request: PlanRequest) -> PlanResponse:
        if not self._affinity_ready:
            raise RuntimeError("STAIR planner received work before CPU affinity was established.")
        if request.tuning_version != constants.STAIR_TUNING_VERSION:
            raise RuntimeError(
                f"STAIR tuning version mismatch: request={request.tuning_version}, "
                f"runtime={constants.STAIR_TUNING_VERSION}."
            )
        runtime = self._models.get(request.model_id)
        if runtime is None:
            raise RuntimeError(f"Unknown STAIR planner model id {request.model_id}.")
        registration = runtime.shared_inputs.spec.registration
        if request.topology_epoch < 0:
            raise RuntimeError("STAIR topology epoch must be non-negative.")
        if runtime.pending is not None:
            raise RuntimeError(f"Model {request.model_id} already has an in-flight STAIR plan.")
        if request.slot_sequence <= runtime.last_slot_sequence:
            raise RuntimeError(
                f"STAIR shared-memory sequence did not advance for model {request.model_id}: "
                f"last={runtime.last_slot_sequence}, request={request.slot_sequence}."
            )
        load, mapping = runtime.shared_inputs.read(request.slot, request.input_digest)
        plan = runtime.policy.plan_rebalance(
            load,
            registration.num_replicas,
            registration.num_groups,
            registration.num_nodes,
            registration.num_ranks,
            mapping,
        )
        runtime.last_slot_sequence = request.slot_sequence
        runtime.pending = _PendingPlan(request=request, load=load, plan=plan)
        return PlanResponse(
            protocol_version=PLANNER_PROTOCOL_VERSION,
            request_id=request.request_id,
            model_id=request.model_id,
            mapping_version=request.mapping_version,
            topology_epoch=request.topology_epoch,
            tuning_version=request.tuning_version,
            slot_sequence=request.slot_sequence,
            plan=PlannerPlanPayload.from_plan(plan),
        )

    def _finalize(self, request: FinalizeRequest) -> PlannerAck:
        runtime = self._models.get(request.model_id)
        if runtime is None:
            raise RuntimeError(f"Unknown STAIR planner model id {request.model_id}.")
        pending = runtime.pending
        if pending is None:
            raise RuntimeError(f"Model {request.model_id} has no in-flight STAIR plan.")
        plan_request = pending.request
        if (
            request.request_id != plan_request.request_id
            or request.topology_epoch != plan_request.topology_epoch
            or request.tuning_version != plan_request.tuning_version
        ):
            raise RuntimeError(f"STAIR finalize metadata is stale for model {request.model_id}.")
        expected_mapping_version = plan_request.mapping_version + len(request.committed_layer_ids)
        if request.mapping_version != expected_mapping_version:
            raise RuntimeError(
                f"STAIR mapping version mismatch for model {request.model_id}: "
                f"expected {expected_mapping_version}, got {request.mapping_version}."
            )
        selected = {layer.layer_idx for layer in pending.plan.selected_layers}
        if not set(request.committed_layer_ids).issubset(selected):
            raise RuntimeError("STAIR finalized a layer that was not selected by the active plan.")
        metrics_by_layer = {metric.layer_idx: metric for metric in request.execution_metrics}
        if set(metrics_by_layer) != set(request.committed_layer_ids):
            raise RuntimeError("STAIR finalized execution metrics do not match committed layers.")
        registration = runtime.shared_inputs.spec.registration
        for layer_idx in request.committed_layer_ids:
            runtime.policy.commit_layer(
                pending.load,
                layer_idx,
                pending.plan.new_mapping[layer_idx],
                registration.num_ranks,
                metrics_by_layer[layer_idx],
            )
        if request.aborted:
            runtime.policy.abort_cycle(pending.plan.plan_id)
        else:
            runtime.policy.finish_cycle(pending.plan.plan_id, request.committed_layer_ids)
        runtime.pending = None
        return PlannerAck(
            PLANNER_PROTOCOL_VERSION,
            "finalize",
            request_id=request.request_id,
            model_id=request.model_id,
        )

    def _handle(self, request: PlannerRequest) -> PlannerAck | PlanResponse:
        _validate_protocol(request)
        if isinstance(request, InitializeRequest):
            return self._initialize(request)
        if isinstance(request, AffinityReadyRequest):
            return self._mark_affinity_ready(request)
        if isinstance(request, PlanRequest):
            return self._plan(request)
        if isinstance(request, FinalizeRequest):
            return self._finalize(request)
        if isinstance(request, ShutdownRequest):
            return PlannerAck(PLANNER_PROTOCOL_VERSION, "shutdown")
        raise TypeError(f"Unsupported STAIR planner request {type(request).__name__}.")

    def run(self) -> int:
        while True:
            request = receive_protocol_message(self._connection)
            if not isinstance(
                request,
                (InitializeRequest, AffinityReadyRequest, PlanRequest, FinalizeRequest, ShutdownRequest),
            ):
                raise TypeError(f"Invalid STAIR planner protocol object {type(request).__name__}.")
            response = self._handle(request)
            send_protocol_message(self._connection, response)
            if isinstance(request, ShutdownRequest):
                return 0

    def close(self) -> None:
        for runtime in self._models.values():
            # Drop tensors backed by the mmap before closing it. This matters
            # when service shutdown races with an in-flight transaction.
            runtime.pending = None
            runtime.shared_inputs.close()
        self._connection.close()


def run_child(control_fd: int) -> int:
    _arm_parent_death_signal()
    torch.set_num_threads(1)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    connection = Connection(control_fd)
    server = StairPlannerServer(connection)
    try:
        return server.run()
    except BaseException as error:
        with suppress(BaseException):
            send_protocol_message(
                connection,
                PlannerFatalResponse(
                    protocol_version=PLANNER_PROTOCOL_VERSION,
                    operation="planner",
                    error_type=type(error).__name__,
                    error_message=str(error),
                ),
            )
        return 1
    finally:
        server.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-fd", type=int, required=True)
    arguments = parser.parse_args()
    return run_child(arguments.control_fd)


if __name__ == "__main__":
    raise SystemExit(main())
