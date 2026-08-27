# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Worker-side lifecycle and transaction client for the STAIR planner."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import NoReturn

import torch
from vllm.logger import logger

from vllm_ascend.distributed.eplb.planner_protocol import (
    PLANNER_HEALTH_TIMEOUT_S,
    PLANNER_PROTOCOL_VERSION,
    AffinityReadyRequest,
    FinalizeRequest,
    InitializeRequest,
    PlannerAck,
    PlannerFatalResponse,
    PlannerModelRegistration,
    PlannerResponse,
    PlanRequest,
    PlanResponse,
    ShutdownRequest,
    receive_protocol_message,
    send_protocol_message,
)
from vllm_ascend.distributed.eplb.planner_shared_memory import StairSharedInputs
from vllm_ascend.distributed.eplb.policy import stair_constants as constants
from vllm_ascend.distributed.eplb.policy.stair_types import StairExecutionMetrics, StairRebalancePlan


class StairPlannerProcessError(RuntimeError):
    """Fatal child process, protocol, or health failure."""


def _fatal_worker_exit(message: str) -> NoReturn:  # pragma: no cover - terminates the worker process.
    logger.critical("Fatal STAIR planner failure: %s", message)
    os._exit(1)


class StairPlannerClient:
    """One persistent rank-zero planner process with synchronous transactions."""

    def __init__(
        self,
        registrations: tuple[PlannerModelRegistration, ...],
        *,
        fatal_callback: Callable[[str], None] = _fatal_worker_exit,
        health_timeout_s: float = PLANNER_HEALTH_TIMEOUT_S,
    ) -> None:
        if not registrations:
            raise ValueError("STAIR planner requires at least one model registration.")
        self._fatal_callback = fatal_callback
        self._health_timeout_s = health_timeout_s
        self._lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._affinity_ready = False
        self._next_request_id = 0
        self._active_requests: dict[str, PlanRequest] = {}
        self._shared_inputs: dict[str, StairSharedInputs] = {}
        try:
            for registration in registrations:
                if registration.model_id in self._shared_inputs:
                    raise ValueError(f"Duplicate STAIR planner model id {registration.model_id}.")
                self._shared_inputs[registration.model_id] = StairSharedInputs.create(registration)
        except BaseException:
            self._close_shared_inputs()
            raise

        parent_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        child_fd = child_socket.fileno()
        inherited_fds = (child_fd, *(shared.spec.shared_fd for shared in self._shared_inputs.values()))
        child_environment = os.environ.copy()
        for variable in (
            "ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_VISIBLE_DEVICES",
            "NPU_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            child_environment.pop(variable, None)
        child_environment.update(
            {
                "ASCEND_RT_VISIBLE_DEVICES": "",
                "VLLM_PLUGINS": "",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vllm_ascend.distributed.eplb.planner_process",
                    "--control-fd",
                    str(child_fd),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                pass_fds=inherited_fds,
                close_fds=True,
                start_new_session=True,
                env=child_environment,
            )
        except BaseException:
            parent_socket.close()
            child_socket.close()
            self._close_shared_inputs()
            raise
        child_socket.close()
        self._connection = Connection(parent_socket.detach())
        try:
            response = self._exchange(
                InitializeRequest(
                    protocol_version=PLANNER_PROTOCOL_VERSION,
                    models=tuple(shared.spec for shared in self._shared_inputs.values()),
                )
            )
            self._expect_ack(response, "initialize")
        except BaseException:
            self._closing = True
            self._process.kill()
            self._process.wait()
            self._connection.close()
            self._close_shared_inputs()
            self._closed = True
            raise
        self._monitor = threading.Thread(
            target=self._monitor_process,
            name="stair-planner-monitor",
            daemon=True,
        )
        self._monitor.start()

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def affinity_ready(self) -> bool:
        return self._affinity_ready

    def _monitor_process(self) -> None:
        return_code = self._process.wait()
        if not self._closing:
            self._fatal_callback(f"planner process {self.pid} exited unexpectedly with status {return_code}")

    def _raise_unavailable(self, detail: str) -> NoReturn:
        return_code = self._process.poll()
        suffix = "still running" if return_code is None else f"exited with status {return_code}"
        raise StairPlannerProcessError(f"{detail}; planner process is {suffix}.")

    def _exchange(self, request: object) -> PlannerResponse:
        with self._lock:
            if self._closed:
                raise StairPlannerProcessError("STAIR planner client is closed.")
            if self._process.poll() is not None:
                self._raise_unavailable("STAIR planner request cannot be sent")
            try:
                send_protocol_message(self._connection, request)
            except (BrokenPipeError, EOFError, OSError) as error:
                self._raise_unavailable(f"STAIR planner control send failed: {error}")
            if not self._connection.poll(self._health_timeout_s):
                self._process.kill()
                self._raise_unavailable(f"STAIR planner health timeout after {self._health_timeout_s:.1f} seconds")
            try:
                response = receive_protocol_message(self._connection)
            except (EOFError, OSError) as error:
                self._raise_unavailable(f"STAIR planner control receive failed: {error}")
        if isinstance(response, PlannerFatalResponse):
            raise StairPlannerProcessError(
                f"STAIR planner {response.operation} failed with {response.error_type}: {response.error_message}"
            )
        if not isinstance(response, (PlannerAck, PlanResponse)):
            raise StairPlannerProcessError(f"STAIR planner returned invalid response {type(response).__name__}.")
        if response.protocol_version != PLANNER_PROTOCOL_VERSION:
            raise StairPlannerProcessError(
                f"STAIR planner response protocol mismatch: expected {PLANNER_PROTOCOL_VERSION}, "
                f"got {response.protocol_version}."
            )
        return response

    @staticmethod
    def _expect_ack(response: PlannerResponse, operation: str) -> PlannerAck:
        if not isinstance(response, PlannerAck) or response.operation != operation:
            raise StairPlannerProcessError(f"STAIR planner expected {operation!r} acknowledgement, got {response!r}.")
        return response

    def mark_affinity_ready(self, planner_cpus: tuple[int, ...]) -> None:
        response = self._exchange(
            AffinityReadyRequest(
                protocol_version=PLANNER_PROTOCOL_VERSION,
                planner_cpus=planner_cpus,
            )
        )
        self._expect_ack(response, "affinity_ready")
        self._affinity_ready = True
        logger.info("STAIR planner process %d is isolated on CPUs %s.", self.pid, planner_cpus)

    def plan(
        self,
        model_id: str,
        load: torch.Tensor,
        mapping: torch.Tensor,
        *,
        mapping_version: int,
        topology_epoch: int,
    ) -> StairRebalancePlan:
        if not self._affinity_ready:
            raise StairPlannerProcessError("STAIR planner CPU affinity is not ready.")
        if model_id in self._active_requests:
            raise StairPlannerProcessError(f"Model {model_id} already has an in-flight STAIR plan.")
        shared_inputs = self._shared_inputs.get(model_id)
        if shared_inputs is None:
            raise StairPlannerProcessError(f"Unknown STAIR planner model id {model_id}.")
        slot, slot_sequence, input_digest = shared_inputs.write(load, mapping)
        self._next_request_id += 1
        request = PlanRequest(
            protocol_version=PLANNER_PROTOCOL_VERSION,
            request_id=self._next_request_id,
            model_id=model_id,
            mapping_version=mapping_version,
            topology_epoch=topology_epoch,
            tuning_version=constants.STAIR_TUNING_VERSION,
            slot=slot,
            slot_sequence=slot_sequence,
            input_digest=input_digest,
        )
        response = self._exchange(request)
        if not isinstance(response, PlanResponse):
            raise StairPlannerProcessError(f"STAIR planner expected a plan response, got {response!r}.")
        expected_metadata = (
            request.request_id,
            request.model_id,
            request.mapping_version,
            request.topology_epoch,
            request.tuning_version,
            request.slot_sequence,
        )
        response_metadata = (
            response.request_id,
            response.model_id,
            response.mapping_version,
            response.topology_epoch,
            response.tuning_version,
            response.slot_sequence,
        )
        if response_metadata != expected_metadata:
            raise StairPlannerProcessError(
                f"STAIR planner response metadata mismatch: expected {expected_metadata}, got {response_metadata}."
            )
        try:
            plan = response.plan.to_plan()
        except (TypeError, ValueError) as error:
            raise StairPlannerProcessError(f"STAIR planner returned an invalid plan payload: {error}") from error
        if plan.new_mapping.device.type != "cpu":
            raise StairPlannerProcessError("STAIR planner returned a non-CPU mapping.")
        self._active_requests[model_id] = request
        return plan

    def finalize(
        self,
        model_id: str,
        *,
        mapping_version: int,
        topology_epoch: int,
        committed_layer_ids: tuple[int, ...],
        execution_metrics: tuple[StairExecutionMetrics, ...],
        aborted: bool = False,
    ) -> None:
        request = self._active_requests.get(model_id)
        if request is None:
            raise StairPlannerProcessError(f"Model {model_id} has no in-flight STAIR plan.")
        response = self._exchange(
            FinalizeRequest(
                protocol_version=PLANNER_PROTOCOL_VERSION,
                request_id=request.request_id,
                model_id=model_id,
                mapping_version=mapping_version,
                topology_epoch=topology_epoch,
                tuning_version=constants.STAIR_TUNING_VERSION,
                committed_layer_ids=committed_layer_ids,
                execution_metrics=execution_metrics,
                aborted=aborted,
            )
        )
        ack = self._expect_ack(response, "finalize")
        if ack.request_id != request.request_id or ack.model_id != model_id:
            raise StairPlannerProcessError("STAIR planner returned a stale finalize acknowledgement.")
        del self._active_requests[model_id]

    def _close_shared_inputs(self) -> None:
        for shared_inputs in self._shared_inputs.values():
            shared_inputs.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        try:
            if self._process.poll() is None:
                try:
                    response = self._exchange(ShutdownRequest(PLANNER_PROTOCOL_VERSION))
                    self._expect_ack(response, "shutdown")
                except BaseException as error:
                    logger.error("STAIR planner graceful shutdown failed: %s", error)
                    self._process.kill()
                self._process.wait(timeout=self._health_timeout_s)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        finally:
            self._closed = True
            self._connection.close()
            self._close_shared_inputs()
