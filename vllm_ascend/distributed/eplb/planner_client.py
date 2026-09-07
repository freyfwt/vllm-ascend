# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Rank-zero lifecycle for the persistent STAIR planner process."""

from dataclasses import dataclass

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.planner_protocol import (
    PlannerRegistration,
    WireOp,
    encode_registration,
    receive_frame,
    send_frame,
)
from vllm_ascend.distributed.eplb.planner_shared_memory import SharedSnapshotBuffer, SharedSnapshotShape
from vllm_ascend.distributed.eplb.planner_subprocess import apply_affinity, spawn_planner
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology


class PlannerProcessError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannerModel:
    model_id: str
    shape: SharedSnapshotShape
    topology: RankTopology
    config: StairConfig


class PlannerClient:
    """Own a pure-CPU child and one double buffer per registered model."""

    def __init__(self, models: tuple[PlannerModel, ...]) -> None:
        if not models or len({model.model_id for model in models}) != len(models):
            raise ValueError("STAIR planner models must be non-empty and unique")
        self._models = {model.model_id: model for model in models}
        self._shared = {model.model_id: SharedSnapshotBuffer.create(model.shape) for model in models}
        self._closed = False
        try:
            self._process, self._connection = spawn_planner(
                tuple(shared.spec.fd for shared in self._shared.values())
            )
        except BaseException:
            self._close_shared()
            raise
        try:
            apply_affinity(self._process, models[0].config)
            for model in models:
                registration = PlannerRegistration(
                    model.model_id,
                    self._shared[model.model_id].spec,
                    model.topology,
                    model.config,
                )
                send_frame(self._connection, WireOp.INITIALIZE, encode_registration(registration))
                operation, payload = self._receive(model.config.planner_heartbeat_timeout_s)
                if operation != WireOp.ACK or payload.decode() != model.model_id:
                    raise PlannerProcessError("STAIR planner initialization acknowledgement mismatch")
        except BaseException:
            self._process.kill()
            self._process.wait()
            self._connection.close()
            self._close_shared()
            self._closed = True
            raise

    @property
    def pid(self) -> int:
        return self._process.pid

    def _receive(self, timeout: float) -> tuple[WireOp, bytes]:
        if not self._connection.poll(timeout):
            raise PlannerProcessError(f"STAIR planner health timeout after {timeout:.1f} seconds")
        operation, payload = receive_frame(self._connection)
        if operation == WireOp.ERROR:
            raise PlannerProcessError(payload.decode(errors="replace"))
        return operation, payload

    def _close_shared(self) -> None:
        for shared in self._shared.values():
            shared.close()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.poll() is None:
                send_frame(self._connection, WireOp.SHUTDOWN)
                operation, payload = self._receive(max(model.config.planner_heartbeat_timeout_s for model in self._models.values()))
                if operation != WireOp.ACK or payload != b"shutdown":
                    raise PlannerProcessError("STAIR planner shutdown acknowledgement mismatch")
                self._process.wait()
        except BaseException:
            self._process.kill()
            self._process.wait()
            raise
        finally:
            self._closed = True
            self._connection.close()
            self._close_shared()
