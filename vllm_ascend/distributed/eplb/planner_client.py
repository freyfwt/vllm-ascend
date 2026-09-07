# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Rank-zero lifecycle for the persistent STAIR planner process."""

from dataclasses import dataclass
from time import monotonic

import numpy as np

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
from vllm_ascend.distributed.eplb.planner_wire import PlanRequest, decode_plan, encode_request
from vllm_ascend.distributed.eplb.policy.stair_candidate import config_digest
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology, RebalancePlan


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
        self._active: PlanRequest | None = None
        self._active_since = 0.0
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

    def submit(
        self,
        model_id: str,
        bin_sums: np.ndarray,
        bin_lengths: np.ndarray,
        placements: np.ndarray,
        placement_epochs: np.ndarray,
        accepted_scores: np.ndarray,
        *,
        planning_round: int,
        snapshot_sequence: int,
        sample_sequence: int,
        stats_schema_epoch: int,
        key_digest: str,
    ) -> bool:
        """Publish one snapshot, or leave it with the caller while busy."""
        if self._closed:
            raise PlannerProcessError("STAIR planner client is closed")
        if self._active is not None:
            return False
        model = self._models.get(model_id)
        if model is None:
            raise PlannerProcessError(f"Unknown STAIR planner model {model_id}")
        slot, sequence, digest = self._shared[model_id].write(
            bin_sums, bin_lengths, placements, placement_epochs, accepted_scores
        )
        request = PlanRequest(
            model_id,
            slot,
            len(bin_lengths),
            sequence,
            planning_round,
            snapshot_sequence,
            sample_sequence,
            stats_schema_epoch,
            digest,
            key_digest,
            config_digest(model.config),
        )
        send_frame(self._connection, WireOp.PLAN, encode_request(request))
        self._active = request
        self._active_since = monotonic()
        return True

    def poll(self) -> RebalancePlan | None:
        """Return a completed plan without waiting for healthy planning work."""
        request = self._active
        if request is None:
            return None
        model = self._models[request.model_id]
        if self._process.poll() is not None:
            raise PlannerProcessError(f"STAIR planner exited with status {self._process.returncode}")
        if not self._connection.poll():
            if monotonic() - self._active_since > model.config.planner_heartbeat_timeout_s:
                self._process.kill()
                raise PlannerProcessError("STAIR planner health timeout")
            return None
        operation, payload = receive_frame(self._connection)
        if operation == WireOp.ERROR:
            raise PlannerProcessError(payload.decode(errors="replace"))
        if operation != WireOp.RESULT:
            raise PlannerProcessError("STAIR planner returned an unexpected operation")
        plan = decode_plan(payload, (model.shape.num_ranks, model.shape.slots_per_rank))
        expected = (
            request.model_id,
            request.planning_round,
            request.snapshot_sequence,
            request.sample_sequence,
            request.stats_schema_epoch,
            request.config_digest,
            model.topology.digest(),
        )
        actual = (
            plan.model_id,
            plan.planning_round,
            plan.snapshot_sequence,
            plan.sample_sequence,
            plan.stats_schema_epoch,
            plan.config_digest,
            plan.topology_digest,
        )
        if actual != expected:
            raise PlannerProcessError("STAIR planner result identity mismatch")
        self._active = None
        return plan

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.poll() is None:
                if self._active is not None:
                    operation, _ = self._receive(
                        self._models[self._active.model_id].config.planner_heartbeat_timeout_s
                    )
                    if operation != WireOp.RESULT:
                        raise PlannerProcessError("STAIR planner drain failed")
                    self._active = None
                send_frame(self._connection, WireOp.SHUTDOWN)
                shutdown_timeout = max(
                    model.config.planner_heartbeat_timeout_s for model in self._models.values()
                )
                operation, payload = self._receive(shutdown_timeout)
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

    def abort(self) -> None:
        """Release a failed child without waiting for graceful drain."""
        if self._closed:
            return
        try:
            if self._process.poll() is None:
                self._process.kill()
            self._process.wait()
        finally:
            self._closed = True
            self._connection.close()
            self._close_shared()
