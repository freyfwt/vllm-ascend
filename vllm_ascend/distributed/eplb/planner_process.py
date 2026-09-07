# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Persistent single-threaded STAIR planner subprocess."""

import argparse
import ctypes
import dataclasses
import os
import signal
from contextlib import suppress
from multiprocessing.connection import Connection

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np

from vllm_ascend.distributed.eplb.planner_protocol import (
    PlannerRegistration,
    WireOp,
    decode_registration,
    receive_frame,
    send_frame,
)
from vllm_ascend.distributed.eplb.planner_shared_memory import SharedSnapshotBuffer
from vllm_ascend.distributed.eplb.planner_wire import decode_request, encode_plan
from vllm_ascend.distributed.eplb.policy.stair import plan_rebalance


class PlannerServer:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self.models: dict[str, tuple[PlannerRegistration, SharedSnapshotBuffer]] = {}
        self.last_sequence: dict[str, int] = {}

    def _initialize(self, payload: bytes) -> None:
        registration = decode_registration(payload)
        if registration.model_id in self.models:
            raise RuntimeError(f"Duplicate STAIR model {registration.model_id}")
        self.models[registration.model_id] = (
            registration,
            SharedSnapshotBuffer.attach(registration.shared),
        )
        self.last_sequence[registration.model_id] = 0
        send_frame(self.connection, WireOp.ACK, registration.model_id.encode())

    def _plan(self, payload: bytes) -> None:
        request = decode_request(payload)
        if request.model_id not in self.models:
            raise RuntimeError(f"Unknown STAIR model {request.model_id}")
        registration, shared = self.models[request.model_id]
        if request.slot_sequence <= self.last_sequence[request.model_id]:
            raise RuntimeError("STAIR shared snapshot sequence did not advance")
        if request.config_digest != request.config_digest.lower():
            raise RuntimeError("STAIR config digest is not canonical")
        arrays = shared.read(request.slot, request.bin_count, request.input_digest)
        bin_sums, lengths, placements, epochs, anchors = arrays
        samples = bin_sums.astype(np.float64) / lengths[:, None, None]
        accepted = tuple(None if np.isnan(value) else float(value) for value in anchors)
        plan = plan_rebalance(
            samples,
            placements,
            epochs,
            accepted,
            registration.topology,
            registration.config,
            sample_weights=lengths,
            model_id=request.model_id,
            planning_round=request.planning_round,
            snapshot_sequence=request.snapshot_sequence,
            sample_sequence=request.sample_sequence,
            stats_schema_epoch=request.stats_schema_epoch,
        )
        plan = dataclasses.replace(plan, config_digest=request.config_digest)
        self.last_sequence[request.model_id] = request.slot_sequence
        send_frame(self.connection, WireOp.RESULT, encode_plan(plan))

    def run(self) -> int:
        try:
            while True:
                operation, payload = receive_frame(self.connection)
                if operation == WireOp.INITIALIZE:
                    self._initialize(payload)
                elif operation == WireOp.PLAN:
                    self._plan(payload)
                elif operation == WireOp.SHUTDOWN:
                    send_frame(self.connection, WireOp.ACK, b"shutdown")
                    return 0
                else:
                    raise RuntimeError(f"Unexpected STAIR planner operation {operation.name}")
        except BaseException as error:
            with suppress(BaseException):
                send_frame(self.connection, WireOp.ERROR, f"{type(error).__name__}: {error}".encode())
            return 1
        finally:
            for _, shared in self.models.values():
                shared.close()
            self.connection.close()


def _arm_parent_death_signal() -> None:
    if not hasattr(signal, "SIGTERM") or os.uname().sysname != "Linux":
        return
    parent = os.getppid()
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(1, signal.SIGTERM) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent:
        raise RuntimeError("STAIR planner parent exited during startup")


def run_child(control_fd: int) -> int:
    _arm_parent_death_signal()
    return PlannerServer(Connection(control_fd)).run()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-fd", type=int, required=True)
    return run_child(parser.parse_args().control_fd)


if __name__ == "__main__":
    raise SystemExit(main())
