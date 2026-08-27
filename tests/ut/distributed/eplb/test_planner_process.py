# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import socket
import threading
from multiprocessing.connection import Connection

import torch

from vllm_ascend.distributed.eplb.planner_process import StairPlannerServer
from vllm_ascend.distributed.eplb.planner_protocol import (
    PLANNER_PROTOCOL_VERSION,
    AffinityReadyRequest,
    FinalizeRequest,
    InitializeRequest,
    PlannerAck,
    PlannerModelRegistration,
    PlanRequest,
    PlanResponse,
    ShutdownRequest,
    receive_protocol_message,
    send_protocol_message,
)
from vllm_ascend.distributed.eplb.planner_shared_memory import StairSharedInputs
from vllm_ascend.distributed.eplb.policy import stair_constants as constants
from vllm_ascend.distributed.eplb.policy.stair_types import StairExecutionMetrics, StairSourceMode, StairTopology


def _registration(num_layers: int = 1) -> PlannerModelRegistration:
    return PlannerModelRegistration(
        model_id="main",
        load_shape=(2, num_layers, 4),
        mapping_shape=(num_layers, 6),
        num_replicas=6,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        topology=StairTopology.contiguous(2, 1),
        layer_expert_bytes=None,
    )


def _exchange(connection: Connection, request):
    send_protocol_message(connection, request)
    return receive_protocol_message(connection)


def test_planner_server_runs_versioned_shared_memory_transaction(monkeypatch):
    owner = StairSharedInputs.create(_registration())
    parent_socket, child_socket = socket.socketpair()
    parent_connection = Connection(parent_socket.detach())
    child_connection = Connection(child_socket.detach())
    server = StairPlannerServer(child_connection)
    result: list[int] = []
    thread = threading.Thread(target=lambda: result.append(server.run()))
    thread.start()
    try:
        initialized = _exchange(
            parent_connection,
            InitializeRequest(PLANNER_PROTOCOL_VERSION, (owner.spec,)),
        )
        assert isinstance(initialized, PlannerAck)
        affinity = (3, 7)
        monkeypatch.setattr(
            "vllm_ascend.distributed.eplb.planner_process.os.sched_getaffinity",
            lambda _pid: set(affinity),
            raising=False,
        )
        ready = _exchange(
            parent_connection,
            AffinityReadyRequest(PLANNER_PROTOCOL_VERSION, affinity),
        )
        assert isinstance(ready, PlannerAck)

        load = torch.tensor([[[1, 1, 100, 1]], [[1, 1, 120, 1]]], dtype=torch.float64)
        mapping = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long)
        slot, sequence, digest = owner.write(load, mapping)
        planned = _exchange(
            parent_connection,
            PlanRequest(
                protocol_version=PLANNER_PROTOCOL_VERSION,
                request_id=1,
                model_id="main",
                mapping_version=0,
                topology_epoch=0,
                tuning_version=constants.STAIR_TUNING_VERSION,
                slot=slot,
                slot_sequence=sequence,
                input_digest=digest,
            ),
        )
        assert isinstance(planned, PlanResponse)
        plan = planned.plan.to_plan()
        assert plan.selected_layers
        committed = tuple(layer.layer_idx for layer in plan.selected_layers)
        metrics = tuple(
            StairExecutionMetrics(
                plan_id=plan.plan_id,
                layer_idx=layer_idx,
                recv_count=1,
                actual_remote_bytes=0,
                transfer_elapsed_ms=1.0,
                source_mode=StairSourceMode.PLUGIN_ORDERED,
            )
            for layer_idx in committed
        )
        finalized = _exchange(
            parent_connection,
            FinalizeRequest(
                protocol_version=PLANNER_PROTOCOL_VERSION,
                request_id=1,
                model_id="main",
                mapping_version=len(committed),
                topology_epoch=0,
                tuning_version=constants.STAIR_TUNING_VERSION,
                committed_layer_ids=committed,
                execution_metrics=metrics,
            ),
        )
        assert isinstance(finalized, PlannerAck)
        shutdown = _exchange(parent_connection, ShutdownRequest(PLANNER_PROTOCOL_VERSION))
        assert isinstance(shutdown, PlannerAck)
    finally:
        thread.join(timeout=5)
        server.close()
        parent_connection.close()
        owner.close()
    assert result == [0]


def test_planner_server_commits_partial_history_before_abort(monkeypatch):
    owner = StairSharedInputs.create(_registration(num_layers=2))
    parent_socket, child_socket = socket.socketpair()
    parent_connection = Connection(parent_socket.detach())
    child_connection = Connection(child_socket.detach())
    server = StairPlannerServer(child_connection)
    result: list[int] = []
    thread = threading.Thread(target=lambda: result.append(server.run()))
    thread.start()
    try:
        _exchange(
            parent_connection,
            InitializeRequest(PLANNER_PROTOCOL_VERSION, (owner.spec,)),
        )
        affinity = (3, 7)
        monkeypatch.setattr(
            "vllm_ascend.distributed.eplb.planner_process.os.sched_getaffinity",
            lambda _pid: set(affinity),
            raising=False,
        )
        _exchange(
            parent_connection,
            AffinityReadyRequest(PLANNER_PROTOCOL_VERSION, affinity),
        )

        load = torch.tensor(
            [
                [[1, 1, 100, 1], [1, 1, 80, 1]],
                [[1, 1, 120, 1], [1, 1, 90, 1]],
            ],
            dtype=torch.float64,
        )
        mapping = torch.tensor(
            [[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 0, 1]],
            dtype=torch.long,
        )
        slot, sequence, digest = owner.write(load, mapping)
        planned = _exchange(
            parent_connection,
            PlanRequest(
                protocol_version=PLANNER_PROTOCOL_VERSION,
                request_id=1,
                model_id="main",
                mapping_version=0,
                topology_epoch=0,
                tuning_version=constants.STAIR_TUNING_VERSION,
                slot=slot,
                slot_sequence=sequence,
                input_digest=digest,
            ),
        )
        assert isinstance(planned, PlanResponse)
        plan = planned.plan.to_plan()
        assert len(plan.selected_layers) == 2
        committed_layer = plan.selected_layers[0].layer_idx
        metric = StairExecutionMetrics(
            plan_id=plan.plan_id,
            layer_idx=committed_layer,
            recv_count=1,
            actual_remote_bytes=0,
            transfer_elapsed_ms=1.0,
            source_mode=StairSourceMode.PLUGIN_ORDERED,
        )
        finalized = _exchange(
            parent_connection,
            FinalizeRequest(
                protocol_version=PLANNER_PROTOCOL_VERSION,
                request_id=1,
                model_id="main",
                mapping_version=1,
                topology_epoch=0,
                tuning_version=constants.STAIR_TUNING_VERSION,
                committed_layer_ids=(committed_layer,),
                execution_metrics=(metric,),
                aborted=True,
            ),
        )
        assert isinstance(finalized, PlannerAck)
        assert committed_layer in server._models["main"].policy.average_to_peak_history
        assert server._models["main"].pending is None
        _exchange(parent_connection, ShutdownRequest(PLANNER_PROTOCOL_VERSION))
    finally:
        thread.join(timeout=5)
        server.close()
        parent_connection.close()
        owner.close()
    assert result == [0]
