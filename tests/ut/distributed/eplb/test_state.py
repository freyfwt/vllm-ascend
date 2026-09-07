# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.eplb import eplb_state as upstream_eplb_state

from vllm_ascend.distributed.eplb import state as eplb_state
from vllm_ascend.distributed.eplb.state import (
    AscendEplbLayerState,
    AscendEplbState,
)


def test_default_async_loop_delegates_to_upstream(monkeypatch):
    start = MagicMock()
    monkeypatch.setattr(upstream_eplb_state.EplbState, "start_async_loop", start)
    state = AscendEplbState.__new__(AscendEplbState)
    state.stair = None
    state.start_async_loop()
    start.assert_called_once_with(rank_mapping=None, is_profile=False)


def test_stair_step_records_polls_and_snapshots_without_upstream_policy():
    state = AscendEplbState.__new__(AscendEplbState)
    state.stair = MagicMock()
    state.stair.active_model_state = None
    state.model_states = {}
    state.expert_rearrangement_step = 1
    state.expert_rearrangement_step_interval = 2
    state.should_record_tensor = torch.zeros((), dtype=torch.bool)

    state.step()

    state.stair.record_step.assert_called_once_with()
    state.stair.poll_and_broadcast.assert_called_once_with()
    state.stair.start_next_layer.assert_called_once_with()
    state.stair.check_worker_health.assert_called_once_with()
    state.stair.snapshot.assert_called_once_with()
    assert state.should_record_tensor.item()


def test_stair_dummy_step_discards_execution_metadata():
    state = AscendEplbState.__new__(AscendEplbState)
    state.stair = MagicMock()
    state.stair.active_model_state = None
    state.expert_rearrangement_step = 0
    state.expert_rearrangement_step_interval = 2
    state.should_record_tensor = None
    state.step(is_dummy=True)
    state.stair.discard_step.assert_called_once_with()
    state.stair.record_step.assert_not_called()


def test_prepare_forward_marks_stair_model_execution(monkeypatch):
    prepare = MagicMock()
    monkeypatch.setattr(upstream_eplb_state.EplbState, "prepare_forward", prepare)
    state = AscendEplbState.__new__(AscendEplbState)
    state.stair = MagicMock()
    model_config = SimpleNamespace(compute_hash=lambda: "draft")
    state.prepare_forward(model_config, 3)
    prepare.assert_called_once_with(model_config, 3, None)
    state.stair.note_execution.assert_called_once_with("draft", False)


def test_layer_state_builds_routing_table_and_preserves_captured_tensor(
    monkeypatch,
):
    old_routing_table = torch.full((2, 2), -1, dtype=torch.int32)
    new_routing_table = torch.tensor([[0, 3], [2, 1]], dtype=torch.int32)
    build_routing_table = MagicMock(side_effect=[old_routing_table, new_routing_table])
    monkeypatch.setattr(
        eplb_state,
        "get_ep_group",
        lambda: SimpleNamespace(rank_in_group=1),
    )
    monkeypatch.setattr(
        eplb_state._eplb_ops,
        "build_expert_replica_routing_table",
        build_routing_table,
    )
    layer_state = AscendEplbLayerState()

    layer_state.set_layer_state(
        0,
        torch.zeros((1, 4), dtype=torch.int32),
        torch.tensor([[[0, 2], [1, 3]]], dtype=torch.int32),
        torch.tensor([[2, 2]], dtype=torch.int32),
    )
    captured_routing_table = layer_state.expert_replica_routing_table
    layer_state.refresh_expert_replica_routing_table()

    assert captured_routing_table is old_routing_table
    assert layer_state.expert_replica_routing_table is captured_routing_table
    torch.testing.assert_close(captured_routing_table, new_routing_table)


def test_stair_layer_state_rejects_routing_shape_change(monkeypatch):
    layer_state = AscendEplbLayerState()
    layer_state.logical_to_physical_map = torch.zeros(2, 2)
    layer_state.logical_replica_count = torch.ones(2)
    layer_state.expert_replica_routing_table = torch.zeros(2, 2)
    layer_state._stair_shape_locked = True
    monkeypatch.setattr(eplb_state, "get_ep_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(
        eplb_state._eplb_ops,
        "build_expert_replica_routing_table",
        lambda *_args: torch.zeros(3, 2),
    )
    with pytest.raises(RuntimeError, match="after graph capture"):
        layer_state.refresh_expert_replica_routing_table()


def test_sync_rearrange_refreshes_all_model_routing_tables(monkeypatch):
    sentinel = object()
    model_states = {"model": object()}

    def upstream_rearrange(self, is_profile=False, rank_mapping=None):
        assert not is_profile
        assert rank_mapping == {0: 0}
        return sentinel

    refresh = MagicMock()
    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "rearrange",
        upstream_rearrange,
    )
    monkeypatch.setattr(eplb_state, "refresh_model_routing_tables", refresh)
    state = AscendEplbState.__new__(AscendEplbState)
    state.is_async = False
    state.model_states = model_states

    result = state.rearrange(rank_mapping={0: 0})

    assert result is sentinel
    refresh.assert_called_once_with(model_states["model"])


def test_async_rearrange_defers_routing_refresh_to_workspace_hook(monkeypatch):
    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "rearrange",
        lambda self, is_profile=False, rank_mapping=None: None,
    )
    refresh = MagicMock()
    monkeypatch.setattr(eplb_state, "refresh_model_routing_tables", refresh)
    state = AscendEplbState.__new__(AscendEplbState)
    state.is_async = True
    state.model_states = {"model": object()}

    state.rearrange(rank_mapping={0: 0})

    refresh.assert_not_called()


def test_from_mapping_refreshes_final_mapping(monkeypatch):
    model_state = object()

    def upstream_from_mapping(cls, **kwargs):
        state = cls.__new__(cls)
        state.model_states = {"model": model_state}
        return state

    refresh = MagicMock()
    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "from_mapping",
        classmethod(upstream_from_mapping),
    )
    monkeypatch.setattr(eplb_state, "refresh_model_routing_tables", refresh)

    state = AscendEplbState.from_mapping(
        model=object(),
        model_config=object(),
        device=torch.device("cpu"),
        parallel_config=object(),
        expanded_physical_to_logical=torch.zeros(1),
    )

    assert isinstance(state, AscendEplbState)
    refresh.assert_called_once_with(model_state)


def test_from_mapping_forwards_release_valid_expert_count(monkeypatch):
    received_count = None

    def upstream_from_mapping(
        cls,
        model,
        model_config,
        device,
        parallel_config,
        expanded_physical_to_logical,
        num_valid_physical_experts,
    ):
        del model, model_config, device, parallel_config
        del expanded_physical_to_logical
        nonlocal received_count
        received_count = num_valid_physical_experts
        state = cls.__new__(cls)
        state.model_states = {}
        return state

    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "from_mapping",
        classmethod(upstream_from_mapping),
    )

    AscendEplbState.from_mapping(
        model=object(),
        model_config=object(),
        device=torch.device("cpu"),
        parallel_config=object(),
        expanded_physical_to_logical=torch.zeros((1, 2)),
        num_valid_physical_experts=1,
    )

    assert received_count == 1


def test_from_mapping_requires_release_valid_expert_count(monkeypatch):
    def upstream_from_mapping(
        cls,
        model,
        model_config,
        device,
        parallel_config,
        expanded_physical_to_logical,
        num_valid_physical_experts,
    ):
        raise AssertionError("release mapping must receive a valid count")

    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "from_mapping",
        classmethod(upstream_from_mapping),
    )

    with pytest.raises(TypeError, match="required by the selected vLLM release"):
        AscendEplbState.from_mapping(
            model=object(),
            model_config=object(),
            device=torch.device("cpu"),
            parallel_config=object(),
            expanded_physical_to_logical=torch.zeros((1, 2)),
        )


def test_init_sets_cuda_device_index_for_npu(monkeypatch):
    parallel_config = MagicMock()
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 5)
    monkeypatch.setattr(torch.cuda, "Event", torch.npu.Event)

    state = AscendEplbState(parallel_config, torch.device("cpu"))

    assert state.cuda_device_index == 5
