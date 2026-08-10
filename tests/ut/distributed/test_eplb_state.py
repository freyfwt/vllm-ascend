# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.distributed.eplb import eplb_state as upstream_eplb_state

from vllm_ascend.distributed import eplb_state
from vllm_ascend.distributed.eplb_policy import SwiftEplbPolicyAdapter
from vllm_ascend.distributed.eplb_state import (
    AscendEplbLayerState,
    AscendEplbState,
)


def test_layer_state_builds_routing_table_and_preserves_captured_tensor(monkeypatch):
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


def test_sync_rearrange_refreshes_all_model_routing_tables(monkeypatch):
    sentinel = object()
    model_states = {"model": object()}

    def upstream_rearrange(self, is_profile=False, rank_mapping=None):
        assert not is_profile
        assert rank_mapping == {0: 0}
        return sentinel

    refresh = MagicMock()
    monkeypatch.setattr(upstream_eplb_state.EplbState, "rearrange", upstream_rearrange)
    monkeypatch.setattr(eplb_state, "refresh_model_routing_tables", refresh)
    state = AscendEplbState.__new__(AscendEplbState)
    state.is_async = False
    state.model_states = model_states

    result = state.rearrange(rank_mapping={0: 0})

    assert result is sentinel
    refresh.assert_called_once_with(model_states["model"])


def test_async_rearrange_defers_routing_table_refresh_to_workspace_hook(monkeypatch):
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

    state.rearrange()

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
        num_valid_physical_experts=1,
    )

    assert isinstance(state, AscendEplbState)
    refresh.assert_called_once_with(model_state)


def test_init_sets_cuda_device_index_for_npu(monkeypatch):
    parallel_config = MagicMock()
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 5)
    # CpuGpuEvent (created in upstream EplbState.__init__) uses torch.cuda.Event,
    # which the MRv2 torch_cuda_wrapper remaps to torch.npu.Event at runtime.
    # Simulate that remapping for this unit test.
    monkeypatch.setattr(torch.cuda, "Event", torch.npu.Event)

    state = AscendEplbState(parallel_config, torch.device("cpu"))

    assert state.cuda_device_index == 5


def test_add_model_keeps_upstream_policy_without_override(monkeypatch):
    upstream_policy = object()

    def upstream_add_model(self, model, model_config):
        self.policy = upstream_policy

    monkeypatch.setattr(upstream_eplb_state.EplbState, "add_model", upstream_add_model)
    monkeypatch.setattr(
        eplb_state,
        "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(placement_policy=None)),
    )
    state = AscendEplbState.__new__(AscendEplbState)

    state.add_model(object(), object())

    assert state.policy is upstream_policy


def test_normal_add_model_selects_swift_policy(monkeypatch):
    monkeypatch.setattr(upstream_eplb_state.EplbState, "add_model", lambda self, model, model_config: None)
    monkeypatch.setattr(
        eplb_state,
        "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(placement_policy="swift")),
    )
    state = AscendEplbState.__new__(AscendEplbState)

    state.add_model(object(), object())

    assert state.policy is SwiftEplbPolicyAdapter


def test_from_mapping_selects_swift_policy_through_add_model(monkeypatch):
    def upstream_add_model(self, model, model_config):
        self.policy = object()

    def upstream_from_mapping(cls, model, model_config, **kwargs):
        state = cls.__new__(cls)
        state.model_states = {}
        state.add_model(model, model_config)
        return state

    monkeypatch.setattr(upstream_eplb_state.EplbState, "add_model", upstream_add_model)
    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "from_mapping",
        classmethod(upstream_from_mapping),
    )
    monkeypatch.setattr(
        eplb_state,
        "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(placement_policy="swift")),
    )

    state = AscendEplbState.from_mapping(
        model=object(),
        model_config=object(),
        device=torch.device("cpu"),
        parallel_config=object(),
        expanded_physical_to_logical=torch.zeros(1),
        num_valid_physical_experts=1,
    )

    assert state.policy is SwiftEplbPolicyAdapter
