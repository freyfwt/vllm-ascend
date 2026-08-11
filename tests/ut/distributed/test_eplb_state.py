# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.distributed.eplb import eplb_state as upstream_eplb_state

from vllm_ascend.distributed import eplb_state
from vllm_ascend.distributed.eplb_policy import (
    FlashLBEplbPolicyAdapter,
    StairEplbPolicyAdapter,
    SwiftEplbPolicyAdapter,
)
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


def test_temporal_policy_rebuilds_window_when_upstream_aggregates_load(monkeypatch):
    physical_load_window = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=torch.int32,
    )
    model_state = SimpleNamespace(
        expert_load_window=physical_load_window,
        physical_to_logical_map=torch.tensor(
            [[0, 1, 0], [1, 0, 1]],
            dtype=torch.int32,
        ),
        model=SimpleNamespace(num_moe_layers=2, num_logical_experts=2),
    )
    state = AscendEplbState.__new__(AscendEplbState)
    state.model_states = {"model": model_state}
    state.num_valid_physical_experts = 3
    state._preserve_expert_load_time_series = True
    reduced_inputs = []

    def upstream_allreduce(self, tensors):
        reduced_inputs.extend(tensor.clone() for tensor in tensors)
        return tensors

    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "_allreduce_list",
        upstream_allreduce,
    )

    result = state._allreduce_list([torch.full((2, 2), -1, dtype=torch.int32)])

    expected = torch.tensor(
        [
            [[4, 2], [5, 10]],
            [[16, 8], [11, 22]],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(reduced_inputs[0], expected.reshape(-1, 2))
    torch.testing.assert_close(result[0], expected)


def test_temporal_policy_accepts_native_upstream_time_series(monkeypatch):
    state = AscendEplbState.__new__(AscendEplbState)
    state._preserve_expert_load_time_series = True
    first = torch.arange(12, dtype=torch.int32).reshape(2, 2, 3)
    second = torch.arange(18, dtype=torch.int32).reshape(2, 3, 3)
    reduced_inputs = []

    def upstream_allreduce(self, tensors):
        reduced_inputs.extend(tensor.clone() for tensor in tensors)
        return tensors

    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "_allreduce_list",
        upstream_allreduce,
    )

    result = state._allreduce_list([first, second])

    torch.testing.assert_close(reduced_inputs[0], first.reshape(-1, 3))
    torch.testing.assert_close(reduced_inputs[1], second.reshape(-1, 3))
    torch.testing.assert_close(result[0], first)
    torch.testing.assert_close(result[1], second)


def test_default_policy_keeps_upstream_allreduce_contract(monkeypatch):
    state = AscendEplbState.__new__(AscendEplbState)
    state._preserve_expert_load_time_series = False
    aggregated_load = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    upstream_allreduce = MagicMock(return_value=[aggregated_load])
    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "_allreduce_list",
        upstream_allreduce,
    )

    result = state._allreduce_list([aggregated_load])

    assert result[0] is aggregated_load
    upstream_allreduce.assert_called_once_with([aggregated_load])


def test_rearrange_scopes_temporal_allreduce_to_policy_cycle(monkeypatch):
    observed_flags = []

    def upstream_rearrange(self, is_profile=False, rank_mapping=None):
        observed_flags.append(self._preserve_expert_load_time_series)

    monkeypatch.setattr(
        upstream_eplb_state.EplbState,
        "rearrange",
        upstream_rearrange,
    )
    state = AscendEplbState.__new__(AscendEplbState)
    state.policy = SimpleNamespace(uses_expert_load_time_series=True)
    state.is_async = True
    state.model_states = {}

    state.rearrange(is_profile=True)

    assert observed_flags == [True]
    assert not state._preserve_expert_load_time_series


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


def test_normal_add_model_selects_flashlb_policy(monkeypatch):
    warm_up = MagicMock()
    monkeypatch.setattr(upstream_eplb_state.EplbState, "add_model", lambda self, model, model_config: None)
    monkeypatch.setattr(FlashLBEplbPolicyAdapter, "warm_up", warm_up)
    monkeypatch.setattr(
        eplb_state,
        "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(placement_policy="flashlb")),
    )
    state = AscendEplbState.__new__(AscendEplbState)

    state.add_model(object(), object())

    assert state.policy is FlashLBEplbPolicyAdapter
    warm_up.assert_called_once_with()


def test_normal_add_model_selects_stair_policy(monkeypatch):
    monkeypatch.setattr(upstream_eplb_state.EplbState, "add_model", lambda self, model, model_config: None)
    monkeypatch.setattr(
        eplb_state,
        "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(placement_policy="stair")),
    )
    state = AscendEplbState.__new__(AscendEplbState)

    state.add_model(object(), object())

    assert state.policy is StairEplbPolicyAdapter


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
