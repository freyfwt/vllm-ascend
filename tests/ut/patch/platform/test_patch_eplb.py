# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from vllm.config import EPLBConfig, ParallelConfig, VllmConfig
from vllm.config import parallel as parallel_module
from vllm.platforms import current_platform

from vllm_ascend.distributed.eplb.policy.stair_types import StairSourceMode, StairTopology
from vllm_ascend.distributed.eplb.transfer_plan import build_transfer_plan, source_ordering_context
from vllm_ascend.patch.platform import patch_eplb


class _FakeNpuPlatform:
    device_type = "npu"

    def __getattr__(self, name):
        return getattr(current_platform, name)


@contextmanager
def _npu_parallel_config_platform():
    proxy = parallel_module.current_platform
    assert isinstance(proxy, patch_eplb._CudaAlikeEplbPlatformProxy)
    original_platform = proxy._platform
    proxy._platform = _FakeNpuPlatform()
    try:
        yield
    finally:
        proxy._platform = original_platform


def test_parallel_and_vllm_config_keep_upstream_validation():
    with (
        _npu_parallel_config_platform(),
        patch("vllm_ascend.logger.configure_ascend_file_logging"),
        patch("vllm_ascend.logger.configure_ascend_logging"),
        patch("vllm.distributed.nixl_utils.is_nixl_available", return_value=False),
    ):
        parallel_config = ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config=EPLBConfig(use_async=True),
        )
        vllm_config = VllmConfig(parallel_config=parallel_config)

    assert vllm_config.parallel_config.enable_eplb
    assert vllm_config.parallel_config.eplb_config.communicator == "torch_gloo"
    assert getattr(ParallelConfig.__post_init__, patch_eplb._PATCH_MARKER, False)


def test_parallel_config_selects_gloo_before_upstream_nixl_probe():
    with (
        _npu_parallel_config_platform(),
        patch("vllm.distributed.nixl_utils.is_nixl_available") as is_nixl_available,
    ):
        parallel_config = ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config=EPLBConfig(use_async=True),
        )

    assert parallel_config.eplb_config.communicator == "torch_gloo"
    is_nixl_available.assert_not_called()


def test_parallel_config_platform_patch_is_idempotent():
    proxy = parallel_module.current_platform

    patch_eplb._patch_parallel_config()

    assert parallel_module.current_platform is proxy


def test_communicator_factory_maps_gloo_to_staged_on_npu(monkeypatch):
    communicator = object()
    gloo_cls = MagicMock(return_value=communicator)
    monkeypatch.setattr(patch_eplb, "AscendGlooEplbCommunicator", gloo_cls)
    coordinator = MagicMock()

    with _npu_parallel_config_platform():
        result = patch_eplb._eplb_communicator.create_eplb_communicator(
            coordinator,
            "torch_gloo",
            [[object()]],
            [object()],
        )

    assert result is communicator
    gloo_cls.assert_called_once_with(cpu_group=coordinator.cpu_group)


def test_communicator_factory_forwards_other_backends_and_additive_parameters():
    sentinel = object()
    calls = []

    def original_factory(
        group_coordinator,
        backend,
        expert_weights,
        expert_buffer,
        *,
        transport_options=None,
    ):
        calls.append(
            (
                group_coordinator,
                backend,
                expert_weights,
                expert_buffer,
                transport_options,
            )
        )
        return sentinel

    wrapped_factory = patch_eplb._wrap_communicator_factory(original_factory)
    coordinator = object()
    expert_weights = object()
    expert_buffer = object()

    with _npu_parallel_config_platform():
        result = wrapped_factory(
            coordinator,
            "nixl",
            expert_weights,
            expert_buffer,
            transport_options={"mode": "future"},
        )

    assert result is sentinel
    assert calls == [
        (
            coordinator,
            "nixl",
            expert_weights,
            expert_buffer,
            {"mode": "future"},
        )
    ]


def test_async_workspace_wrapper_refreshes_committed_layer(monkeypatch):
    call_order: list[str] = []
    consumed_event = MagicMock()
    consumed_event.record.side_effect = lambda _stream=None: call_order.append("ack")
    pending_result = SimpleNamespace(
        layer_idx=3,
        transfer_metadata=object(),
        consumed_event=consumed_event,
    )
    commit_policy_layer = MagicMock(side_effect=lambda *_args: call_order.append("policy"))
    state = SimpleNamespace(commit_policy_layer=commit_policy_layer)
    model_state = SimpleNamespace(
        pending_result=pending_result,
        rebalanced=True,
        _ascend_eplb_state=state,
        _ascend_eplb_committed_layers=0,
    )
    refresh = MagicMock(side_effect=lambda *_args: call_order.append("refresh"))
    monkeypatch.setattr(patch_eplb, "refresh_model_routing_tables", refresh)

    def original_move(model_state, ep_rank, *, future_option=None):
        assert ep_rank == 2
        assert future_option == "future"
        call_order.append("move")
        model_state.pending_result.consumed_event.record()
        model_state.pending_result = None
        return "moved"

    wrapped_move = patch_eplb._wrap_move_to_workspace(original_move)
    result = wrapped_move(model_state, 2, future_option="future")

    assert result == "moved"
    refresh.assert_called_once_with(model_state, 3)
    state.commit_policy_layer.assert_called_once_with(model_state, 3)
    assert model_state._ascend_eplb_committed_layers == 1
    assert call_order == ["move", "refresh", "policy", "ack"]


def test_async_workspace_wrapper_acknowledges_no_transfer_cycle(monkeypatch):
    consumed_event = MagicMock()
    pending_result = SimpleNamespace(
        layer_idx=1,
        transfer_metadata=patch_eplb.NO_TRANSFER_CYCLE_COMPLETE,
        consumed_event=consumed_event,
    )
    model_state = SimpleNamespace(
        pending_result=pending_result,
        rebalanced=True,
        model=SimpleNamespace(num_moe_layers=2),
    )
    original_move_called = False

    def original_move(model_state, ep_rank):
        nonlocal original_move_called
        original_move_called = True

    refresh = MagicMock()
    monkeypatch.setattr(patch_eplb, "refresh_model_routing_tables", refresh)

    wrapped_move = patch_eplb._wrap_move_to_workspace(original_move)
    result = wrapped_move(model_state, 0)

    assert result is None
    assert model_state.rebalanced is False
    assert model_state.pending_result is None
    consumed_event.record.assert_called_once_with()
    assert original_move_called is False
    refresh.assert_not_called()


def test_source_ordering_wrapper_reorders_without_changing_rank_sets(monkeypatch):
    current = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int64)
    candidate = np.array([[0, 1], [0, 2], [0, 3], [0, 3]], dtype=np.int64)
    topology = StairTopology.from_rank_to_node((0, 1, 1, 0))
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.transfer_plan._SOURCE_ORDERING_PATCH_ENABLED",
        True,
    )
    plan = build_transfer_plan(current, candidate, topology, expert_bytes=1)

    def original_helper(expert_ids, num_local_experts, old_indices, new_indices):
        del expert_ids, num_local_experts, old_indices, new_indices
        return {0: [0, 1]}, {0: [2, 3]}

    wrapped = patch_eplb._wrap_get_ep_ranks_with_experts_batch(original_helper)
    with source_ordering_context(plan):
        send_map, recv_map = wrapped(
            np.array([0]),
            2,
            current.reshape(-1),
            candidate.reshape(-1),
        )

    assert set(send_map[0]) == {0, 1}
    assert set(recv_map[0]) == {2, 3}
    assert recv_map[0] == list(plan.recv_order(0))
    assert plan.source_mode is StairSourceMode.PLUGIN_ORDERED
