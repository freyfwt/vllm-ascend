# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import EPLBConfig, ParallelConfig, VllmConfig
from vllm.config import parallel as parallel_module
from vllm.platforms import current_platform

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
    with _npu_parallel_config_platform():
        parallel_config = ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config=EPLBConfig(use_async=False),
        )
        vllm_config = VllmConfig(parallel_config=parallel_config)

    assert vllm_config.parallel_config.enable_eplb
    assert not getattr(ParallelConfig.__post_init__, patch_eplb._PATCH_MARKER, False)


def test_parallel_config_platform_patch_is_idempotent():
    proxy = parallel_module.current_platform

    patch_eplb._patch_parallel_config()

    assert parallel_module.current_platform is proxy


def test_communicator_factory_maps_tensor_lists_to_hccl(monkeypatch):
    communicator = object()
    communicator_cls = MagicMock(return_value=communicator)
    monkeypatch.setattr(patch_eplb, "HcclEplbCommunicator", communicator_cls)
    coordinator = MagicMock()

    with _npu_parallel_config_platform():
        result = patch_eplb._eplb_communicator.create_eplb_communicator(
            coordinator,
            "torch_nccl",
            [[object()]],
            [object()],
        )

    assert result is communicator
    communicator_cls.assert_called_once_with(coordinator.device_group)


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
    pending_result = SimpleNamespace(layer_idx=3)
    model_state = SimpleNamespace(pending_result=pending_result)
    refresh = MagicMock()
    monkeypatch.setattr(patch_eplb, "refresh_model_routing_tables", refresh)

    def original_move(model_state, ep_rank, *, future_option=None):
        assert ep_rank == 2
        assert future_option == "future"
        model_state.pending_result = None
        return "moved"

    wrapped_move = patch_eplb._wrap_move_to_workspace(original_move)
    result = wrapped_move(model_state, 2, future_option="future")

    assert result == "moved"
    refresh.assert_called_once_with(model_state, 3)


class _CycleComplete(Exception):
    pass


class _OneCycleEvent:
    def __init__(self):
        self.wait_count = 0

    def wait(self, stream):
        self.wait_count += 1
        if self.wait_count > 1:
            raise _CycleComplete


def _run_one_async_cycle(monkeypatch, old_map, new_map):
    pending_layers = []
    transfer_metadata = object()
    transfer_layer = MagicMock(return_value=transfer_metadata)
    all_reduce = MagicMock()

    class _ConsumedEvent:
        def wait(self, stream):
            pending_layers.append(model_state.pending_result.layer_idx)
            model_state.pending_result = None

    stream = MagicMock()
    communicator = MagicMock()
    model_state = SimpleNamespace(
        communicator=communicator,
        model=SimpleNamespace(
            num_moe_layers=old_map.shape[0],
            expert_weights=[[object()]] * old_map.shape[0],
        ),
        physical_to_logical_map=old_map,
        expert_buffer=[object()],
        rebalanced=True,
        pending_result=None,
    )
    state = SimpleNamespace(
        rearrange_event=_OneCycleEvent(),
        is_async=True,
        model_states={"model": model_state},
    )
    device_group = MagicMock()
    device_group.rank.return_value = 0
    cpu_group = MagicMock()
    cpu_group.size.return_value = 1
    group = SimpleNamespace(device_group=device_group, cpu_group=cpu_group)

    monkeypatch.setattr(patch_eplb._eplb_async_worker, "get_eplb_group", lambda: group)
    monkeypatch.setattr(
        patch_eplb._eplb_async_worker,
        "run_rebalance_experts",
        lambda *args, **kwargs: new_map,
    )
    monkeypatch.setattr(patch_eplb._eplb_async_worker, "transfer_layer", transfer_layer)
    monkeypatch.setattr(patch_eplb._eplb_async_worker, "CpuGpuEvent", _ConsumedEvent)
    monkeypatch.setattr(patch_eplb._eplb_async_worker.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(patch_eplb._eplb_async_worker.torch.distributed, "all_reduce", all_reduce)

    with pytest.raises(_CycleComplete):
        patch_eplb._transfer_run_periodically(state, stream)

    return model_state, stream, communicator, transfer_layer, all_reduce, pending_layers


def test_async_worker_skips_fully_unchanged_cycle(monkeypatch):
    placement = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)

    model_state, stream, communicator, transfer_layer, all_reduce, pending_layers = _run_one_async_cycle(
        monkeypatch, placement, placement.clone()
    )

    communicator.set_stream.assert_called_once_with(stream)
    transfer_layer.assert_not_called()
    stream.synchronize.assert_not_called()
    all_reduce.assert_not_called()
    assert pending_layers == []
    assert model_state.pending_result is None
    assert model_state.rebalanced is False


def test_async_worker_transfers_only_changed_layers_and_completes_cycle(monkeypatch):
    old_map = torch.tensor([[0, 1], [0, 1], [1, 0]], dtype=torch.int32)
    new_map = torch.tensor([[0, 1], [1, 0], [1, 0]], dtype=torch.int32)

    model_state, stream, _, transfer_layer, all_reduce, pending_layers = _run_one_async_cycle(
        monkeypatch,
        old_map,
        new_map,
    )

    transfer_layer.assert_called_once()
    assert transfer_layer.call_args.kwargs["layer_idx"] == 1
    torch.testing.assert_close(
        transfer_layer.call_args.kwargs["old_layer_indices"],
        old_map[1],
    )
    torch.testing.assert_close(
        transfer_layer.call_args.kwargs["new_layer_indices"],
        new_map[1],
    )
    stream.synchronize.assert_called_once_with()
    all_reduce.assert_called_once()
    assert pending_layers == [1]
    assert model_state.pending_result is None
    assert model_state.rebalanced is False
