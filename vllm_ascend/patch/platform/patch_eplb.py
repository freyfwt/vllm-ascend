# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Patch the remaining vLLM EPLB construction points for Ascend."""

from functools import wraps
from inspect import signature

from vllm.config import parallel as _parallel_config
from vllm.distributed.eplb import async_worker as _eplb_async_worker
from vllm.distributed.eplb import eplb_communicator as _eplb_communicator
from vllm.distributed.eplb import eplb_state as _eplb_state

from vllm_ascend.distributed.eplb_communicator import (
    AscendGlooEplbCommunicator,
    HcclEplbCommunicator,
)
from vllm_ascend.distributed.eplb_state import refresh_model_routing_tables

_PATCH_MARKER = "_vllm_ascend_eplb_patch"
_NO_TRANSFER_CYCLE_COMPLETE = object()


class _CudaAlikeEplbPlatformProxy:
    """Delegate platform operations while exposing EPLB validation capability."""

    def __init__(self, platform) -> None:
        self._platform = platform

    def is_cuda_alike(self) -> bool:
        return _is_npu_platform(self._platform) or self._platform.is_cuda_alike()

    def __getattr__(self, name):
        return getattr(self._platform, name)


def _is_npu_platform(platform) -> bool:
    return getattr(platform, "device_type", None) == "npu"


def _patch_parallel_config() -> None:
    platform = _parallel_config.current_platform
    if not isinstance(platform, _CudaAlikeEplbPlatformProxy):
        # This module-local reference is read when ParallelConfig validates
        # EPLB. Communicator selection remains an NPUPlatform responsibility.
        _parallel_config.current_platform = _CudaAlikeEplbPlatformProxy(platform)


def _wrap_communicator_factory(original_factory):
    factory_signature = signature(original_factory)
    required_parameters = {
        "group_coordinator",
        "backend",
        "expert_weights",
        "expert_buffer",
    }
    if not required_parameters.issubset(factory_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: communicator factory signature changed.")

    @wraps(original_factory)
    def _create_eplb_communicator(*args, **kwargs):
        bound = factory_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        backend = bound.arguments["backend"]
        if _is_npu_platform(_parallel_config.current_platform):
            group_coordinator = bound.arguments["group_coordinator"]
            if backend == "torch_nccl":
                return HcclEplbCommunicator(group_coordinator.device_group)
            if backend == "torch_gloo":
                # The upstream factory accesses expert_weights[0][0].device to
                # pick the process group, but Ascend's EplbExpertTensorList
                # wraps per-expert tensors and does not expose .device at the
                # top level. Create the gloo communicator directly with the
                # cpu_group to bypass that device-type probe. The Ascend
                # subclass also disables the profile buffer reservation
                # collective, which is incompatible with EplbExpertTensorList.
                return AscendGlooEplbCommunicator(
                    cpu_group=group_coordinator.cpu_group,
                )
        return original_factory(*args, **kwargs)

    setattr(_create_eplb_communicator, _PATCH_MARKER, True)
    return _create_eplb_communicator


def _patch_communicator_factory() -> None:
    original_factory = _eplb_communicator.create_eplb_communicator
    if getattr(original_factory, _PATCH_MARKER, False):
        return
    wrapped_factory = _wrap_communicator_factory(original_factory)
    _eplb_communicator.create_eplb_communicator = wrapped_factory
    # eplb_state imports the factory by name, so update its retained binding.
    _eplb_state.create_eplb_communicator = wrapped_factory


def _wrap_move_to_workspace(original_move):
    move_signature = signature(original_move)
    required_parameters = {"model_state", "ep_rank"}
    if not required_parameters.issubset(move_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: async workspace move signature changed.")

    @wraps(original_move)
    def _move_to_workspace(*args, **kwargs):
        bound = move_signature.bind(*args, **kwargs)
        model_state = bound.arguments["model_state"]
        pending_result = model_state.pending_result
        layer_idx = pending_result.layer_idx if pending_result is not None else None
        if (
            pending_result is not None
            and getattr(pending_result, "transfer_metadata", None) is _NO_TRANSFER_CYCLE_COMPLETE
        ):
            assert layer_idx == model_state.model.num_moe_layers - 1
            model_state.rebalanced = False
            model_state.pending_result = None
            pending_result.consumed_event.record()
            return None
        result = original_move(*bound.args, **bound.kwargs)
        if layer_idx is not None:
            refresh_model_routing_tables(model_state, layer_idx)
        return result

    setattr(_move_to_workspace, _PATCH_MARKER, True)
    return _move_to_workspace


def _patch_async_move_to_workspace() -> None:
    original_move = _eplb_state._move_to_workspace
    if not getattr(original_move, _PATCH_MARKER, False):
        _eplb_state._move_to_workspace = _wrap_move_to_workspace(original_move)


def _transfer_run_periodically(
    state,
    cuda_stream,
    is_profile: bool = False,
) -> None:
    """Run upstream async EPLB while omitting unchanged layers entirely."""
    while True:
        state.rearrange_event.wait(stream=cuda_stream)

        eplb_group = _eplb_async_worker.get_eplb_group().device_group
        eplb_cpu_group = _eplb_async_worker.get_eplb_group().cpu_group
        ep_rank = eplb_group.rank()

        assert state.is_async
        for model_state in state.model_states.values():
            layer_idx = 0
            model_state.communicator.set_stream(cuda_stream)
            num_layers = model_state.model.num_moe_layers

            with _eplb_async_worker.torch.cuda.stream(cuda_stream):
                physical_to_logical_map_cpu = model_state.physical_to_logical_map.cpu()

            new_physical_to_logical_map = _eplb_async_worker.run_rebalance_experts(
                model_state,
                state,
                physical_to_logical_map_cpu,
                cuda_stream,
            )

            while layer_idx < num_layers:
                old_layer_indices = physical_to_logical_map_cpu[layer_idx]
                new_layer_indices = new_physical_to_logical_map[layer_idx]

                # Both tensors contain the complete global placement, so every
                # rank makes the same decision without another collective. An
                # unchanged layer needs no transfer, map commit, routing-table
                # refresh, stream synchronization, or main-thread acknowledgement.
                if _eplb_async_worker.torch.equal(
                    old_layer_indices,
                    new_layer_indices,
                ):
                    layer_idx += 1
                    continue

                flag = _eplb_async_worker.torch.tensor(
                    [int(model_state.rebalanced)],
                    dtype=_eplb_async_worker.torch.int32,
                    device="cpu",
                )
                _eplb_async_worker.torch.distributed.all_reduce(
                    flag,
                    group=eplb_cpu_group,
                )
                if int(flag.item()) != eplb_cpu_group.size():
                    _eplb_async_worker.logger.warning(
                        "async worker (rank=%d): layer %d coordinated stop (flag_sum=%d, group_size=%d)",
                        ep_rank,
                        layer_idx,
                        int(flag.item()),
                        eplb_cpu_group.size(),
                    )
                    model_state.rebalanced = False
                    break

                transfer_metadata = _eplb_async_worker.transfer_layer(
                    old_layer_indices=old_layer_indices,
                    new_layer_indices=new_layer_indices,
                    expert_weights=model_state.model.expert_weights[layer_idx],
                    expert_weights_buffer=model_state.expert_buffer,
                    communicator=model_state.communicator,
                    ep_group=eplb_group,
                    is_profile=is_profile,
                    cuda_stream=cuda_stream,
                    layer_idx=layer_idx,
                )

                cuda_stream.synchronize()
                consumed_event = _eplb_async_worker.CpuGpuEvent()
                model_state.pending_result = _eplb_async_worker.AsyncEplbLayerResult(
                    layer_idx=layer_idx,
                    new_physical_to_logical_map=new_layer_indices,
                    transfer_metadata=transfer_metadata,
                    consumed_event=consumed_event,
                )

                consumed_event.wait(stream=cuda_stream)
                assert model_state.pending_result is None
                layer_idx += 1

            # Upstream normally ends a cycle when the main thread consumes the
            # final layer. If that layer is skipped, publish a no-transfer
            # completion result so all main threads finish through the existing
            # all-ranks-ready protocol. Setting rebalanced=False directly here
            # races with the main-thread collective and can deadlock ranks.
            if layer_idx == num_layers and model_state.rebalanced:
                consumed_event = _eplb_async_worker.CpuGpuEvent()
                model_state.pending_result = _eplb_async_worker.AsyncEplbLayerResult(
                    layer_idx=num_layers - 1,
                    new_physical_to_logical_map=new_physical_to_logical_map[-1],
                    transfer_metadata=_NO_TRANSFER_CYCLE_COMPLETE,
                    consumed_event=consumed_event,
                )
                consumed_event.wait(stream=cuda_stream)
                assert model_state.pending_result is None


def _patch_async_transfer_worker() -> None:
    original_worker = _eplb_async_worker.transfer_run_periodically
    if getattr(original_worker, _PATCH_MARKER, False):
        return
    worker_signature = signature(original_worker)
    required_parameters = {"state", "cuda_stream", "is_profile"}
    if not required_parameters.issubset(worker_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: async worker signature changed.")
    setattr(_transfer_run_periodically, _PATCH_MARKER, True)
    _eplb_async_worker.transfer_run_periodically = _transfer_run_periodically


_patch_parallel_config()
_patch_communicator_factory()
_patch_async_move_to_workspace()
_patch_async_transfer_worker()
