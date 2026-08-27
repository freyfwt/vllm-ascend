# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend asynchronous EPLB execution with zero-change layer elision."""

# TODO(upstream-eplb): Delete this local worker fork once vLLM exposes hooks
# for changed-layer selection and explicit zero-transfer cycle completion.
# Until then, changes to the upstream async-worker protocol must be mirrored
# here and covered by contract tests so this copy cannot silently drift.

import threading
import time
from typing import TYPE_CHECKING

import torch
from vllm.distributed.eplb.eplb_utils import CpuGpuEvent
from vllm.distributed.eplb.rebalance_execute import AsyncEplbLayerResult
from vllm.distributed.parallel_state import get_eplb_group
from vllm.logger import logger

from vllm_ascend.distributed.eplb.policy import stair_constants as constants
from vllm_ascend.distributed.eplb.policy.stair_types import (
    StairExecutionMetrics,
    StairRebalancePlan,
    StairSourceMode,
)
from vllm_ascend.distributed.eplb.transfer_plan import transfer_layer_with_plan

if TYPE_CHECKING:
    from vllm.distributed.eplb.eplb_state import EplbModelState

    from vllm_ascend.distributed.eplb.state import AscendEplbState


class NoTransferCycleComplete:
    """Marker consumed by the main thread when a cycle has no final transfer."""


NO_TRANSFER_CYCLE_COMPLETE = NoTransferCycleComplete()
ASYNC_EPLB_CYCLE_COMMITTED_LOG = "Ascend async EPLB cycle committed"


def start_async_worker(
    state: "AscendEplbState",
    is_profile: bool = False,
) -> threading.Thread:
    """Start the state-owned Ascend transfer worker."""
    rank = get_eplb_group().device_group.rank()
    device_index: int | None = state.cuda_device_index
    assert state.is_async

    def thread_target() -> None:
        assert device_index is not None
        torch.accelerator.set_device_index(device_index)
        stream = torch.cuda.Stream(device=device_index)
        try:
            transfer_run_periodically(state=state, cuda_stream=stream, is_profile=is_profile)
        except Exception as exc:  # pragma: no cover - diagnostic path
            logger.exception("async loop error (Rank %d): %s", rank, str(exc))

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    return thread


def _run_rebalance_plan(
    model_state: "EplbModelState",
    physical_to_logical_map_cpu: torch.Tensor,
    cuda_stream: torch.cuda.Stream,
) -> StairRebalancePlan:
    assert model_state.eplb_stats is not None
    stats = model_state.eplb_stats
    with torch.cuda.stream(cuda_stream):
        load_window_cpu = stats.global_expert_load_window.cpu()
    plan = model_state._ascend_eplb_policy.plan_rebalance(
        load_window_cpu,
        stats.num_replicas,
        stats.num_groups,
        stats.num_nodes,
        stats.num_gpus,
        physical_to_logical_map_cpu,
    )
    if plan.new_mapping.device != torch.device("cpu"):
        raise RuntimeError(
            "Statistical Temporal-Aware Incremental Rebalancing (STAIR) must return a CPU physical-to-logical map."
        )
    model_state._ascend_eplb_policy_load = load_window_cpu
    model_state._ascend_eplb_active_plan = plan
    return plan


def _broadcast_rank_zero_plan(
    model_state: "EplbModelState",
    physical_to_logical_map_cpu: torch.Tensor,
    cuda_stream: torch.cuda.Stream,
    cpu_group,
    ep_rank: int,
) -> StairRebalancePlan:
    """Plan once on rank zero and install the immutable result everywhere."""
    if cpu_group.size() <= 1:
        return _run_rebalance_plan(model_state, physical_to_logical_map_cpu, cuda_stream)

    plan: StairRebalancePlan | None = None
    if ep_rank == 0:
        plan = _run_rebalance_plan(model_state, physical_to_logical_map_cpu, cuda_stream)
    payload = [plan]
    group_root = torch.distributed.get_global_rank(cpu_group, 0)
    torch.distributed.broadcast_object_list(payload, src=group_root, group=cpu_group)
    received = payload[0]
    if not isinstance(received, StairRebalancePlan):
        raise RuntimeError("STAIR rank-zero plan broadcast returned an invalid payload.")
    if ep_rank != 0:
        assert model_state.eplb_stats is not None
        with torch.cuda.stream(cuda_stream):
            load_window_cpu = model_state.eplb_stats.global_expert_load_window.cpu()
        model_state._ascend_eplb_policy_load = load_window_cpu
        model_state._ascend_eplb_active_plan = received
    return received


def _run_rebalance_experts(
    model_state: "EplbModelState",
    physical_to_logical_map_cpu: torch.Tensor,
    cuda_stream: torch.cuda.Stream,
) -> torch.Tensor:
    """Compatibility wrapper for focused tests and older local callers."""
    return _run_rebalance_plan(model_state, physical_to_logical_map_cpu, cuda_stream).new_mapping


def _validate_plan_budget(plan: StairRebalancePlan) -> None:
    usage = plan.budget_usage
    if plan.timed_out and plan.selected_layers:
        raise RuntimeError(f"STAIR timed-out plan {plan.plan_id} must not contain executable layers.")
    if plan.planner_elapsed_ms >= constants.MAX_PLANNER_MS and plan.selected_layers:
        raise RuntimeError(f"STAIR overdue plan {plan.plan_id} must not contain executable layers.")
    if (
        usage.selected_layers > constants.MAX_LAYERS_PER_CYCLE
        or usage.expert_transfers > constants.MAX_EXPERT_TRANSFERS_PER_CYCLE
        or usage.total_bytes > constants.MAX_TRANSFER_BYTES_PER_CYCLE
        or usage.cross_node_bytes > constants.MAX_CROSS_NODE_BYTES_PER_CYCLE
    ):
        raise RuntimeError(f"STAIR plan {plan.plan_id} exceeds an execution hard budget.")


def _publish_result(
    model_state: "EplbModelState",
    layer_idx: int,
    new_mapping: torch.Tensor,
    transfer_metadata,
    cuda_stream: torch.cuda.Stream,
    execution_metrics: StairExecutionMetrics | None = None,
) -> None:
    consumed_event = CpuGpuEvent()
    model_state.pending_result = AsyncEplbLayerResult(
        layer_idx=layer_idx,
        new_physical_to_logical_map=new_mapping,
        transfer_metadata=transfer_metadata,  # type: ignore[arg-type]
        consumed_event=consumed_event,
    )
    model_state._ascend_eplb_pending_execution_metrics = execution_metrics
    consumed_event.wait(stream=cuda_stream)
    assert model_state.pending_result is None
    model_state._ascend_eplb_pending_execution_metrics = None


def transfer_run_periodically(
    state: "AscendEplbState",
    cuda_stream: torch.cuda.Stream,
    is_profile: bool = False,
) -> None:
    """Transfer only changed layers and publish one explicit cycle completion."""
    while True:
        state.rearrange_event.wait(stream=cuda_stream)

        eplb_coordinator = get_eplb_group()
        eplb_group = eplb_coordinator.device_group
        eplb_cpu_group = eplb_coordinator.cpu_group
        ep_rank = eplb_group.rank()

        for model_state in state.model_states.values():
            model_state._ascend_eplb_committed_layers = 0
            model_state._ascend_eplb_committed_layer_ids = []
            model_state.communicator.set_stream(cuda_stream)
            with torch.cuda.stream(cuda_stream):
                old_mapping = model_state.physical_to_logical_map.cpu()
            plan = _broadcast_rank_zero_plan(
                model_state,
                old_mapping,
                cuda_stream,
                eplb_cpu_group,
                ep_rank,
            )
            _validate_plan_budget(plan)
            new_mapping = plan.new_mapping
            cycle_completed = True

            for layer_plan in plan.selected_layers:
                layer_idx = layer_plan.layer_idx
                flag = torch.tensor([int(model_state.rebalanced)], dtype=torch.int32, device="cpu")
                torch.distributed.all_reduce(flag, group=eplb_cpu_group)
                flag_sum = int(flag.item())
                if flag_sum != eplb_cpu_group.size():
                    logger.warning(
                        "async worker (rank=%d): layer %d coordinated stop (flag_sum=%d, group_size=%d)",
                        ep_rank,
                        layer_idx,
                        flag_sum,
                        eplb_cpu_group.size(),
                    )
                    model_state.rebalanced = False
                    cycle_completed = False
                    model_state._ascend_eplb_policy.abort_cycle(plan.plan_id)
                    break

                transfer_start = time.perf_counter()
                transfer_metadata, source_mode = transfer_layer_with_plan(
                    layer_plan=layer_plan,
                    old_layer_indices=old_mapping[layer_idx],
                    new_layer_indices=new_mapping[layer_idx],
                    expert_weights=model_state.model.expert_weights[layer_idx],
                    expert_weights_buffer=model_state.expert_buffer,
                    communicator=model_state.communicator,
                    ep_group=eplb_group,
                    is_profile=is_profile,
                    cuda_stream=cuda_stream,
                    layer_idx=layer_idx,
                )
                cuda_stream.synchronize()
                transfer_cost = layer_plan.transfer_cost
                if layer_plan.transfer_plan is not None:
                    transfer_cost = (
                        layer_plan.transfer_plan.cost
                        if source_mode is StairSourceMode.PLUGIN_ORDERED
                        else layer_plan.transfer_plan.executor_default_cost
                    )
                execution_metrics = StairExecutionMetrics(
                    plan_id=plan.plan_id,
                    layer_idx=layer_idx,
                    recv_count=int(transfer_metadata.recv_count),
                    actual_remote_bytes=transfer_cost.total_bytes - transfer_cost.local_copy_bytes,
                    transfer_elapsed_ms=(time.perf_counter() - transfer_start) * 1000,
                    source_mode=source_mode,
                )
                _publish_result(
                    model_state,
                    layer_idx,
                    new_mapping[layer_idx],
                    transfer_metadata,
                    cuda_stream,
                    execution_metrics,
                )

            if cycle_completed and model_state.rebalanced:
                _publish_result(
                    model_state,
                    model_state.model.num_moe_layers - 1,
                    new_mapping[-1],
                    NO_TRANSFER_CYCLE_COMPLETE,
                    cuda_stream,
                )
            if cycle_completed:
                committed_layer_ids = tuple(getattr(model_state, "_ascend_eplb_committed_layer_ids", ()))
                model_state._ascend_eplb_policy.finish_cycle(plan.plan_id, committed_layer_ids)
                model_state._ascend_eplb_active_plan = None
                committed_layers = getattr(model_state, "_ascend_eplb_committed_layers", 0)
                if ep_rank == 0 and committed_layers > 0:
                    logger.info(
                        "%s: model=%s, changed_layers=%d",
                        ASYNC_EPLB_CYCLE_COMMITTED_LOG,
                        model_state.model_name,
                        committed_layers,
                    )
