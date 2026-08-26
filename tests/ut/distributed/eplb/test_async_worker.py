# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.eplb import async_worker as eplb_async_worker
from vllm_ascend.distributed.eplb.policy.stair_types import (
    StairBalanceScore,
    StairBudgetUsage,
    StairLayerPlan,
    StairRebalancePlan,
    StairSourceMode,
    StairTransferCost,
)
from vllm_ascend.distributed.eplb.state import AscendEplbState


class _CycleComplete(Exception):
    pass


class _OneCycleEvent:
    def __init__(self):
        self.wait_count = 0

    def wait(self, stream):
        self.wait_count += 1
        if self.wait_count > 1:
            raise _CycleComplete


def test_rebalance_uses_model_owned_policy(monkeypatch):
    load_window = torch.ones((2, 1, 4), dtype=torch.int32)
    old_map = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    new_map = torch.tensor([[0, 2, 1, 3]], dtype=torch.int32)
    model_policy = MagicMock()
    plan = StairRebalancePlan(
        new_mapping=new_map,
        selected_layers=(),
        rejected_layers=(),
        budget_usage=StairBudgetUsage(),
        planner_elapsed_ms=1.0,
        plan_id="plan",
        topology_hash="flat",
    )
    model_policy.plan_rebalance.return_value = plan
    model_state = SimpleNamespace(
        _ascend_eplb_policy=model_policy,
        _ascend_eplb_policy_load=None,
        eplb_stats=SimpleNamespace(
            global_expert_load_window=load_window,
            num_replicas=4,
            num_groups=1,
            num_nodes=1,
            num_gpus=2,
        ),
    )
    monkeypatch.setattr(eplb_async_worker.torch.cuda, "stream", lambda stream: nullcontext())

    result = eplb_async_worker._run_rebalance_experts(model_state, old_map, MagicMock())

    assert result is new_map
    model_policy.plan_rebalance.assert_called_once_with(
        load_window,
        4,
        1,
        1,
        2,
        old_map,
    )
    assert model_state._ascend_eplb_policy_load is load_window
    assert model_state._ascend_eplb_active_plan is plan


def _plan_from_mapping(old_map: torch.Tensor, new_map: torch.Tensor) -> StairRebalancePlan:
    score = StairBalanceScore(1.0, 1.0, 1.0)
    selected = tuple(
        StairLayerPlan(
            layer_idx=layer_idx,
            placement=new_map[layer_idx].reshape(1, -1).numpy(),
            current_score=score,
            candidate_score=score,
            balance_gain=1.0,
            utility=float(old_map.shape[0] - layer_idx),
            transfer_cost=StairTransferCost(expert_transfers=1),
        )
        for layer_idx in range(old_map.shape[0])
        if not torch.equal(old_map[layer_idx], new_map[layer_idx])
    )
    usage = StairBudgetUsage(selected_layers=len(selected), expert_transfers=len(selected))
    return StairRebalancePlan(
        new_mapping=new_map,
        selected_layers=selected,
        rejected_layers=(),
        budget_usage=usage,
        planner_elapsed_ms=1.0,
        plan_id="plan",
        topology_hash="flat",
    )


def _run_one_cycle(monkeypatch, old_map, new_map, *, commit_results=True):
    pending_layers: list[int] = []
    completed_cycles: list[int] = []
    transfer_metadata = SimpleNamespace(recv_count=1)
    transfer_layer = MagicMock(return_value=(transfer_metadata, StairSourceMode.EXECUTOR_DEFAULT))
    all_reduce = MagicMock()
    cycle_log = MagicMock()

    class _ConsumedEvent:
        def wait(self, stream):
            result = model_state.pending_result
            if result.transfer_metadata is eplb_async_worker.NO_TRANSFER_CYCLE_COMPLETE:
                completed_cycles.append(result.layer_idx)
                model_state.rebalanced = False
            else:
                pending_layers.append(result.layer_idx)
                if commit_results:
                    model_state._ascend_eplb_committed_layers += 1
                    model_state._ascend_eplb_committed_layer_ids.append(result.layer_idx)
                    if result.layer_idx == model_state.model.num_moe_layers - 1:
                        model_state.rebalanced = False
            model_state.pending_result = None

    stream = MagicMock()
    model_state = SimpleNamespace(
        communicator=MagicMock(),
        model=SimpleNamespace(
            num_moe_layers=old_map.shape[0],
            expert_weights=[[object()]] * old_map.shape[0],
        ),
        model_name="model",
        physical_to_logical_map=old_map,
        expert_buffer=[object()],
        rebalanced=True,
        pending_result=None,
        _ascend_eplb_policy=MagicMock(),
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
    coordinator = SimpleNamespace(device_group=device_group, cpu_group=cpu_group)

    monkeypatch.setattr(eplb_async_worker, "get_eplb_group", lambda: coordinator)
    monkeypatch.setattr(eplb_async_worker, "_run_rebalance_plan", lambda *args: _plan_from_mapping(old_map, new_map))
    monkeypatch.setattr(eplb_async_worker, "transfer_layer_with_plan", transfer_layer)
    monkeypatch.setattr(eplb_async_worker, "CpuGpuEvent", _ConsumedEvent)
    monkeypatch.setattr(eplb_async_worker.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(eplb_async_worker.torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(eplb_async_worker.logger, "info", cycle_log)

    with pytest.raises(_CycleComplete):
        eplb_async_worker.transfer_run_periodically(cast(AscendEplbState, state), stream)

    return model_state, stream, transfer_layer, all_reduce, pending_layers, completed_cycles, cycle_log


def test_async_worker_skips_fully_unchanged_cycle(monkeypatch):
    placement = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    model_state, stream, transfer_layer, all_reduce, pending_layers, completed_cycles, cycle_log = _run_one_cycle(
        monkeypatch,
        placement,
        placement.clone(),
    )

    transfer_layer.assert_not_called()
    stream.synchronize.assert_not_called()
    all_reduce.assert_not_called()
    assert pending_layers == []
    assert completed_cycles == [1]
    assert model_state.rebalanced is False
    cycle_log.assert_not_called()


def test_async_worker_transfers_only_changed_layers(monkeypatch):
    old_map = torch.tensor([[0, 1], [0, 1], [1, 0]], dtype=torch.int32)
    new_map = torch.tensor([[0, 1], [1, 0], [1, 0]], dtype=torch.int32)
    model_state, stream, transfer_layer, all_reduce, pending_layers, completed_cycles, cycle_log = _run_one_cycle(
        monkeypatch,
        old_map,
        new_map,
    )

    transfer_layer.assert_called_once()
    assert transfer_layer.call_args.kwargs["layer_idx"] == 1
    stream.synchronize.assert_called_once_with()
    all_reduce.assert_called_once()
    assert pending_layers == [1]
    assert completed_cycles == [2]
    cycle_log.assert_called_once_with(
        "%s: model=%s, changed_layers=%d",
        eplb_async_worker.ASYNC_EPLB_CYCLE_COMMITTED_LOG,
        "model",
        1,
    )


def test_async_worker_does_not_log_acknowledged_but_uncommitted_cycle(monkeypatch):
    old_map = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    new_map = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    *_, cycle_log = _run_one_cycle(
        monkeypatch,
        old_map,
        new_map,
        commit_results=False,
    )

    cycle_log.assert_not_called()


def test_async_worker_logs_cycle_when_final_layer_changes(monkeypatch):
    old_map = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int32)
    new_map = torch.tensor([[0, 1], [0, 1], [1, 0]], dtype=torch.int32)
    model_state, _, transfer_layer, _, pending_layers, completed_cycles, cycle_log = _run_one_cycle(
        monkeypatch,
        old_map,
        new_map,
    )

    transfer_layer.assert_called_once()
    assert transfer_layer.call_args.kwargs["layer_idx"] == 2
    assert pending_layers == [2]
    assert completed_cycles == []
    assert model_state.rebalanced is False
    cycle_log.assert_called_once_with(
        "%s: model=%s, changed_layers=%d",
        eplb_async_worker.ASYNC_EPLB_CYCLE_COMMITTED_LOG,
        "model",
        1,
    )
