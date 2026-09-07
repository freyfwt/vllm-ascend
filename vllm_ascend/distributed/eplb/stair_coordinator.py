# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""EP-group planning and execution coordination for STAIR."""

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed import get_ep_group, get_eplb_group
from vllm.logger import logger

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.plan_validation import validate_plan
from vllm_ascend.distributed.eplb.planner_client import PlannerClient, PlannerModel
from vllm_ascend.distributed.eplb.planner_shared_memory import SharedSnapshotShape
from vllm_ascend.distributed.eplb.planner_wire import decode_plan, encode_plan
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology, RebalancePlan
from vllm_ascend.distributed.eplb.stair_runtime import StairModelRuntime, canonical_model_id
from vllm_ascend.distributed.eplb.stair_snapshot import sync_snapshot
from vllm_ascend.distributed.eplb.stair_worker import StairTransferWorker
from vllm_ascend.distributed.eplb.topology import discover_topology


@dataclass(frozen=True)
class PendingSnapshot:
    runtime: StairModelRuntime
    bin_sums: np.ndarray
    bin_lengths: np.ndarray
    placements: np.ndarray
    placement_epochs: np.ndarray
    accepted_scores: np.ndarray
    planning_round: int
    snapshot_sequence: int
    sample_sequence: int
    key_digest: str


class StairCoordinator:
    def __init__(self, config: StairConfig, phase: str, device: torch.device, window_size: int) -> None:
        self.config = config
        self.phase = phase
        self.device = device
        self.window_size = window_size
        self.models: dict[str, StairModelRuntime] = {}
        self.models_by_id: dict[str, StairModelRuntime] = {}
        self.topology: RankTopology | None = None
        self.planner: PlannerClient | None = None
        self.worker: StairTransferWorker | None = None
        self.planning_round = 0
        self.outer_step_key = 0
        self.stats_schema_epoch = 0
        self._pending: dict[str, Any] = {}
        self._submitted: Any | None = None
        self._execution_queue: list[tuple[RebalancePlan, StairModelRuntime, Any]] = []
        self._active_layer: tuple[RebalancePlan, StairModelRuntime, Any] | None = None
        self._planner_restarts = 0
        self.disabled = False

    def register(self, model_key: str, model_config: Any, model_state: Any, parallel_config: Any) -> None:
        role = "main" if not self.models else "draft"
        if role == "draft" and self.phase != "all":
            raise ValueError("Draft-model STAIR requires load_collection_phase='all'")
        ranks = get_ep_group().device_group.size()
        model_id = canonical_model_id(model_config, role, parallel_config)
        runtime = StairModelRuntime.create(
            model_id,
            model_state,
            window_size=self.window_size,
            num_ranks=ranks,
            device=self.device,
        )
        self.models[model_key] = runtime
        self.models_by_id[model_id] = runtime
        model_state._stair_runtime = runtime

        def commit_hook(layer_idx: int) -> None:
            layer = getattr(model_state, "_stair_pending_layer", None)
            if layer is None or layer.layer_idx != layer_idx:
                raise RuntimeError("STAIR committed a layer outside its active transaction")
            runtime.commit(layer)
            del model_state._stair_pending_layer

        model_state._stair_commit_hook = commit_hook

    def note_execution(self, model_key: str, has_prefill: bool) -> None:
        runtime = self.models.get(model_key)
        if runtime is not None:
            runtime.note_execution(has_prefill)

    def record_step(self) -> None:
        self.outer_step_key += 1
        default_executed = self.phase == "all" and len(self.models) > 1
        for runtime in self.models.values():
            runtime.record_step(self.outer_step_key, default_executed=default_executed)

    def start(self) -> None:
        ep_group, eplb_group = get_ep_group(), get_eplb_group()
        self.topology = discover_topology(ep_group, eplb_group)
        rank = ep_group.device_group.rank()
        if rank == 0:
            self.planner = PlannerClient(self._planner_models())
        device_index = torch.accelerator.current_device_index()
        self.worker = StairTransferWorker(device_index, rank)

    def _planner_models(self) -> tuple[PlannerModel, ...]:
        assert self.topology is not None
        return tuple(
            PlannerModel(
                runtime.model_id,
                SharedSnapshotShape(
                    self.config.sample_size,
                    runtime.ring.values.shape[1],
                    runtime.ring.values.shape[2],
                    runtime.num_ranks,
                    runtime.model_state.model.num_physical_experts // runtime.num_ranks,
                ),
                self.topology,
                self.config,
            )
            for runtime in self.models.values()
        )

    def _restart_planner(self) -> bool:
        if self._planner_restarts >= self.config.planner_restart_limit:
            return False
        try:
            if self.planner is not None:
                self.planner.abort()
            if self._submitted is not None:
                model_id = self._submitted.runtime.model_id
                current = self._pending.get(model_id)
                if current is None or current.snapshot_sequence < self._submitted.snapshot_sequence:
                    self._pending[model_id] = self._submitted
            self._submitted = None
            self.planner = PlannerClient(self._planner_models())
            self._planner_restarts += 1
            self._try_submit()
            return True
        except BaseException:
            logger.exception("STAIR planner restart failed")
            self.planner = None
            return False

    def snapshot(self) -> None:
        if self.topology is None:
            raise RuntimeError("STAIR coordinator has not started")
        self.planning_round += 1
        ep_group = get_ep_group()
        rank = ep_group.device_group.rank()
        for runtime in sorted(self.models.values(), key=lambda value: value.model_id):
            synchronized = sync_snapshot(
                runtime.ring,
                phase=self.phase,
                sample_size=self.config.sample_size,
                cpu_group=ep_group.cpu_group,
                device_group=ep_group.device_group,
                group_rank=rank,
            )
            if synchronized is None:
                continue
            runtime.accept_snapshot(synchronized.selected_keys)
            if rank == 0:
                assert synchronized.bin_sums is not None
                self._pending[runtime.model_id] = PendingSnapshot(
                    runtime,
                    synchronized.bin_sums.numpy(),
                    np.asarray(synchronized.bin_lengths, dtype=np.int64),
                    runtime.placements().astype(np.int32),
                    runtime.placement_epochs.copy(),
                    runtime.accepted_scores.copy(),
                    self.planning_round,
                    runtime.snapshot_sequence,
                    runtime.sample_sequence,
                    synchronized.key_digest,
                )
        self._try_submit()

    def _try_submit(self) -> None:
        if self.planner is None or self._submitted is not None or not self._pending:
            return
        pending = min(
            self._pending.values(),
            key=lambda value: (value.planning_round, value.runtime.model_id, value.snapshot_sequence),
        )
        submitted = self.planner.submit(
            pending.runtime.model_id,
            pending.bin_sums,
            pending.bin_lengths,
            pending.placements,
            pending.placement_epochs,
            pending.accepted_scores,
            planning_round=pending.planning_round,
            snapshot_sequence=pending.snapshot_sequence,
            sample_sequence=pending.sample_sequence,
            stats_schema_epoch=self.stats_schema_epoch,
            key_digest=pending.key_digest,
        )
        if submitted:
            self._submitted = pending
            del self._pending[pending.runtime.model_id]

    def poll_local_plan(self) -> RebalancePlan | None:
        if self.planner is None:
            return None
        plan = self.planner.poll()
        if plan is None:
            return None
        pending = self._submitted
        if pending is None or self.topology is None:
            raise RuntimeError("STAIR planner returned an untracked result")
        runtime = pending.runtime
        try:
            validate_plan(
                plan,
                runtime.placements(),
                runtime.placement_epochs,
                self.topology,
                self.config,
                model_id=runtime.model_id,
                current_planning_round=self.planning_round,
                snapshot_sequence=pending.snapshot_sequence,
                sample_sequence=pending.sample_sequence,
                stats_schema_epoch=self.stats_schema_epoch,
            )
        except ValueError as error:
            logger.warning("Discarding stale or invalid STAIR plan: %s", error)
            self._submitted = None
            self._try_submit()
            return None
        self._submitted = None
        self._try_submit()
        return plan

    def poll_and_broadcast(self) -> None:
        """Share at most one completed plan without pickle object collectives."""
        if self.topology is None or self.disabled:
            return
        ep_group = get_ep_group()
        cpu_group = ep_group.cpu_group
        rank = ep_group.device_group.rank()
        payload = b""
        status = 0
        if rank == 0:
            try:
                plan = self.poll_local_plan()
                payload = b"" if plan is None else encode_plan(plan)
                status = len(payload)
            except BaseException as error:
                if self._restart_planner():
                    logger.warning("Restarted STAIR planner after failure: %s", error)
                else:
                    logger.exception("Disabling STAIR after planner failure: %s", error)
                    status = -1
        length = torch.tensor((status,), dtype=torch.int64)
        source = dist.get_global_rank(cpu_group, 0)
        dist.broadcast(length, src=source, group=cpu_group)
        if int(length[0]) < 0:
            self.disabled = True
            return
        if int(length[0]) == 0:
            return
        data = torch.tensor(list(payload), dtype=torch.uint8) if rank == 0 else torch.empty(int(length[0]), dtype=torch.uint8)
        dist.broadcast(data, src=source, group=cpu_group)
        any_runtime = next(iter(self.models.values()))
        shape = (any_runtime.num_ranks, any_runtime.model_state.model.num_physical_experts // any_runtime.num_ranks)
        plan = decode_plan(bytes(data.tolist()), shape)
        runtime = self.models_by_id.get(plan.model_id)
        if runtime is None:
            self.disabled = True
            return
        self._execution_queue.extend((plan, runtime, layer) for layer in plan.layers)

    def _prepare_layer(self, plan: RebalancePlan, runtime: StairModelRuntime, layer: Any) -> bool:
        assert self.topology is not None
        valid = True
        try:
            validate_plan(
                dataclasses.replace(plan, layers=(layer,)),
                runtime.placements(),
                runtime.placement_epochs,
                self.topology,
                self.config,
                model_id=runtime.model_id,
                current_planning_round=self.planning_round,
                snapshot_sequence=plan.snapshot_sequence,
                sample_sequence=plan.sample_sequence,
                stats_schema_epoch=self.stats_schema_epoch,
            )
        except ValueError:
            valid = False
        flag = torch.tensor((int(valid),), dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=get_ep_group().cpu_group)
        return bool(flag[0])

    def start_next_layer(self) -> None:
        if self.disabled or self._active_layer is not None:
            return
        if self.worker is None or self.topology is None:
            raise RuntimeError("STAIR coordinator has not started")
        while self._execution_queue:
            plan, runtime, layer = self._execution_queue.pop(0)
            if not self._prepare_layer(plan, runtime, layer):
                continue
            runtime.model_state._stair_pending_layer = layer
            runtime.model_state.rebalanced = True
            self._active_layer = (plan, runtime, layer)
            from vllm_ascend.distributed.eplb.stair_worker import TransferWork

            self.worker.submit(TransferWork(runtime.model_state, layer, self.topology))
            return

    def finish_active_layer(self) -> None:
        if self._active_layer is None:
            raise RuntimeError("STAIR has no active layer to finish")
        _, runtime, _ = self._active_layer
        runtime.model_state.rebalanced = False
        self._active_layer = None
        self.start_next_layer()

    @property
    def active_model_state(self) -> Any | None:
        return None if self._active_layer is None else self._active_layer[1].model_state

    def check_worker_health(self) -> None:
        healthy = self.worker is not None and self.worker.failure is None
        flag = torch.tensor((int(healthy),), dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=get_ep_group().cpu_group)
        if not bool(flag[0]):
            raise RuntimeError("STAIR transfer failed after PREPARE; terminating consistently")

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()
        if self._active_layer is not None:
            _, runtime, _ = self._active_layer
            runtime.model_state.rebalanced = False
            if hasattr(runtime.model_state, "_stair_pending_layer"):
                del runtime.model_state._stair_pending_layer
            self._active_layer = None
        self._execution_queue.clear()
        if self.planner is not None:
            self.planner.close()
