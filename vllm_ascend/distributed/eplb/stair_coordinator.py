# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""EP-group planning and execution coordination for STAIR."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.distributed import get_ep_group, get_eplb_group

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.plan_validation import validate_plan
from vllm_ascend.distributed.eplb.planner_client import PlannerClient, PlannerModel
from vllm_ascend.distributed.eplb.planner_shared_memory import SharedSnapshotShape
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
            models = tuple(
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
            self.planner = PlannerClient(models)
        device_index = torch.accelerator.current_device_index()
        self.worker = StairTransferWorker(device_index, rank)

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
            runtime.snapshot_sequence += 1
            runtime.sample_sequence += synchronized.sample_count
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
        self._submitted = None
        self._try_submit()
        return plan
