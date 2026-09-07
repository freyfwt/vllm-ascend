# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""EP-group planning and execution coordination for STAIR."""

from typing import Any

import torch
from vllm.distributed import get_ep_group, get_eplb_group

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.planner_client import PlannerClient, PlannerModel
from vllm_ascend.distributed.eplb.planner_shared_memory import SharedSnapshotShape
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology
from vllm_ascend.distributed.eplb.stair_runtime import StairModelRuntime, canonical_model_id
from vllm_ascend.distributed.eplb.stair_worker import StairTransferWorker
from vllm_ascend.distributed.eplb.topology import discover_topology


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
