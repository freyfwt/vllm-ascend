# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Per-model STAIR state that remains owned by the inference process."""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm_ascend.distributed.eplb.logical_ring import LogicalLoadRing
from vllm_ascend.distributed.eplb.policy.stair_types import LayerPlan, validate_placement


def canonical_model_id(model_config: Any, role: str, parallel_config: Any) -> str:
    identity = ":".join(
        (
            model_config.compute_hash(),
            role,
            str(parallel_config.tensor_parallel_size),
            str(parallel_config.pipeline_parallel_size),
            str(parallel_config.data_parallel_size),
            str(parallel_config.expert_parallel_size),
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()


@dataclass
class StairModelRuntime:
    model_id: str
    model_state: Any
    ring: LogicalLoadRing
    num_ranks: int
    placement_epochs: np.ndarray
    accepted_scores: np.ndarray
    snapshot_sequence: int = 0
    sample_sequence: int = 0
    last_sampled_outer_step: int = -1
    executed: bool = False
    has_prefill: bool = False

    @classmethod
    def create(
        cls,
        model_id: str,
        model_state: Any,
        *,
        window_size: int,
        num_ranks: int,
        device: torch.device,
    ) -> "StairModelRuntime":
        model = model_state.model
        if model.num_physical_experts % num_ranks:
            raise ValueError("STAIR physical experts must divide evenly across EP ranks")
        ring = LogicalLoadRing(
            window_size,
            model.num_moe_layers,
            model.num_logical_experts,
            device,
        )
        runtime = cls(
            model_id,
            model_state,
            ring,
            num_ranks,
            np.zeros(model.num_moe_layers, dtype=np.int64),
            np.full(model.num_moe_layers, np.nan, dtype=np.float64),
        )
        for layer in runtime.placements():
            validate_placement(layer, model.num_logical_experts)
        return runtime

    def note_execution(self, has_prefill: bool) -> None:
        self.executed = True
        self.has_prefill |= has_prefill

    def record_step(self, outer_step_key: int) -> None:
        self.ring.record(
            self.model_state.expert_load_pass,
            self.model_state.physical_to_logical_map,
            outer_step_key,
            executed=self.executed,
            has_prefill=self.has_prefill,
        )
        self.model_state.expert_load_pass.zero_()
        self.executed = False
        self.has_prefill = False

    def discard_step(self) -> None:
        self.model_state.expert_load_pass.zero_()
        self.executed = False
        self.has_prefill = False

    def placements(self) -> np.ndarray:
        mapping = self.model_state.physical_to_logical_map.detach().to(device="cpu", dtype=torch.long).numpy()
        return mapping.reshape(mapping.shape[0], self.num_ranks, -1)

    def accepted_score_tuple(self) -> tuple[float | None, ...]:
        return tuple(None if np.isnan(value) else float(value) for value in self.accepted_scores)

    def accept_snapshot(self, selected_keys: tuple[int, ...]) -> None:
        self.snapshot_sequence += 1
        self.sample_sequence += sum(key > self.last_sampled_outer_step for key in selected_keys)
        self.last_sampled_outer_step = max(selected_keys, default=self.last_sampled_outer_step)

    def commit(self, layer: LayerPlan) -> None:
        self.placement_epochs[layer.layer_idx] += 1
        self.accepted_scores[layer.layer_idx] = layer.candidate_score.mean
