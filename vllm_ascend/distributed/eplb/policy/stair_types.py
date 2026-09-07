# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Immutable values shared by the STAIR planner and executor."""

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BalanceScore:
    mean: float
    p95: float
    maximum: float


@dataclass(frozen=True)
class RankTopology:
    """Explicit EP/EPLB identities; algorithms never infer rank locality."""

    node_by_ep_rank: tuple[int, ...]
    eplb_rank_by_ep_rank: tuple[int, ...]

    def __post_init__(self) -> None:
        ranks = len(self.node_by_ep_rank)
        if ranks == 0 or sorted(self.eplb_rank_by_ep_rank) != list(range(ranks)):
            raise ValueError("STAIR requires a non-empty EP/EPLB rank bijection")

    @property
    def num_ranks(self) -> int:
        return len(self.node_by_ep_rank)

    def same_node(self, first: int, second: int) -> bool:
        return self.node_by_ep_rank[first] == self.node_by_ep_rank[second]

    def digest(self) -> str:
        payload = np.asarray(
            (self.node_by_ep_rank, self.eplb_rank_by_ep_rank),
            dtype="<i4",
        ).tobytes()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LayerPlan:
    layer_idx: int
    old_placement: np.ndarray
    new_placement: np.ndarray
    source_rank: np.ndarray
    source_slot: np.ndarray
    base_placement_epoch: int
    current_score: BalanceScore
    candidate_score: BalanceScore

    def __post_init__(self) -> None:
        shape = self.old_placement.shape
        arrays = (self.old_placement, self.new_placement, self.source_rank, self.source_slot)
        if len(shape) != 2 or any(value.shape != shape for value in arrays):
            raise ValueError("STAIR layer plan arrays must share a two-dimensional shape")
        for value in arrays:
            value.setflags(write=False)

    def execution_bytes(self) -> bytes:
        """Return the canonical discrete payload used by the plan digest."""
        header = np.asarray((self.layer_idx, self.base_placement_epoch), dtype="<i8")
        arrays = (self.old_placement, self.new_placement, self.source_rank, self.source_slot)
        return header.tobytes() + b"".join(np.asarray(value, dtype="<i4").tobytes() for value in arrays)


@dataclass(frozen=True)
class RebalancePlan:
    model_id: str
    planning_round: int
    snapshot_sequence: int
    sample_sequence: int
    stats_schema_epoch: int
    config_digest: str
    topology_digest: str
    layers: tuple[LayerPlan, ...]

    def digest(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.model_id,
            str(self.planning_round),
            str(self.snapshot_sequence),
            str(self.sample_sequence),
            str(self.stats_schema_epoch),
            self.config_digest,
            self.topology_digest,
        ):
            encoded = value.encode()
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
        for layer in self.layers:
            digest.update(layer.execution_bytes())
        return digest.hexdigest()


def validate_placement(placement: np.ndarray, num_experts: int) -> None:
    """Validate the stricter placement domain required by STAIR."""
    placement = np.asarray(placement)
    if placement.ndim != 2 or placement.size < num_experts:
        raise ValueError("STAIR placement must be [ranks, slots] and cover all experts")
    ranks, _ = placement.shape
    if placement.size > num_experts * ranks:
        raise ValueError("STAIR placement has more slots than unique expert/rank pairs")
    if np.any(placement < 0) or np.any(placement >= num_experts):
        raise ValueError("STAIR placement contains an invalid expert id")
    if len(np.unique(placement)) != num_experts:
        raise ValueError("STAIR placement must cover every logical expert")
    for rank, experts in enumerate(placement):
        if len(np.unique(experts)) != len(experts):
            raise ValueError(f"STAIR placement repeats an expert on EP rank {rank}")
