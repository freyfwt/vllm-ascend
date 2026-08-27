# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Statistical Temporal-Aware Incremental Rebalancing (STAIR) policy."""

import hashlib
import time
from collections import Counter
from dataclasses import replace

import numpy as np
import torch
from vllm.logger import logger

from vllm_ascend.distributed.eplb.policy import stair_constants as constants
from vllm_ascend.distributed.eplb.policy.stair_search import (
    MiniFlashTreeSearch,
    SearchCandidate,
    build_candidate_from_replica_counts,
    build_greedy_candidate,
    build_greedy_layer_candidate,
    validate_layer_placement,
)
from vllm_ascend.distributed.eplb.policy.stair_stats import (
    StairLoadStats,
    compute_balance_gain,
    compute_balance_metrics,
    rank_loads,
    replica_counts,
)
from vllm_ascend.distributed.eplb.policy.stair_types import (
    StairBalanceScore,
    StairBudgetUsage,
    StairCandidateKind,
    StairExecutionMetrics,
    StairLayerPlan,
    StairRebalancePlan,
    StairRejectReason,
    StairSourceMode,
    StairTopology,
    StairTransferCost,
)
from vllm_ascend.distributed.eplb.transfer_plan import build_transfer_plan


def _replica_counts(placement: np.ndarray, num_experts: int) -> np.ndarray:
    return replica_counts(placement, num_experts)


def _rank_loads(expert_load: np.ndarray, placement: np.ndarray, num_experts: int) -> np.ndarray:
    return rank_loads(expert_load, placement, num_experts)


def compute_balance_score(expert_load: np.ndarray, placement: np.ndarray) -> float:
    """Return the mean peak-to-average rank load for one MoE layer."""
    return compute_balance_metrics(expert_load, placement).mean_imbalance


def _validate_layer_placement(placement: np.ndarray, num_experts: int, num_ranks: int) -> None:
    validate_layer_placement(placement, num_experts, num_ranks)


def _build_incremental_candidate(
    logical_load: np.ndarray,
    current_placement: np.ndarray,
    num_ranks: int,
    topology: StairTopology | None = None,
) -> np.ndarray:
    """Compatibility wrapper around the topology-aware greedy search."""
    resolved_topology = topology or StairTopology.contiguous(num_ranks, 1)
    return build_greedy_candidate(logical_load, current_placement, resolved_topology)


class StairEplbPolicy:
    """Generate and temporally filter incremental expert placements."""

    uses_expert_load_time_series = True

    def __init__(
        self,
        topology: StairTopology | None = None,
        layer_expert_bytes: tuple[int, ...] | None = None,
        *,
        runtime_history: bool = True,
    ) -> None:
        constants.validate_stair_constants()
        self._configured_topology = topology
        self._layer_expert_bytes = layer_expert_bytes
        self._runtime_history = runtime_history
        self.average_to_peak_history: dict[int, float] = {}
        self.load_stats = StairLoadStats()
        self._shape_key: tuple[int, ...] | None = None
        self._expected_layer_placements: dict[int, np.ndarray] = {}
        self._pending_plan: StairRebalancePlan | None = None
        self.execution_metrics: list[StairExecutionMetrics] = []
        self._mini_search = MiniFlashTreeSearch()

    def configure_runtime(
        self,
        layer_expert_bytes: tuple[int, ...] | None,
        *,
        topology: StairTopology | None = None,
    ) -> None:
        """Update setup-only weight sizes after rejecting active-plan mutation."""
        if self._pending_plan is not None:
            raise RuntimeError("Cannot reconfigure STAIR while a rebalance plan is active.")
        if self._layer_expert_bytes != layer_expert_bytes or (
            topology is not None and self._configured_topology != topology
        ):
            self._layer_expert_bytes = layer_expert_bytes
            if topology is not None:
                self._configured_topology = topology
            self.average_to_peak_history.clear()
            self._expected_layer_placements.clear()
            self.load_stats.reset()
            self._shape_key = None

    def _resolve_topology(self, num_nodes: int, num_ranks: int) -> StairTopology:
        topology = self._configured_topology
        if topology is None:
            return StairTopology.contiguous(num_ranks, num_nodes)
        if len(topology.rank_to_node) != num_ranks:
            raise ValueError("STAIR discovered topology does not match num_ranks.")
        if not topology.is_flat_fallback and topology.num_nodes != num_nodes:
            raise ValueError("STAIR discovered topology does not match num_nodes.")
        return topology

    def _prepare_history(
        self,
        expert_load: np.ndarray,
        current_placement: np.ndarray,
        topology: StairTopology,
    ) -> None:
        shape_key = (
            expert_load.shape[1],
            expert_load.shape[2],
            current_placement.shape[1],
            len(topology.rank_to_node),
            int(topology.topology_hash, 16),
            constants.STAIR_TUNING_VERSION,
        )
        if self._shape_key != shape_key:
            self.average_to_peak_history.clear()
            self._expected_layer_placements.clear()
            self.load_stats.reset()
            self._shape_key = shape_key
            return

        reset_layers: list[int] = []
        for layer_id, expected in list(self._expected_layer_placements.items()):
            if layer_id >= current_placement.shape[0] or not np.array_equal(current_placement[layer_id], expected):
                self.average_to_peak_history.pop(layer_id, None)
                self._expected_layer_placements.pop(layer_id, None)
                reset_layers.append(layer_id)
        self.load_stats.reset_layers(reset_layers)

    def _needs_temporal_update(self, layer_id: int, current_score: float, num_ranks: int) -> bool:
        past_ratio = self.average_to_peak_history.get(layer_id)
        if past_ratio is None:
            return True
        if num_ranks < constants.SMALL_WORLD_SIZE:
            threshold_ratio = constants.SMALL_WORLD_UPDATE_THRESHOLD_RATIO
            threshold_value = constants.SMALL_WORLD_UPDATE_THRESHOLD_VALUE
        else:
            threshold_ratio = constants.TEMPORAL_UPDATE_THRESHOLD_RATIO
            threshold_value = constants.TEMPORAL_UPDATE_THRESHOLD_VALUE
        current_ratio = 1.0 / current_score
        return current_ratio < past_ratio * threshold_ratio or current_ratio < threshold_value

    def _prepare_inputs(
        self,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if old_global_expert_indices is None:
            raise ValueError("STAIR EPLB requires the current physical-to-logical map.")
        if num_replicas <= 0 or num_groups <= 0 or num_nodes <= 0 or num_ranks <= 0:
            raise ValueError("num_replicas, num_groups, num_nodes, and num_ranks must be positive.")
        if num_replicas % num_ranks != 0:
            raise ValueError(f"num_replicas ({num_replicas}) must be divisible by num_ranks ({num_ranks}).")
        if num_ranks % num_nodes != 0:
            raise ValueError(f"num_ranks ({num_ranks}) must be divisible by num_nodes ({num_nodes}).")
        if weight.ndim != 3:
            raise ValueError(f"STAIR EPLB requires [window, layers, experts] load, got {tuple(weight.shape)}.")
        if weight.device.type != "cpu" or old_global_expert_indices.device.type != "cpu":
            raise ValueError("STAIR EPLB policy inputs must be CPU tensors.")
        if old_global_expert_indices.shape != (weight.shape[1], num_replicas):
            raise ValueError(
                "Current placement shape must be [layers, num_replicas], got "
                f"{tuple(old_global_expert_indices.shape)} for weight shape {tuple(weight.shape)} "
                f"and num_replicas={num_replicas}."
            )

        expert_load = weight.detach().to(dtype=torch.float64).contiguous().numpy()
        current = old_global_expert_indices.detach().to(dtype=torch.long).contiguous().numpy().copy()
        if not np.all(np.isfinite(expert_load)) or np.any(expert_load < 0):
            raise ValueError("expert_load must contain finite, non-negative values.")
        if expert_load.shape[0] == 0 or expert_load.shape[2] == 0:
            raise ValueError("expert_load window and expert dimensions must be nonzero.")

        slots_per_rank = num_replicas // num_ranks
        current_by_rank = current.reshape(current.shape[0], num_ranks, slots_per_rank)
        for layer_placement in current_by_rank:
            _validate_layer_placement(layer_placement, expert_load.shape[2], num_ranks)
        if self._layer_expert_bytes is not None and len(self._layer_expert_bytes) != current.shape[0]:
            raise ValueError("STAIR layer expert byte sizes must match the number of MoE layers.")
        return expert_load, current, slots_per_rank

    def _build_layer_plan(
        self,
        layer_idx: int,
        layer_load: np.ndarray,
        current: np.ndarray,
        candidate: np.ndarray,
        num_ranks: int,
        topology: StairTopology | None = None,
        candidate_kind: StairCandidateKind = StairCandidateKind.GREEDY,
        search_candidates: int = 0,
        search_elapsed_ms: float = 0.0,
        current_score: StairBalanceScore | None = None,
        check_temporal: bool = True,
    ) -> StairLayerPlan:
        resolved_topology = topology or StairTopology.contiguous(num_ranks, 1)
        if current_score is None:
            current_score = compute_balance_metrics(layer_load, current)
        candidate_score = compute_balance_metrics(layer_load, candidate)
        gain = compute_balance_gain(current_score, candidate_score)
        accepted = True
        reject_reason = StairRejectReason.NONE
        if check_temporal and not self._needs_temporal_update(layer_idx, current_score.mean_imbalance, num_ranks):
            accepted = False
            reject_reason = StairRejectReason.TEMPORAL_GATE
        elif candidate_score.p95_imbalance > current_score.p95_imbalance + constants.TAIL_REGRESSION_TOLERANCE:
            accepted = False
            reject_reason = StairRejectReason.TAIL_REGRESSION
        elif gain <= constants.MIN_BALANCE_GAIN:
            accepted = False
            reject_reason = StairRejectReason.NO_BALANCE_GAIN

        if not accepted:
            return StairLayerPlan(
                layer_idx=layer_idx,
                placement=candidate.copy(),
                current_score=current_score,
                candidate_score=candidate_score,
                balance_gain=gain,
                utility=0.0,
                transfer_cost=StairTransferCost(),
                candidate_kind=candidate_kind,
                source_mode=StairSourceMode.EXECUTOR_DEFAULT,
                accepted=False,
                reject_reason=reject_reason,
                search_candidates=search_candidates,
                search_elapsed_ms=search_elapsed_ms,
            )

        expert_bytes = 0 if self._layer_expert_bytes is None else self._layer_expert_bytes[layer_idx]
        transfer_plan = build_transfer_plan(
            current,
            candidate,
            resolved_topology,
            expert_bytes,
        )
        transfer_cost = transfer_plan.budget_cost
        utility_denominator = max(
            transfer_plan.cost.weighted_bytes,
            float(transfer_plan.cost.expert_transfers),
            1.0,
        )
        utility = gain / utility_denominator
        redundant_slots = current.size - layer_load.shape[1]
        layer_transfer_limit = max(
            2,
            redundant_slots * constants.MAX_LAYER_TRANSFER_REDUNDANCY_MULTIPLIER,
        )
        if transfer_cost.max_rank_pair_transfers > constants.MAX_TRANSFERS_PER_RANK_PAIR:
            accepted = False
            reject_reason = StairRejectReason.INVALID_TOPOLOGY
        elif transfer_cost.expert_transfers > layer_transfer_limit:
            accepted = False
            reject_reason = StairRejectReason.LAYER_BUDGET
        elif transfer_cost.weighted_bytes > 0:
            gain_per_gib = gain / (transfer_cost.weighted_bytes / 1024**3)
            if gain_per_gib < constants.MIN_GAIN_PER_GIB:
                accepted = False
                reject_reason = StairRejectReason.LOW_UTILITY
        return StairLayerPlan(
            layer_idx=layer_idx,
            placement=candidate.copy(),
            current_score=current_score,
            candidate_score=candidate_score,
            balance_gain=gain,
            utility=utility,
            transfer_cost=transfer_cost,
            transfer_plan=transfer_plan,
            candidate_kind=candidate_kind,
            source_mode=transfer_plan.source_mode,
            accepted=accepted,
            reject_reason=reject_reason,
            search_candidates=search_candidates,
            search_elapsed_ms=search_elapsed_ms,
        )

    def _search_layer_plan(
        self,
        greedy_plan: StairLayerPlan,
        layer_load: np.ndarray,
        layer_risk_load: np.ndarray,
        current: np.ndarray,
        num_ranks: int,
        topology: StairTopology,
    ) -> StairLayerPlan:
        """Search a small replica-count neighborhood and reuse all hard gates."""
        root_counts = _replica_counts(greedy_plan.placement, layer_risk_load.size)
        root = SearchCandidate(
            placement=greedy_plan.placement,
            replica_counts=root_counts,
            objective=greedy_plan.utility,
            path_key=tuple(int(value) for value in root_counts),
        )

        def evaluate(counts: np.ndarray) -> SearchCandidate | None:
            placement = build_candidate_from_replica_counts(
                layer_risk_load,
                current,
                topology,
                counts,
            )
            if np.array_equal(placement, current):
                return None
            current_score = compute_balance_metrics(layer_load, current)
            candidate_score = compute_balance_metrics(layer_load, placement)
            gain = compute_balance_gain(current_score, candidate_score)
            if (
                gain <= constants.MIN_BALANCE_GAIN
                or candidate_score.p95_imbalance > current_score.p95_imbalance + constants.TAIL_REGRESSION_TOLERANCE
            ):
                return None
            expert_bytes = 0 if self._layer_expert_bytes is None else self._layer_expert_bytes[greedy_plan.layer_idx]
            transfer_plan = build_transfer_plan(
                current,
                placement,
                topology,
                expert_bytes,
            )
            cost = transfer_plan.budget_cost
            if cost.max_rank_pair_transfers > constants.MAX_TRANSFERS_PER_RANK_PAIR or cost.expert_transfers > max(
                2,
                (current.size - layer_load.shape[1]) * constants.MAX_LAYER_TRANSFER_REDUNDANCY_MULTIPLIER,
            ):
                return None
            objective = gain / max(
                transfer_plan.cost.weighted_bytes,
                float(transfer_plan.cost.expert_transfers),
                1.0,
            )
            return SearchCandidate(
                placement=placement,
                replica_counts=counts.copy(),
                objective=objective,
                path_key=tuple(int(value) for value in counts),
            )

        result = self._mini_search.search(
            root,
            layer_risk_load,
            num_ranks,
            evaluate,
        )
        if result.candidate.path_key == root.path_key:
            return replace(
                greedy_plan,
                search_candidates=result.evaluated_candidates,
                search_elapsed_ms=result.elapsed_ms,
            )
        return self._build_layer_plan(
            greedy_plan.layer_idx,
            layer_load,
            current,
            result.candidate.placement,
            num_ranks,
            topology,
            StairCandidateKind.MINI_FLASH_TREE,
            result.evaluated_candidates,
            result.elapsed_ms,
            check_temporal=False,
        )

    @staticmethod
    def _compute_plan_digest(
        current: np.ndarray,
        selected_layers: tuple[StairLayerPlan, ...],
        topology_hash: str,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(current.tobytes())
        digest.update(topology_hash.encode())
        digest.update(str(constants.STAIR_TUNING_VERSION).encode())
        digest.update(
            (f"{constants.MAX_TRANSFERS_PER_RANK_PAIR}:{constants.MAX_LAYER_TRANSFER_REDUNDANCY_MULTIPLIER}").encode()
        )
        for layer_plan in selected_layers:
            digest.update(layer_plan.layer_idx.to_bytes(4, byteorder="little", signed=False))
            digest.update(layer_plan.placement.tobytes())
            if layer_plan.transfer_plan is not None:
                digest.update(layer_plan.transfer_plan.source_rank_by_position.tobytes())
                digest.update(layer_plan.transfer_plan.source_mode.value.encode())
        return digest.hexdigest()[:16]

    def plan_rebalance(
        self,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> StairRebalancePlan:
        """Return a deterministic, budgeted execution plan on CPU."""
        expert_load, current, slots_per_rank = self._prepare_inputs(
            weight,
            num_replicas,
            num_groups,
            num_nodes,
            num_ranks,
            old_global_expert_indices,
        )
        start_time = time.perf_counter()
        topology = self._resolve_topology(num_nodes, num_ranks)
        topology_hash = topology.topology_hash
        self._prepare_history(expert_load, current, topology)
        assert self._shape_key is not None
        self.load_stats.update(expert_load, self._shape_key)
        current_by_rank = current.reshape(current.shape[0], num_ranks, slots_per_rank)
        if not np.any(expert_load):
            selected_layers: tuple[StairLayerPlan, ...] = ()
            plan = StairRebalancePlan(
                new_mapping=torch.from_numpy(current.copy()).to(dtype=torch.long).contiguous(),
                selected_layers=selected_layers,
                rejected_layers=(),
                budget_usage=StairBudgetUsage(),
                planner_elapsed_ms=(time.perf_counter() - start_time) * 1000,
                plan_id=self._compute_plan_digest(current, selected_layers, topology_hash),
                topology_hash=topology_hash,
            )
            self._pending_plan = plan
            return plan

        risk_load = self.load_stats.risk_load()
        proposed: list[StairLayerPlan] = []
        rejected: list[StairLayerPlan] = []
        evaluated_layers = 0
        current_scores = [
            compute_balance_metrics(
                expert_load[:, layer_id, :],
                current_by_rank[layer_id],
            )
            for layer_id in range(current.shape[0])
        ]
        eligible_layers: list[int] = []
        for layer_id in range(current.shape[0]):
            current_score = current_scores[layer_id]
            if max(current_score.mean_imbalance, current_score.p95_imbalance) < constants.IMBALANCE_THRESHOLD:
                self.load_stats.note_candidate(layer_id, True)
                continue
            if not self._needs_temporal_update(layer_id, current_score.mean_imbalance, num_ranks):
                rejected.append(
                    StairLayerPlan(
                        layer_idx=layer_id,
                        placement=current_by_rank[layer_id].copy(),
                        current_score=current_score,
                        candidate_score=current_score,
                        balance_gain=0.0,
                        utility=0.0,
                        transfer_cost=StairTransferCost(),
                        candidate_kind=StairCandidateKind.CURRENT,
                        source_mode=StairSourceMode.EXECUTOR_DEFAULT,
                        accepted=False,
                        reject_reason=StairRejectReason.TEMPORAL_GATE,
                    )
                )
                continue
            eligible_layers.append(layer_id)

        eligible_layers.sort(
            key=lambda layer_id: (
                -current_scores[layer_id].p95_imbalance,
                -current_scores[layer_id].mean_imbalance,
                layer_id,
            )
        )
        for layer_id in eligible_layers:
            layer_load = expert_load[:, layer_id, :]
            current_score = current_scores[layer_id]
            candidate = build_greedy_layer_candidate(
                risk_load[layer_id],
                current_by_rank[layer_id],
                topology,
            )
            if np.array_equal(candidate, current_by_rank[layer_id]):
                evaluated_layers += 1
                self.load_stats.note_candidate(layer_id, False)
                continue
            layer_plan = self._build_layer_plan(
                layer_id,
                layer_load,
                current_by_rank[layer_id],
                candidate,
                num_ranks,
                topology,
                current_score=current_score,
                check_temporal=False,
            )
            if constants.ENABLE_MINI_FLASH_TREE and self.load_stats.should_search(
                layer_id, layer_plan.current_score, layer_plan.balance_gain
            ):
                layer_plan = self._search_layer_plan(
                    layer_plan,
                    layer_load,
                    risk_load[layer_id],
                    current_by_rank[layer_id],
                    num_ranks,
                    topology,
                )
            evaluated_layers += 1
            self.load_stats.note_candidate(layer_id, layer_plan.accepted)
            (proposed if layer_plan.accepted else rejected).append(layer_plan)

        proposed.sort(key=lambda plan: (-plan.utility, plan.layer_idx))
        result = current_by_rank.copy()
        usage = StairBudgetUsage()
        selected: list[StairLayerPlan] = []
        for layer_plan in proposed:
            if usage.can_add(layer_plan):
                usage.add(layer_plan)
                selected.append(layer_plan)
                result[layer_plan.layer_idx] = layer_plan.placement
                continue
            rejected.append(
                replace(
                    layer_plan,
                    accepted=False,
                    reject_reason=StairRejectReason.CYCLE_BUDGET,
                )
            )

        selected_layers = tuple(selected)
        planner_elapsed_ms = (time.perf_counter() - start_time) * 1000
        plan = StairRebalancePlan(
            new_mapping=torch.from_numpy(result.reshape(current.shape).copy()).to(dtype=torch.long).contiguous(),
            selected_layers=selected_layers,
            rejected_layers=tuple(sorted(rejected, key=lambda item: item.layer_idx)),
            budget_usage=usage,
            planner_elapsed_ms=planner_elapsed_ms,
            plan_id=self._compute_plan_digest(current, selected_layers, topology_hash),
            topology_hash=topology_hash,
        )
        self._pending_plan = plan
        rejected_by_reason = Counter(item.reject_reason.value for item in plan.rejected_layers)
        rejection_summary = ",".join(f"{reason}={rejected_by_reason[reason]}" for reason in sorted(rejected_by_reason))
        rejected_pair_transfers = [item.transfer_cost.max_rank_pair_transfers for item in plan.rejected_layers]
        pair_transfer_range = (
            f"{min(rejected_pair_transfers)}..{max(rejected_pair_transfers)}" if rejected_pair_transfers else "none"
        )
        logger.info(
            "STAIR EPLB plan %s completed in %.3f ms; evaluated %d layers, selected %d layers, "
            "rejected [%s], rejected rank-pair transfers [%s].",
            plan.plan_id,
            plan.planner_elapsed_ms,
            evaluated_layers,
            len(selected_layers),
            rejection_summary,
            pair_transfer_range,
        )
        return plan

    def rebalance_experts(
        self,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a CPU physical-to-logical map through the vLLM policy contract."""
        return self.plan_rebalance(
            weight,
            num_replicas,
            num_groups,
            num_nodes,
            num_ranks,
            old_global_expert_indices,
        ).new_mapping

    def commit_layer(
        self,
        expert_load: torch.Tensor,
        layer_idx: int,
        committed_placement: torch.Tensor,
        num_ranks: int,
        execution_metrics: StairExecutionMetrics | None = None,
    ) -> None:
        """Record hysteresis only after one layer is actually committed."""
        if not self._runtime_history:
            return
        load_array = expert_load.detach().to(dtype=torch.float64).contiguous().numpy()
        placement_array = committed_placement.detach().to(dtype=torch.long).contiguous().numpy()
        if load_array.ndim != 3 or not 0 <= layer_idx < load_array.shape[1]:
            raise ValueError("expert_load must contain the committed layer.")
        if placement_array.ndim != 1 or placement_array.size % num_ranks != 0:
            raise ValueError("committed_placement must be one-dimensional and divisible by num_ranks.")
        placement = placement_array.reshape(num_ranks, -1)
        _validate_layer_placement(placement, load_array.shape[2], num_ranks)
        score = compute_balance_score(load_array[:, layer_idx, :], placement)
        self.average_to_peak_history[layer_idx] = 1.0 / score
        self._expected_layer_placements[layer_idx] = placement_array.copy()
        if execution_metrics is not None:
            self.execution_metrics.append(replace(execution_metrics, committed=True))

    def finish_cycle(self, plan_id: str, committed_layers: tuple[int, ...]) -> None:
        """Close a plan after the worker observes all main-thread acknowledgements."""
        if self._pending_plan is None:
            return
        if self._pending_plan.plan_id != plan_id:
            raise RuntimeError("STAIR attempted to finish a stale rebalance plan.")
        selected = {plan.layer_idx for plan in self._pending_plan.selected_layers}
        if not set(committed_layers).issubset(selected):
            raise RuntimeError("STAIR committed a layer that was not selected by the active plan.")
        self._pending_plan = None

    def abort_cycle(self, plan_id: str) -> None:
        """Discard an uncommitted plan without changing committed history."""
        if self._pending_plan is not None and self._pending_plan.plan_id == plan_id:
            self._pending_plan = None
