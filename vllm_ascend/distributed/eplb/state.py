# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend EPLB state for Statistical Temporal-Aware Incremental Rebalancing (STAIR)."""

import inspect
from dataclasses import fields
from typing import Any

import torch
from torch.distributed import all_reduce
from vllm.distributed import get_ep_group
from vllm.distributed.eplb import eplb_state as _eplb_state
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import logger

from vllm_ascend.distributed.eplb.policy.stair import StairEplbPolicy
from vllm_ascend.distributed.eplb.policy.stair_types import (
    StairExecutionMetrics,
    StairRebalancePlan,
    StairTopology,
)
from vllm_ascend.distributed.eplb.transfer_plan import compute_layer_expert_bytes
from vllm_ascend.ops.fused_moe import eplb as _eplb_ops


def _upstream_from_mapping_accepts_valid_expert_count() -> bool:
    """Return whether the selected vLLM uses the release mapping contract."""
    return "num_valid_physical_experts" in inspect.signature(_eplb_state.EplbState.from_mapping).parameters


def _discover_stair_topology(rank_mapping: dict[int, int] | None = None) -> StairTopology:
    """Collect and validate the actual EPLB rank-to-node relation."""
    coordinator = get_ep_group()
    num_ranks = coordinator.world_size
    if rank_mapping is not None:
        active_ranks = sorted((new_rank, old_rank) for old_rank, new_rank in rank_mapping.items() if new_rank >= 0)
        if [new_rank for new_rank, _ in active_ranks] != list(range(num_ranks)):
            raise ValueError("STAIR rank mapping must describe every active EPLB rank exactly once.")
    if num_ranks == 1:
        return StairTopology.from_rank_to_node((0,))

    same_node = []
    for source_rank in range(num_ranks):
        flags = tuple(bool(value) for value in in_the_same_node_as(coordinator.cpu_group, source_rank))
        if len(flags) != num_ranks:
            raise RuntimeError("STAIR topology discovery returned an invalid rank count.")
        same_node.append(flags)
    for rank in range(num_ranks):
        if not same_node[rank][rank]:
            raise RuntimeError("STAIR topology relation must be reflexive.")
        for peer in range(num_ranks):
            if same_node[rank][peer] != same_node[peer][rank]:
                raise RuntimeError("STAIR topology relation must be symmetric.")
            if same_node[rank][peer]:
                for other in range(num_ranks):
                    if same_node[peer][other] and not same_node[rank][other]:
                        raise RuntimeError("STAIR topology relation must be transitive.")

    node_by_rank = [-1] * num_ranks
    next_node_id = 0
    for rank in range(num_ranks):
        if node_by_rank[rank] >= 0:
            continue
        for peer in range(num_ranks):
            if same_node[rank][peer]:
                node_by_rank[peer] = next_node_id
        next_node_id += 1
    return StairTopology.from_rank_to_node(tuple(node_by_rank))


def _compute_model_layer_expert_bytes(model: Any) -> tuple[int, ...] | None:
    """Compute transfer bytes when the model exposes real expert weight views."""
    expert_weights = getattr(model, "expert_weights", None)
    if expert_weights is None:
        return None
    return compute_layer_expert_bytes(expert_weights)


class AscendEplbLayerState(_eplb_state.EplbLayerState):
    """EPLB layer state with a graph-stable replica routing table."""

    def __init__(self) -> None:
        super().__init__()
        self.expert_replica_routing_table: torch.Tensor | None = None

    @classmethod
    def from_upstream(cls, state: _eplb_state.EplbLayerState) -> "AscendEplbLayerState":
        ascend_state = cls()
        for field in fields(_eplb_state.EplbLayerState):
            setattr(ascend_state, field.name, getattr(state, field.name))
        return ascend_state

    def set_layer_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        super().set_layer_state(
            moe_layer_idx,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )
        self.refresh_expert_replica_routing_table()

    def refresh_expert_replica_routing_table(self) -> None:
        logical_to_physical_map = self.logical_to_physical_map
        logical_replica_count = self.logical_replica_count
        if logical_to_physical_map is None or logical_replica_count is None:
            raise RuntimeError("Cannot build the replica routing table before EPLB layer state is initialized.")

        new_routing_table = _eplb_ops.build_expert_replica_routing_table(
            logical_to_physical_map,
            logical_replica_count,
            get_ep_group().rank_in_group,
        )
        if (
            self.expert_replica_routing_table is not None
            and self.expert_replica_routing_table.shape == new_routing_table.shape
        ):
            self.expert_replica_routing_table.copy_(new_routing_table, non_blocking=True)
        else:
            self.expert_replica_routing_table = new_routing_table


def refresh_model_routing_tables(model_state: Any, layer_idx: int | None = None) -> None:
    """Refresh all routing tables, or one table after an async map commit."""
    layers = list(model_state.model.moe_layers)
    selected_layers = enumerate(layers) if layer_idx is None else ((layer_idx, layers[layer_idx]),)
    for _, layer in selected_layers:
        layer_state = layer.eplb_state
        if isinstance(layer_state, AscendEplbLayerState):
            layer_state.refresh_expert_replica_routing_table()


class AscendEplbState(_eplb_state.EplbState):
    """STAIR state, load-window preservation, and Ascend async lifecycle."""

    policy: Any
    cuda_device_index: int | None
    async_worker: Any

    def __init__(self, parallel_config, device: torch.device) -> None:
        super().__init__(parallel_config, device)
        # Upstream's profile path still calls the state-level policy. Runtime
        # hysteresis lives on each model state so main and draft models cannot
        # overwrite one another's layer history.
        self._stair_topology: StairTopology | None
        try:
            self._stair_topology = _discover_stair_topology()
        except Exception as error:
            try:
                num_ranks = get_ep_group().world_size
            except Exception:
                num_ranks = 0
            self._stair_topology = StairTopology.flat_fallback(num_ranks) if num_ranks else None
            logger.warning_once(
                "STAIR could not discover a reliable node topology and will price all remote transfers "
                "conservatively: %s",
                error,
            )
        self._profile_policy = StairEplbPolicy(self._stair_topology)
        self.policy = self._profile_policy
        self._has_fresh_recorded_load = False
        self._preserve_expert_load_time_series = False
        self._planner_client: Any = None
        self._use_process_planner = False
        self._planner_topology_epoch = 0
        if self.cuda_device_index is None:
            self.cuda_device_index = torch.accelerator.current_device_index()

    def add_model(self, model, model_config) -> None:
        if getattr(self, "_use_process_planner", False):
            raise RuntimeError("Models cannot be registered after the STAIR planner process has started.")
        super().add_model(model, model_config)
        self.is_async = True
        model_state = self.model_states[model_config.compute_hash()]
        model_state_any: Any = model_state
        layer_expert_bytes = _compute_model_layer_expert_bytes(model)
        model_state_any._ascend_eplb_layer_expert_bytes = layer_expert_bytes
        model_state_any._ascend_eplb_policy = StairEplbPolicy(
            getattr(self, "_stair_topology", None),
            layer_expert_bytes,
        )
        model_state_any._ascend_eplb_state = self
        model_state_any._ascend_eplb_model_id = model_config.compute_hash()
        model_state_any._ascend_eplb_mapping_version = 0
        model_state_any._ascend_eplb_active_plan = None
        model_state_any._ascend_eplb_policy_load = None
        model_state_any._ascend_eplb_pending_execution_metrics = None
        model_state_any._ascend_eplb_committed_layer_ids = []
        model_state_any._ascend_eplb_execution_metrics = []
        # super().add_model() replaces the state-level policy with the
        # configured upstream policy. Restore the STAIR profile policy.
        self.policy = self._profile_policy
        logger.info("Selected Ascend EPLB placement policy: Statistical Temporal-Aware Incremental Rebalancing (STAIR)")

    def start_async_loop(
        self,
        rank_mapping: dict[int, int] | None = None,
        is_profile: bool = False,
    ) -> None:
        if self.async_worker is not None:
            return
        if rank_mapping is not None:
            topology = _discover_stair_topology(rank_mapping)
            if topology != self._stair_topology:
                self._stair_topology = topology
                self._profile_policy.configure_runtime(None, topology=topology)
                self._planner_topology_epoch += 1
                for model_state in self.model_states.values():
                    model_state._ascend_eplb_policy.configure_runtime(
                        model_state._ascend_eplb_layer_expert_bytes,
                        topology=topology,
                    )
        if not is_profile:
            self._start_process_planner()
        from vllm_ascend.distributed.eplb.async_worker import start_async_worker

        self.async_worker = start_async_worker(self, is_profile=is_profile)

    def _start_process_planner(self) -> None:
        """Create the one persistent planner owned by EP rank zero."""
        if getattr(self, "_use_process_planner", False):
            return
        topology = self._stair_topology
        if topology is None:
            raise RuntimeError("STAIR process planner requires a validated EPLB topology.")
        from vllm.distributed.parallel_state import get_eplb_group

        ep_rank = get_eplb_group().device_group.rank()
        self._use_process_planner = True
        if ep_rank != 0:
            return
        from vllm_ascend.distributed.eplb.planner_client import StairPlannerClient
        from vllm_ascend.distributed.eplb.planner_protocol import PlannerModelRegistration

        registrations = []
        for model_id, model_state in self.model_states.items():
            model = model_state.model
            registrations.append(
                PlannerModelRegistration(
                    model_id=model_id,
                    load_shape=(
                        self.expert_load_window_size,
                        model.num_moe_layers,
                        model.num_logical_experts,
                    ),
                    mapping_shape=(model.num_moe_layers, model.num_physical_experts),
                    num_replicas=model.num_physical_experts,
                    num_groups=model.num_expert_groups,
                    num_nodes=topology.num_nodes,
                    num_ranks=len(topology.rank_to_node),
                    topology=topology,
                    layer_expert_bytes=model_state._ascend_eplb_layer_expert_bytes,
                )
            )
        self._planner_client = StairPlannerClient(tuple(registrations))

    @property
    def planner_pid(self) -> int | None:
        client = self._planner_client
        return None if client is None else client.pid

    def mark_planner_affinity_ready(self, planner_cpus: tuple[int, ...]) -> None:
        client = self._planner_client
        if client is not None:
            client.mark_affinity_ready(planner_cpus)

    def plan_with_process(
        self,
        model_state: Any,
        load_window: torch.Tensor,
        mapping: torch.Tensor,
    ) -> StairRebalancePlan:
        client = self._planner_client
        if client is None:
            raise RuntimeError("Only EPLB rank zero owns the STAIR planner process.")
        return client.plan(
            model_state._ascend_eplb_model_id,
            load_window,
            mapping,
            mapping_version=model_state._ascend_eplb_mapping_version,
            topology_epoch=self._planner_topology_epoch,
        )

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        if not is_dummy and not is_profile and self._should_record_current_step(log_stats=log_stats):
            self._has_fresh_recorded_load = True
        super().step(is_dummy=is_dummy, is_profile=is_profile, log_stats=log_stats)

    def _has_global_fresh_recorded_load(self) -> bool:
        """Synchronize whether any EP rank recorded load since rearranging."""
        ep_group = get_ep_group()
        cpu_group = getattr(ep_group, "cpu_group", None)
        if cpu_group is not None:
            if cpu_group.size() <= 1:
                return self._has_fresh_recorded_load
            flag = torch.tensor((self._has_fresh_recorded_load,), dtype=torch.int32, device="cpu")
            all_reduce(flag, group=cpu_group)
            return bool(flag.item())

        device_group = ep_group.device_group
        if device_group.size() <= 1:
            return self._has_fresh_recorded_load
        flag = torch.tensor((self._has_fresh_recorded_load,), dtype=torch.int32, device=self.device)
        all_reduce(flag, group=device_group)
        return bool(flag.item())

    def _build_logical_expert_load_time_series(self) -> list[torch.Tensor]:
        logical_load_windows = []
        for model_state in self.model_states.values():
            physical_load_window = model_state.expert_load_window
            physical_to_logical_map = model_state.physical_to_logical_map
            invalid_idx = model_state.model.num_logical_experts
            logical_load_window = torch.zeros(
                physical_load_window.shape[0],
                model_state.model.num_moe_layers,
                invalid_idx + 1,
                dtype=physical_load_window.dtype,
                device=physical_load_window.device,
            )
            logical_load_window.scatter_add_(
                dim=-1,
                index=physical_to_logical_map.masked_fill(
                    physical_to_logical_map < 0,
                    invalid_idx,
                )
                .unsqueeze(0)
                .expand_as(physical_load_window)
                .long(),
                src=physical_load_window,
            )
            logical_load_windows.append(logical_load_window[..., :-1])
        return logical_load_windows

    def _allreduce_list(self, tensor_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Preserve the STAIR window axis across the upstream collective."""
        if not self._preserve_expert_load_time_series:
            return super()._allreduce_list(tensor_list)
        temporal_load_windows = (
            tensor_list
            if all(tensor.dim() == 3 for tensor in tensor_list)
            else self._build_logical_expert_load_time_series()
        )
        shapes = [tensor.shape for tensor in temporal_load_windows]
        flattened = [tensor.reshape(-1, tensor.shape[-1]).contiguous() for tensor in temporal_load_windows]
        reduced = super()._allreduce_list(flattened)
        return [tensor.reshape(shape) for tensor, shape in zip(reduced, shapes)]

    def rearrange(
        self,
        is_profile: bool = False,
        rank_mapping: dict[int, int] | None = None,
    ) -> torch.Tensor | None:
        should_gate = (
            hasattr(self, "_has_fresh_recorded_load")
            and not is_profile
            and rank_mapping is None
            and not self.parallel_config.enable_elastic_ep
        )
        if should_gate and not self._has_global_fresh_recorded_load():
            return None

        self._preserve_expert_load_time_series = True
        try:
            result = super().rearrange(is_profile=is_profile, rank_mapping=rank_mapping)
        finally:
            self._preserve_expert_load_time_series = False
        if not is_profile:
            self._has_fresh_recorded_load = False
        return result

    def commit_policy_layer(self, model_state: Any, layer_idx: int) -> None:
        """Commit STAIR hysteresis after upstream commits weights and maps."""
        load_window = getattr(model_state, "_ascend_eplb_policy_load", None)
        if load_window is None or model_state.eplb_stats is None:
            return
        committed_mapping = model_state.physical_to_logical_map[layer_idx].cpu()
        if getattr(self, "_use_process_planner", False):
            active_plan: StairRebalancePlan | None = model_state._ascend_eplb_active_plan
            if active_plan is None:
                raise RuntimeError("STAIR observed a mapping commit without an active process plan.")
            selected_layers = {item.layer_idx for item in active_plan.selected_layers}
            if layer_idx not in selected_layers:
                raise RuntimeError(f"STAIR observed an unplanned mapping commit for layer {layer_idx}.")
            if not torch.equal(committed_mapping.to(dtype=torch.long), active_plan.new_mapping[layer_idx]):
                raise RuntimeError(f"STAIR committed mapping does not match the active plan for layer {layer_idx}.")
        else:
            model_state._ascend_eplb_policy.commit_layer(
                load_window,
                layer_idx,
                committed_mapping,
                model_state.eplb_stats.num_gpus,
                getattr(model_state, "_ascend_eplb_pending_execution_metrics", None),
            )
        committed_layer_ids = getattr(model_state, "_ascend_eplb_committed_layer_ids", None)
        if committed_layer_ids is not None:
            committed_layer_ids.append(layer_idx)
        execution_metrics: StairExecutionMetrics | None = getattr(
            model_state,
            "_ascend_eplb_pending_execution_metrics",
            None,
        )
        if execution_metrics is not None:
            model_state._ascend_eplb_execution_metrics.append(execution_metrics)
        model_state._ascend_eplb_mapping_version = getattr(model_state, "_ascend_eplb_mapping_version", 0) + 1

    def finish_policy_cycle(
        self,
        model_state: Any,
        plan: StairRebalancePlan,
        *,
        aborted: bool = False,
    ) -> None:
        """Finalize process-owned history after main-thread commit acknowledgements."""
        committed_layer_ids = tuple(getattr(model_state, "_ascend_eplb_committed_layer_ids", ()))
        execution_metrics = tuple(getattr(model_state, "_ascend_eplb_execution_metrics", ()))
        if getattr(self, "_use_process_planner", False):
            if self._planner_client is not None:
                self._planner_client.finalize(
                    model_state._ascend_eplb_model_id,
                    mapping_version=model_state._ascend_eplb_mapping_version,
                    topology_epoch=self._planner_topology_epoch,
                    committed_layer_ids=committed_layer_ids,
                    execution_metrics=execution_metrics,
                    aborted=aborted,
                )
        elif aborted:
            model_state._ascend_eplb_policy.abort_cycle(plan.plan_id)
        else:
            model_state._ascend_eplb_policy.finish_cycle(plan.plan_id, committed_layer_ids)
        model_state._ascend_eplb_active_plan = None

    def close(self) -> None:
        client = self._planner_client
        if client is not None:
            client.close()
            self._planner_client = None

    @classmethod
    def from_mapping(
        cls,
        model,
        model_config,
        device: torch.device,
        parallel_config,
        expanded_physical_to_logical: torch.Tensor,
        num_valid_physical_experts: int | None = None,
    ) -> "AscendEplbState":
        from_mapping_kwargs: dict[str, Any] = dict(
            model=model,
            model_config=model_config,
            device=device,
            parallel_config=parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
        )
        if _upstream_from_mapping_accepts_valid_expert_count():
            if num_valid_physical_experts is None:
                raise TypeError("num_valid_physical_experts is required by the selected vLLM release mapping contract")
            from_mapping_kwargs["num_valid_physical_experts"] = num_valid_physical_experts
        state = super().from_mapping(**from_mapping_kwargs)
        for model_state in state.model_states.values():
            refresh_model_routing_tables(model_state)
        return state
