# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Model Runner V2 adapters for Ascend-owned EPLB policies."""

import time

import torch
from vllm.distributed.eplb.policy.abstract import AbstractEplbPolicy
from vllm.logger import logger

from vllm_ascend.eplb.core.policy.policy_swift_balancer import SwiftBalanceEplb


def _expand_logical_load_to_slots(
    logical_load: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
) -> torch.Tensor:
    """Convert V2 logical load to the per-replica load expected by Swift."""
    if logical_load.ndim != 2:
        raise ValueError(f"logical_load must be two-dimensional, got shape {tuple(logical_load.shape)}.")
    if physical_to_logical_map.ndim != 2:
        raise ValueError(
            f"physical_to_logical_map must be two-dimensional, got shape {tuple(physical_to_logical_map.shape)}."
        )
    if logical_load.device.type != "cpu" or physical_to_logical_map.device.type != "cpu":
        raise ValueError("Swift EPLB policy inputs must be CPU tensors.")
    if logical_load.shape[0] != physical_to_logical_map.shape[0]:
        raise ValueError(
            "logical_load and physical_to_logical_map must have the same number of layers, "
            f"got {logical_load.shape[0]} and {physical_to_logical_map.shape[0]}."
        )

    placement = physical_to_logical_map.detach().to(dtype=torch.long).clone()
    num_logical_experts = logical_load.shape[1]
    if bool((placement < 0).any()) or bool((placement >= num_logical_experts).any()):
        raise ValueError("physical_to_logical_map contains an invalid logical expert index.")

    replica_counts = torch.zeros(
        logical_load.shape,
        dtype=torch.long,
        device="cpu",
    )
    replica_counts.scatter_add_(
        1,
        placement,
        torch.ones_like(placement, dtype=torch.long),
    )
    if bool((replica_counts == 0).any()):
        raise ValueError("Every logical expert must have at least one physical replica.")

    logical_load_fp64 = logical_load.detach().to(dtype=torch.float64).clone()
    slot_replica_counts = replica_counts.gather(1, placement)
    return logical_load_fp64.gather(1, placement) / slot_replica_counts


class SwiftEplbPolicyAdapter(AbstractEplbPolicy):
    """Adapt the existing Model Runner V1 Swift policy to the V2 contract."""

    @classmethod
    def rebalance_experts(
        cls,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del num_groups
        if old_global_expert_indices is None:
            raise ValueError("Swift EPLB requires the current physical-to-logical map.")
        if num_replicas <= 0 or num_ranks <= 0 or num_nodes <= 0:
            raise ValueError("num_replicas, num_ranks, and num_nodes must be positive.")
        if num_replicas % num_ranks != 0:
            raise ValueError(f"num_replicas ({num_replicas}) must be divisible by num_ranks ({num_ranks}).")
        if num_ranks % num_nodes != 0:
            raise ValueError(f"num_ranks ({num_ranks}) must be divisible by num_nodes ({num_nodes}).")
        if old_global_expert_indices.shape != (weight.shape[0], num_replicas):
            raise ValueError(
                "Current placement shape must be [layers, num_replicas], got "
                f"{tuple(old_global_expert_indices.shape)} for weight shape {tuple(weight.shape)} "
                f"and num_replicas={num_replicas}."
            )

        slots_per_rank = num_replicas // num_ranks
        old_placement = old_global_expert_indices.detach().to(dtype=torch.long).clone()
        slot_load = _expand_logical_load_to_slots(weight, old_placement)
        legacy_placement = old_placement.reshape(weight.shape[0], num_ranks, slots_per_rank)
        legacy_slot_load = slot_load.reshape(weight.shape[0], num_ranks, slots_per_rank)

        start_time = time.perf_counter()
        policy = SwiftBalanceEplb()
        policy.num_die_per_host = num_ranks // num_nodes
        _, _, new_placement = policy.rebalance_experts(
            legacy_placement,
            legacy_slot_load,
            is_node_redundant=False,
        )
        compute_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Swift EPLB policy computation completed in %.3f ms.", compute_ms)

        return (
            torch.as_tensor(new_placement, dtype=torch.long, device="cpu")
            .reshape(weight.shape[0], num_replicas)
            .contiguous()
        )
