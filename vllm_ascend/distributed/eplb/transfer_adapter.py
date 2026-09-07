# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Execute immutable STAIR source plans without re-deriving senders."""

from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
import torch
from vllm.distributed.eplb.rebalance_execute import TransferMetadata

from vllm_ascend.distributed.eplb.policy.stair_types import LayerPlan, RankTopology


def validate_transfer_capability(communicator: object) -> None:
    for method in ("add_send", "add_recv", "execute", "set_stream"):
        if not callable(getattr(communicator, method, None)):
            raise RuntimeError(f"STAIR communicator lacks explicit {method} capability")


def stage_layer(
    layer_plan: LayerPlan,
    expert_weights: Sequence[torch.Tensor],
    expert_buffers: Sequence[torch.Tensor],
    communicator: object,
    topology: RankTopology,
    *,
    ep_rank: int,
    cuda_stream: torch.cuda.Stream | None,
) -> TransferMetadata:
    """Stage one local rank using only the planner-selected source rows."""
    validate_transfer_capability(communicator)
    ranks, local_experts = layer_plan.new_placement.shape
    if topology.num_ranks != ranks or len(expert_weights) != len(expert_buffers):
        raise ValueError("STAIR transfer tensor schema or topology changed")
    if any(weight.shape[0] != local_experts for weight in expert_weights) or any(
        buffer.shape != weight.shape for buffer, weight in zip(expert_buffers, expert_weights)
    ):
        raise ValueError("STAIR transfer tensors do not match the registered expert schema")

    is_unchanged = np.zeros(local_experts, dtype=np.bool_)
    is_received_locally = np.zeros(local_experts, dtype=np.bool_)
    recv_primary_mask = np.zeros(local_experts, dtype=np.bool_)
    recv_expert_ids = np.full(local_experts, -1, dtype=np.int64)
    recv_dst_rows = np.full(local_experts, -1, dtype=np.int32)
    recv_count = 0
    stream_context = torch.cuda.stream(cuda_stream) if cuda_stream is not None else nullcontext()

    old_flat = layer_plan.old_placement.reshape(-1)
    set_context = getattr(communicator, "set_transfer_context", None)
    if callable(set_context):
        set_context(old_flat, layer_plan.layer_idx)
    operations = []
    for dst in range(ranks):
        for dst_slot, expert in enumerate(layer_plan.new_placement[dst]):
            src = int(layer_plan.source_rank[dst, dst_slot])
            src_slot = int(layer_plan.source_slot[dst, dst_slot])
            operations.append((layer_plan.layer_idx, dst, dst_slot, int(expert), src, src_slot))
    operations.sort()

    with stream_context:
        for _, dst, dst_slot, expert, src, src_slot in operations:
            if src == dst:
                if ep_rank == dst:
                    is_received_locally[dst_slot] = True
                    is_unchanged[dst_slot] = src_slot == dst_slot
                    if src_slot != dst_slot:
                        for weight, buffer in zip(expert_weights, expert_buffers):
                            buffer[dst_slot].copy_(weight[src_slot], non_blocking=True)
                continue
            if ep_rank == src:
                tensors = [weight[src_slot] for weight in expert_weights]
                communicator.add_send(tensors, topology.eplb_rank_by_ep_rank[dst], expert)
            if ep_rank == dst:
                tensors = [buffer[dst_slot] for buffer in expert_buffers]
                communicator.add_recv(tensors, topology.eplb_rank_by_ep_rank[src], expert)
                recv_primary_mask[dst_slot] = True
                recv_expert_ids[recv_count] = expert
                recv_dst_rows[recv_count] = dst_slot
                recv_count += 1
    communicator.execute()
    return TransferMetadata(
        is_unchanged=is_unchanged,
        is_received_locally=is_received_locally,
        recv_primary_mask=recv_primary_mask,
        recv_count=recv_count,
        recv_expert_ids=recv_expert_ids,
        recv_dst_rows=recv_dst_rows,
    )
