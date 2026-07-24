# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.utils.torch_utils import direct_register_custom_op

EPLB_LOOKUP_NUM_ROWS = 1024


def build_physical_id_lookup(
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    ep_rank: int,
) -> torch.Tensor:
    """Build a rank-aware logical-to-global-physical expert lookup."""
    if logical_to_physical_map.ndim != 2:
        raise ValueError("logical_to_physical_map must be a 2D tensor.")
    if logical_replica_count.ndim != 1:
        raise ValueError("logical_replica_count must be a 1D tensor.")
    if logical_to_physical_map.shape[0] != logical_replica_count.shape[0]:
        raise ValueError("Logical expert dimensions must match.")

    num_logical_experts = logical_replica_count.shape[0]
    device = logical_to_physical_map.device
    table_rows = torch.arange(EPLB_LOOKUP_NUM_ROWS, dtype=torch.int64, device=device)[:, None]
    logical_expert_ids = torch.arange(num_logical_experts, dtype=torch.int64, device=device)[None, :]
    replica_count = logical_replica_count.to(torch.int64).clamp_min(1)[None, :]
    replica_indices = (table_rows + ep_rank + logical_expert_ids) % replica_count
    lookup = logical_to_physical_map.gather(1, replica_indices.T).T
    return lookup.to(torch.int32).contiguous()


def map_to_physical(
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    ep_rank: int,
) -> torch.Tensor:
    """Map logical expert IDs to global physical IDs with periodic placement."""
    if topk_ids.numel() == 0:
        return topk_ids
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must be a 2D tensor.")

    logical_ids = topk_ids.to(torch.int64) if topk_ids.device.type == "cpu" else topk_ids
    valid = (logical_ids >= 0) & (logical_ids < logical_replica_count.shape[0])
    safe_logical_ids = torch.where(valid, logical_ids, 0)
    replica_count = logical_replica_count[safe_logical_ids].to(torch.int64).clamp_min(1)
    token_rows = torch.arange(
        topk_ids.shape[0],
        dtype=torch.int64,
        device=topk_ids.device,
    )
    token_rows = torch.remainder(token_rows, EPLB_LOOKUP_NUM_ROWS)
    replica_indices = (token_rows[:, None] + ep_rank + safe_logical_ids) % replica_count
    physical_ids = logical_to_physical_map[
        safe_logical_ids,
        replica_indices,
    ].to(topk_ids.dtype)
    return torch.where(valid, physical_ids, -1)


def normalize_local_expert_load(
    expert_tokens: torch.Tensor,
    group_list_type: int,
    num_local_physical_experts: int,
) -> torch.Tensor:
    """Convert device-private GMM counts to normalized local expert load."""
    if expert_tokens.numel() < num_local_physical_experts:
        raise ValueError("expert_tokens has fewer entries than the number of local physical experts.")

    local_load = expert_tokens[:num_local_physical_experts]
    if group_list_type != 1:
        local_load = torch.cat((local_load[:1], local_load[1:] - local_load[:-1]))
    return local_load


def _map_to_physical_fake(
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    ep_rank: int,
) -> torch.Tensor:
    del logical_to_physical_map, logical_replica_count, ep_rank
    return torch.empty_like(topk_ids)


direct_register_custom_op(
    op_name="ascend_eplb_map_to_physical",
    op_func=map_to_physical,
    fake_impl=_map_to_physical_fake,
    dispatch_key="PrivateUse1",
)
