# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Spike: Triton fused gating/map/record kernels for STAIR EPLB load recording.

Two kernels are prototyped here (PR1 design doc, route experiment):

- ``gating_map_record``: full fusion in ONE launch. Sigmoid/bias gating,
  grouped top-k, replica-table mapping, and local-expert load recording.
- ``map_record``: map+record only, consumes the CANN ``moe_gating_top_k``
  output. Two launches together with the unmodified CANN gating op.

Semantics are lifted from
``csrc/moe/moe_gating_top_k/op_kernel/moe_gating_top_k_e_k_fullload.h``
(sigmoid path, ``group_select_mode=1``):

  score = sigmoid(x); key = score + bias
  group_score = sum of the top-2 keys inside each GROUP_SIZE group
  select the top K_GROUP groups by group_score, then the top K experts
  among them by key; weight = score / (sum(score) + eps) * scaling

Weights intentionally use pre-bias scores, matching the CANN kernel's
gather from ``xSigmoidTensor``. Ties are broken by the lowest expert
index; the CANN ``Sort32`` tie order is unspecified, so the parity
harness measures the mismatch rate instead of assuming it is zero.
"""

import torch
from vllm.triton_utils import tl, triton

_BIG = tl.constexpr(1 << 30)
_NEG_INF = tl.constexpr(float("-inf"))


# ---------------------------------------------------------------------------
# Verbatim copies of the current-route Triton kernels and the routing-table
# builder (vllm_ascend/ops/triton/eplb.py, vllm_ascend/ops/fused_moe/eplb.py).
# The spike keeps them local so it can import without the full vllm_ascend
# package; the full-fusion kernel's map segment below must stay identical to
# _map_to_physical_kernel for the bit-exact parity gate.
# ---------------------------------------------------------------------------

@triton.jit
def _map_to_physical_kernel(
    topk_ids_ptr,
    routing_table_ptr,
    physical_ids_ptr,
    num_logical_experts,
    numel,
    topk,
    routing_table_rows,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    logical_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=-1).to(tl.int64)
    valid_logical_id = (logical_id >= 0) & (logical_id < num_logical_experts)
    safe_logical_id = tl.where(valid_logical_id, logical_id, 0)

    token_idx = offsets // topk
    routing_row = token_idx % routing_table_rows
    routing_index = routing_row * num_logical_experts + safe_logical_id
    physical_id = tl.load(
        routing_table_ptr + routing_index,
        mask=mask & valid_logical_id,
        other=-1,
    )
    tl.store(physical_ids_ptr + offsets, physical_id, mask=mask)


@triton.jit
def _record_expert_tokens_kernel(
    expert_tokens_ptr,
    expert_load_ptr,
    record_enabled_ptr,
    num_local_experts,
    local_expert_start,
    group_list_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    if tl.load(record_enabled_ptr) != 0:
        mask = offsets < num_local_experts
        current = tl.load(expert_tokens_ptr + offsets, mask=mask, other=0)
        if group_list_type == 1:
            local_load = current
        else:
            previous_offset = tl.maximum(offsets - 1, 0)
            previous = tl.load(expert_tokens_ptr + previous_offset, mask=mask & (offsets > 0), other=0)
            local_load = current - previous
        load_offsets = local_expert_start + offsets
        previous_load = tl.load(expert_load_ptr + load_offsets, mask=mask, other=0)
        tl.store(expert_load_ptr + load_offsets, previous_load + local_load, mask=mask)


def map_to_physical_triton(topk_ids: torch.Tensor, expert_replica_routing_table: torch.Tensor) -> torch.Tensor:
    if topk_ids.numel() == 0:
        return topk_ids
    physical_ids = torch.empty_like(topk_ids)
    numel = topk_ids.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    _map_to_physical_kernel[grid](
        topk_ids,
        expert_replica_routing_table,
        physical_ids,
        expert_replica_routing_table.shape[1],
        numel,
        topk_ids.shape[1],
        expert_replica_routing_table.shape[0],
        BLOCK_SIZE=256,
    )
    return physical_ids


def record_expert_tokens_triton(
    expert_tokens: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
    group_list_type: int,
    local_expert_start: int,
) -> None:
    num_local_experts = expert_tokens.numel()
    if num_local_experts == 0:
        return
    _record_expert_tokens_kernel[(1,)](
        expert_tokens,
        expert_load_view,
        record_enabled,
        num_local_experts,
        local_expert_start,
        group_list_type=group_list_type,
        BLOCK_SIZE=triton.next_power_of_2(num_local_experts),
    )


EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS = 1024


def build_expert_replica_routing_table(
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    ep_rank: int,
) -> torch.Tensor:
    num_logical_experts = logical_replica_count.shape[0]
    device = logical_to_physical_map.device
    table_rows = torch.arange(
        EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS, dtype=torch.int64, device=device
    )[:, None]
    logical_expert_ids = torch.arange(num_logical_experts, dtype=torch.int64, device=device)[None, :]
    replica_count = logical_replica_count.to(torch.int64).clamp_min(1)[None, :]
    replica_indices = (table_rows + ep_rank + logical_expert_ids) % replica_count
    routing_table = logical_to_physical_map.gather(1, replica_indices.T).T
    return routing_table.to(torch.int32).contiguous()


@triton.jit
def gating_map_record_kernel(
    x_ptr,  # [num_tokens, num_experts] router logits (fp32/bf16, cast in-kernel)
    bias_ptr,  # [num_experts] e_score_correction_bias
    table_ptr,  # [TABLE_ROWS, num_experts] int32 replica routing table
    record_enabled_ptr,  # 0-D device int, non-zero enables recording
    num_valid_ptr,  # 0-D device int, tokens beyond this are padding
    load_ptr,  # [num_physical] int32 expert load view, atomically accumulated
    y_ptr,  # [num_tokens, K] fp32 topk weights
    ids_ptr,  # [num_tokens, K] int32 physical expert ids
    num_tokens,
    num_experts,
    local_expert_start,
    eps,
    scaling,
    num_physical,
    K: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    K_GROUP: tl.constexpr,
    LOCAL_COUNT: tl.constexpr,
    LOCAL_COUNT_POW2: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    E_ALIGN: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
):
    tok0 = tl.program_id(0) * TOKENS_PER_PROGRAM
    trows = tok0 + tl.arange(0, TOKENS_PER_PROGRAM)  # (TS,)
    tmask = trows < num_tokens
    offs = tl.arange(0, E_ALIGN)
    emask = offs < num_experts
    lmask = tmask[:, None] & emask[None, :]

    x = tl.load(x_ptr + trows[:, None] * num_experts + offs[None, :], mask=lmask, other=0.0).to(tl.float32)
    score = tl.sigmoid(x)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs, mask=emask, other=0.0).to(tl.float32)
        key = tl.where(lmask, score + bias[None, :], _NEG_INF)
    else:
        key = tl.where(lmask, score, _NEG_INF)

    g_idx = offs // GROUP_SIZE
    garange = tl.arange(0, GROUP_COUNT)

    # Group score: sum of the top-2 keys per group. The group-max element is
    # removed by index (not value) so duplicated values are handled; axis=1
    # reductions batch TOKENS_PER_PROGRAM tokens per block op.
    gs = tl.zeros((TOKENS_PER_PROGRAM, GROUP_COUNT), dtype=tl.float32)
    for g in tl.static_range(GROUP_COUNT):
        gm = g_idx[None, :] == g
        k1 = tl.max(tl.where(gm, key, _NEG_INF), axis=1)
        i1 = tl.min(tl.where(gm & (key == k1[:, None]), offs[None, :], _BIG), axis=1)
        k2 = tl.max(tl.where(gm & (offs[None, :] != i1[:, None]), key, _NEG_INF), axis=1)
        gs = tl.where(garange[None, :] == g, k1[:, None] + k2[:, None], gs)

    # Select the top K_GROUP groups per token; record the winning expert mask.
    sel = tl.zeros((TOKENS_PER_PROGRAM, E_ALIGN), dtype=tl.int32)
    for _ in tl.static_range(K_GROUP):
        gv = tl.max(gs, axis=1)
        gi = tl.min(tl.where(gs == gv[:, None], garange[None, :], GROUP_COUNT), axis=1)
        sel = sel | (g_idx[None, :] == gi[:, None]).to(tl.int32)
        gs = tl.where(garange[None, :] == gi[:, None], _NEG_INF, gs)

    # Top-K experts per token among the selected groups, then map + record.
    cand = tl.where((sel > 0) & lmask, key, _NEG_INF)
    karange = tl.arange(0, K)
    lc = tl.arange(0, LOCAL_COUNT_POW2)
    hits = tl.zeros((TOKENS_PER_PROGRAM, LOCAL_COUNT_POW2), dtype=tl.int32)
    sc = tl.zeros((TOKENS_PER_PROGRAM, K), dtype=tl.float32)
    rec_on = tl.load(record_enabled_ptr) != 0
    tok_valid = trows < tl.load(num_valid_ptr)
    active = rec_on & tok_valid & tmask

    for j in tl.static_range(K):
        v = tl.max(cand, axis=1)
        e = tl.min(tl.where(cand == v[:, None], offs[None, :], _BIG), axis=1)
        s = tl.sum(tl.where(offs[None, :] == e[:, None], score, 0.0), axis=1)
        phys = tl.load(table_ptr + (trows % TABLE_ROWS) * num_experts + e, mask=tmask, other=-1)
        tl.store(ids_ptr + trows * K + j, phys, mask=tmask)
        local = phys - local_expert_start
        hits += ((lc[None, :] == local[:, None]) & active[:, None]).to(tl.int32)
        cand = tl.where(offs[None, :] == e[:, None], _NEG_INF, cand)
        sc = tl.where(karange[None, :] == j, s[:, None], sc)

    ysum = tl.sum(sc, axis=1)
    tl.store(
        y_ptr + trows[:, None] * K + karange[None, :],
        sc / (ysum[:, None] + eps) * scaling,
        mask=tmask[:, None],
    )

    # One vector atomic per program instead of K scalar atomics per token.
    lc_ok = (lc < LOCAL_COUNT) & (lc < num_physical - local_expert_start)
    row_lc = tl.broadcast_to(lc[None, :], (TOKENS_PER_PROGRAM, LOCAL_COUNT_POW2))
    tl.atomic_add(
        load_ptr + local_expert_start + row_lc,
        hits,
        mask=tl.broadcast_to(lc_ok[None, :], (TOKENS_PER_PROGRAM, LOCAL_COUNT_POW2)) & active[:, None],
    )


@triton.jit
def map_record_kernel(
    topk_ids_ptr,  # [numel] logical expert ids from the CANN gating op
    table_ptr,
    physical_ptr,  # [numel] mapped physical ids
    record_enabled_ptr,
    num_valid_ptr,
    load_ptr,
    num_experts,
    numel,
    topk,
    local_expert_start,
    LOCAL_COUNT: tl.constexpr,
    LOCAL_COUNT_POW2: tl.constexpr,
    LOCAL_CHUNK: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    logical = tl.load(topk_ids_ptr + offs, mask=mask, other=-1).to(tl.int64)
    valid = (logical >= 0) & (logical < num_experts) & mask
    safe = tl.where(valid, logical, 0)
    tok = offs // topk
    row = tok % TABLE_ROWS
    phys = tl.load(table_ptr + row * num_experts + safe, mask=valid, other=-1).to(tl.int32)
    tl.store(physical_ptr + offs, phys, mask=mask)

    rec_on = tl.load(record_enabled_ptr) != 0
    tok_valid = tok < tl.load(num_valid_ptr)
    active = rec_on & tok_valid & valid

    # Per-element vector atomic: no (BLOCK, LOCAL_COUNT) broadcast, which
    # would exceed the A3 UB budget.
    local = phys - local_expert_start
    hit = active & (local >= 0) & (local < LOCAL_COUNT)
    tl.atomic_add(load_ptr + local_expert_start + local, 1, mask=hit)


def gating_map_record(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    routing_table: torch.Tensor,
    record_enabled: torch.Tensor,
    num_valid_tokens: torch.Tensor,
    expert_load_view: torch.Tensor,
    local_expert_start: int,
    local_expert_count: int,
    k: int,
    k_group: int,
    group_count: int,
    routed_scaling_factor: float,
    eps: float = 1e-20,
    num_warps: int = 4,
    tokens_per_program: int | None = None,
):
    """Full-fusion variant C. Returns (weights [T,K] fp32, physical ids [T,K] int32)."""
    num_tokens, num_experts = logits.shape
    group_size = num_experts // group_count
    if tokens_per_program is None:
        # Keep enough programs in flight for occupancy while amortizing the
        # serial per-program reduction chain. The backend only tolerates
        # power-of-two tile heights (TS=5 aborted in parseSelect) and TS=1
        # or TS>16 hit other shape quirks.
        ts = max(2, min(16, num_tokens // 96))
        tokens_per_program = 2 ** (ts.bit_length() - 1)
    weights = torch.empty((num_tokens, k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((num_tokens, k), dtype=torch.int32, device=logits.device)
    gating_map_record_kernel[(triton.cdiv(num_tokens, tokens_per_program),)](
        logits,
        bias if bias is not None else logits,  # dummy pointer when unused
        routing_table,
        record_enabled,
        num_valid_tokens,
        expert_load_view,
        weights,
        ids,
        num_tokens,
        num_experts,
        local_expert_start,
        eps,
        routed_scaling_factor,
        expert_load_view.numel(),
        K=k,
        GROUP_COUNT=group_count,
        GROUP_SIZE=group_size,
        K_GROUP=k_group,
        LOCAL_COUNT=local_expert_count,
        LOCAL_COUNT_POW2=triton.next_power_of_2(max(local_expert_count, 1)),
        TABLE_ROWS=routing_table.shape[0],
        HAS_BIAS=bias is not None,
        E_ALIGN=triton.next_power_of_2(num_experts),
        TOKENS_PER_PROGRAM=tokens_per_program,
        num_warps=num_warps,
    )
    return weights, ids


def map_record(
    topk_ids: torch.Tensor,
    routing_table: torch.Tensor,
    record_enabled: torch.Tensor,
    num_valid_tokens: torch.Tensor,
    expert_load_view: torch.Tensor,
    local_expert_start: int,
    local_expert_count: int,
    block_size: int = 256,
):
    """Variant B: map+record fusion on top of the CANN gating output."""
    numel = topk_ids.numel()
    physical = torch.empty_like(topk_ids)
    map_record_kernel[(triton.cdiv(numel, block_size),)](
        topk_ids.reshape(-1),
        routing_table,
        physical.reshape(-1),
        record_enabled,
        num_valid_tokens,
        expert_load_view,
        routing_table.shape[1],
        numel,
        topk_ids.shape[1],
        local_expert_start,
        LOCAL_COUNT=local_expert_count,
        LOCAL_COUNT_POW2=triton.next_power_of_2(max(local_expert_count, 1)),
        LOCAL_CHUNK=16,
        TABLE_ROWS=routing_table.shape[0],
        BLOCK_SIZE=block_size,
    )
    return physical


def reference_gating_torch(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    k: int,
    k_group: int,
    group_count: int,
    routed_scaling_factor: float,
    eps: float = 1e-20,
):
    """Torch reference of the CANN sigmoid gating semantics (parity oracle)."""
    score = torch.sigmoid(logits.float())
    key = score + bias if bias is not None else score
    num_tokens, num_experts = key.shape
    group_size = num_experts // group_count
    key_g = key.view(num_tokens, group_count, group_size)

    top2 = key_g.topk(2, dim=-1).values.sum(-1)
    group_rank = top2.argsort(dim=-1, descending=True, stable=True)
    selected_groups = torch.zeros_like(key, dtype=torch.bool)
    selected_groups.view(num_tokens, group_count, group_size).scatter_(
        1, group_rank[:, :k_group, None].expand(-1, -1, group_size), True
    )

    masked = torch.where(selected_groups, key, torch.full_like(key, -float("inf")))
    expert_rank = masked.argsort(dim=-1, descending=True, stable=True)
    ids = expert_rank[:, :k]
    weights = torch.gather(score, 1, ids)
    weights = weights / (weights.sum(-1, keepdim=True) + eps) * routed_scaling_factor
    return weights, ids
