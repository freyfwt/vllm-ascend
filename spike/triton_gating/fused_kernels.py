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

from vllm.triton_utils import tl, triton

_BIG = 1 << 30
_NEG_INF = float("-inf")


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
):
    tok = tl.program_id(0)
    offs = tl.arange(0, E_ALIGN)
    emask = offs < num_experts

    x = tl.load(x_ptr + tok * num_experts + offs, mask=emask, other=0.0).to(tl.float32)
    score = tl.sigmoid(x)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs, mask=emask, other=0.0).to(tl.float32)
        key = tl.where(emask, score + bias, _NEG_INF)
    else:
        key = tl.where(emask, score, _NEG_INF)

    g_idx = offs // GROUP_SIZE
    garange = tl.arange(0, GROUP_COUNT)

    # Group score: sum of the top-2 keys per group. The group-max element is
    # removed by index (not value) so duplicated values are handled.
    gs = tl.full((GROUP_COUNT,), _NEG_INF, tl.float32)
    for g in tl.static_range(GROUP_COUNT):
        gm = g_idx == g
        k1 = tl.max(tl.where(gm, key, _NEG_INF))
        i1 = tl.min(tl.where(gm & (key == k1), offs, _BIG))
        k2 = tl.max(tl.where(gm & (offs != i1), key, _NEG_INF))
        gs = tl.where(garange == g, k1 + k2, gs)

    # Select the top K_GROUP groups; record the winning expert mask.
    sel = tl.zeros((E_ALIGN,), dtype=tl.int32)
    for _ in tl.static_range(K_GROUP):
        gv = tl.max(gs)
        gi = tl.min(tl.where(gs == gv, garange, GROUP_COUNT))
        sel = sel | (g_idx == gi).to(tl.int32)
        gs = tl.where(garange == gi, _NEG_INF, gs)

    # Top-K experts among the selected groups, then map + record per winner.
    cand = tl.where(sel > 0, key, _NEG_INF)
    karange = tl.arange(0, K)
    lc = tl.arange(0, LOCAL_COUNT_POW2)
    hits = tl.zeros((LOCAL_COUNT_POW2,), dtype=tl.int32)
    sc = tl.zeros((K,), dtype=tl.float32)
    rec_on = tl.load(record_enabled_ptr) != 0
    tok_valid = tok < tl.load(num_valid_ptr)
    active = rec_on & tok_valid

    for j in tl.static_range(K):
        v = tl.max(cand)
        e = tl.min(tl.where(cand == v, offs, _BIG)).to(tl.int32)
        s = tl.sum(tl.where(offs == e, score, 0.0))
        phys = tl.load(table_ptr + (tok % TABLE_ROWS) * num_experts + e)
        tl.store(ids_ptr + tok * K + j, phys)
        local = phys - local_expert_start
        hits += ((lc == local) & active).to(tl.int32)
        cand = tl.where(offs == e, _NEG_INF, cand)
        sc = tl.where(karange == j, s, sc)

    tl.store(y_ptr + tok * K + karange, sc / (tl.sum(sc) + eps) * scaling)

    # One vector atomic per program instead of K scalar atomics per token.
    tl.atomic_add(
        load_ptr + local_expert_start + lc,
        hits,
        mask=(lc < LOCAL_COUNT) & active & (lc < num_physical - local_expert_start),
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

    lc = tl.arange(0, LOCAL_COUNT_POW2)
    local = phys.to(tl.int64) - local_expert_start
    hits = tl.sum(((lc[None, :] == local[:, None]) & active[:, None]).to(tl.int32), axis=0)
    any_active = tl.sum(active.to(tl.int32)) > 0
    tl.atomic_add(
        load_ptr + local_expert_start + lc,
        hits,
        mask=(lc < LOCAL_COUNT) & any_active,
    )


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
):
    """Full-fusion variant C. Returns (weights [T,K] fp32, physical ids [T,K] int32)."""
    num_tokens, num_experts = logits.shape
    group_size = num_experts // group_count
    weights = torch.empty((num_tokens, k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((num_tokens, k), dtype=torch.int32, device=logits.device)
    gating_map_record_kernel[(num_tokens,)](
        logits,
        bias if bias is not None else logits,  # dummy pointer when unused
        routing_table,
        record_enabled,
        num_valid_tokens,
        expert_load_view,
        weights,
        ids,
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
