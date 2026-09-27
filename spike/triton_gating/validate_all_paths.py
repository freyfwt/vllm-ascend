# SPDX-License-Identifier: Apache-2.0
"""Bit-exact validation of all fused gating routes against the CANN ops.

Routes covered: sigmoid (regression), softmax renorm 0/1 with and without
bias, and the DeepSeek V4 hash route. Every route checks ids (exact),
weights (max abs diff) and the recorded load (integer-equal).
"""

import os

import torch
import torch_npu  # noqa: F401

import vllm_ascend.vllm_ascend_C  # noqa: F401  (registers _C_ascend ops)

from vllm_ascend.ops.triton.eplb_map_record import (
    gating_map_record,
    hash_map_record,
)

NUM_LOGICAL = 256
NUM_REDUNDANT = 32
EP_SIZE = 8
EP_RANK = 1
K = 8
K_GROUP = 4
GROUP_COUNT = 8
SCALING = 2.5
EPS = 1e-20
TABLE_ROWS = 1024


def make_table(device):
    gen = torch.Generator(device="cpu").manual_seed(11)
    l2p = torch.stack([torch.arange(NUM_LOGICAL), torch.full((NUM_LOGICAL,), -1)], dim=1)
    rep = torch.ones(NUM_LOGICAL, dtype=torch.int64)
    for i in range(NUM_REDUNDANT):
        l2p[i, 1] = NUM_LOGICAL + i
        rep[i] = 2
    rows = torch.arange(TABLE_ROWS, dtype=torch.int64)[:, None]
    logical = torch.arange(NUM_LOGICAL, dtype=torch.int64)[None, :]
    counts = rep[None, :]
    idx = (rows + EP_RANK + logical) % counts
    table = l2p.gather(1, idx.T).T.to(torch.int32).contiguous().to(device)
    local_count = (NUM_LOGICAL + NUM_REDUNDANT) // EP_SIZE
    return table, EP_RANK * local_count, local_count


def cann_gating(logits, bias, norm_type, renorm):
    w, ids, _ = torch.ops._C_ascend.moe_gating_top_k(
        logits,
        k=K,
        k_group=K_GROUP,
        group_count=GROUP_COUNT,
        group_select_mode=1,
        renorm=renorm,
        norm_type=norm_type,
        out_flag=False,
        routed_scaling_factor=SCALING,
        eps=EPS,
        bias_opt=bias,
    )
    return w, ids.to(torch.int32)


def cann_hash(logits, input_ids, tid2eid, bias):
    w, ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
        x=logits,
        k=K,
        bias=bias,
        input_ids=input_ids,
        tid2eid=tid2eid,
        k_group=1,
        group_count=1,
        routed_scaling_factor=SCALING,
        eps=EPS,
        group_select_mode=1,
        renorm=0,
        norm_type=2,
        out_flag=False,
    )
    return w, ids.to(torch.int32)


def check(name, ids_a, w_a, ids_c, w_c, load_a, load_c):
    ids_eq = bool((ids_a == ids_c).all())
    wdiff = (w_a.float() - w_c.float()).abs().max().item()
    load_eq = bool((load_a == load_c).all())
    verdict = "PASS" if ids_eq and load_eq and wdiff < 1e-5 else "FAIL"
    print(f"{name}: {verdict} ids_eq={ids_eq} load_eq={load_eq} wdiff={wdiff:.2e}")
    if not ids_eq:
        mm = (ids_a != ids_c).nonzero()[:5]
        print("  mismatch examples:", mm.tolist(),
              "A:", ids_a[mm[:, 0], mm[:, 1]].tolist(), "C:", ids_c[mm[:, 0], mm[:, 1]].tolist())
    return verdict == "PASS"


def main():
    torch.npu.set_device(0)
    device = "npu:0"
    T = 512
    gen = torch.Generator(device="cpu").manual_seed(3)
    logits = torch.randn(T, NUM_LOGICAL, generator=gen).to(device)
    bias = (torch.randn(NUM_LOGICAL, generator=gen) * 0.5).to(device)
    table, local_start, local_count = make_table(device)
    record_on = torch.ones((), dtype=torch.int32, device=device)
    num_valid = torch.full((), T, dtype=torch.int32, device=device)
    num_physical = NUM_LOGICAL + NUM_REDUNDANT
    ok = True

    def bincount_load(physical):
        in_local = (physical >= local_start) & (physical < local_start + local_count)
        return torch.bincount(
            (physical[in_local] - local_start).flatten(), minlength=local_count
        ).to(torch.int32)

    def full_load(physical):
        load = torch.zeros(num_physical, dtype=torch.int32, device=device)
        load[: local_count] = bincount_load(physical)  # placeholder layout guard
        full = torch.zeros(num_physical, dtype=torch.int32, device=device)
        counts = torch.bincount((physical - local_start).clamp(0, num_physical).flatten(), minlength=num_physical)
        full[:] = counts[:num_physical].to(torch.int32)
        return full

    # --- sigmoid (regression) ---
    load_a = torch.zeros(num_physical, dtype=torch.int32, device=device)
    load_c = torch.zeros(num_physical, dtype=torch.int32, device=device)
    w_a, ids_a = cann_gating(logits, bias, 1, 1)
    phys_a = ids_a  # cann returns logical ids; map manually below
    # map logical -> physical through the table for load comparison
    rows = torch.arange(T, device=device) % TABLE_ROWS
    phys_a = table[rows[:, None], ids_a]
    load_a[: local_count] = bincount_load(phys_a)[  # only local slice matters
        :local_count] if False else load_a[:local_count]
    load_a[:] = 0
    load_a[local_start: local_start + local_count] = bincount_load(phys_a)
    w_c, ids_c = gating_map_record(
        logits, bias, table, record_on, num_valid, load_c, local_start, local_count,
        k=K, k_group=K_GROUP, group_count=GROUP_COUNT, routed_scaling_factor=SCALING,
        norm_type=1, renorm=True,
        tokens_per_program=int(os.getenv("TS_OVERRIDE") or 0) or None,
    )
    torch.npu.synchronize()
    ok &= check("sigmoid+bias renorm1", phys_a, w_a, ids_c, w_c,
                load_a[local_start: local_start + local_count],
                load_c[local_start: local_start + local_count])
    print("DBG row0 A:", phys_a[0].tolist())
    print("DBG row0 C:", ids_c[0].tolist())
    print("DBG row1 A:", phys_a[1].tolist())
    print("DBG row1 C:", ids_c[1].tolist())
    print("DBG row2 A:", phys_a[2].tolist())
    print("DBG row2 C:", ids_c[2].tolist())

    # --- softmax renorm 0/1, with/without bias ---
    for norm_bias in (bias, None):
        for rn in (0, 1):
            tag = f"softmax bias={norm_bias is not None} renorm={rn}"
            load_a = torch.zeros(num_physical, dtype=torch.int32, device=device)
            load_c = torch.zeros(num_physical, dtype=torch.int32, device=device)
            w_a, ids_a = cann_gating(logits, norm_bias, 0, rn)
            phys_a = table[torch.arange(T, device=device)[:, None] % TABLE_ROWS, ids_a]
            load_a[local_start: local_start + local_count] = bincount_load(phys_a)
            w_c, ids_c = gating_map_record(
                logits, norm_bias, table, record_on, num_valid, load_c, local_start,
                local_count, k=K, k_group=K_GROUP, group_count=GROUP_COUNT,
                routed_scaling_factor=SCALING, norm_type=0, renorm=bool(rn),
            )
            torch.npu.synchronize()
            ok &= check(tag, phys_a, w_a, ids_c, w_c,
                        load_a[local_start: local_start + local_count],
                        load_c[local_start: local_start + local_count])

    # --- hash (DSV4-style): tid2eid lookup, sigmoid weights ---
    num_ids = 4096
    # tid2eid holds LOGICAL expert ids (< num_logical); the replica table
    # maps them to physical slots, exactly like production.
    tid2eid = torch.randint(0, NUM_LOGICAL, (num_ids, K), dtype=torch.int32, generator=gen).to(device)
    input_ids = torch.randint(0, num_ids, (T,), dtype=torch.int64, generator=gen).to(device)
    load_a = torch.zeros(num_physical, dtype=torch.int32, device=device)
    load_c = torch.zeros(num_physical, dtype=torch.int32, device=device)
    import os
    hash_bias = None if os.getenv("HASH_NO_BIAS") else bias
    w_a, ids_a = cann_hash(logits, input_ids, tid2eid, hash_bias)
    # cann hash already returns final expert ids (logical == table domain here:
    # route them through the table the same way the Triton kernel does)
    phys_a = table[torch.arange(T, device=device)[:, None] % TABLE_ROWS, ids_a]
    load_a[local_start: local_start + local_count] = bincount_load(phys_a)
    w_c, ids_c = hash_map_record(
        logits, input_ids, tid2eid, table, record_on, num_valid, load_c,
        local_start, local_count, k=K, routed_scaling_factor=SCALING,
    )
    print("  first-row ids A:", ids_a[0].tolist(), "C:", ids_c[0].tolist(),
          "lookup:", tid2eid[input_ids[:1].cpu()].tolist())
    torch.npu.synchronize()
    ok &= check("hash (tid2eid)", phys_a, w_a, ids_c, w_c,
                load_a[local_start: local_start + local_count],
                load_c[local_start: local_start + local_count])

    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
