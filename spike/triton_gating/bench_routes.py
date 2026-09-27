# SPDX-License-Identifier: Apache-2.0
"""Crossover benchmark: unfused CANN route vs the fused Triton kernel."""

import argparse
import statistics

import torch
import torch_npu  # noqa: F401

import vllm_ascend.vllm_ascend_C  # noqa: F401
from vllm_ascend.ops.triton.eplb import (
    map_to_physical_triton,
    record_expert_tokens_triton,
)
from vllm_ascend.ops.triton.eplb_map_record import gating_map_record

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


def time_fn(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.npu.synchronize()
        samples.append(s.elapsed_time(e))
    samples.sort()
    return samples[len(samples) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="64,256,512,1024,6144")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    torch.npu.set_device(0)
    device = "npu:0"
    num_physical = NUM_LOGICAL + NUM_REDUNDANT
    local_count = num_physical // EP_SIZE
    local_start = EP_RANK * local_count

    gen = torch.Generator(device="cpu").manual_seed(7)
    rows = torch.arange(TABLE_ROWS, dtype=torch.int64)[:, None]
    logical = torch.arange(NUM_LOGICAL, dtype=torch.int64)[None, :]
    l2p = torch.stack([
        torch.arange(NUM_LOGICAL),
        torch.where(torch.arange(NUM_LOGICAL) < NUM_REDUNDANT,
                    torch.arange(NUM_LOGICAL) + NUM_LOGICAL, torch.full((NUM_LOGICAL,), -1)),
    ], dim=1)
    rep = torch.where(torch.arange(NUM_LOGICAL) < NUM_REDUNDANT, 2, 1)
    idx = (rows + EP_RANK + logical) % rep[None, :]
    table = l2p.gather(1, idx.T).T.to(torch.int32).contiguous().to(device)

    for T in [int(t) for t in args.shapes.split(",")]:
        logits = torch.randn(T, NUM_LOGICAL, generator=gen).to(device)
        bias = (torch.randn(NUM_LOGICAL, generator=gen) * 0.5).to(device)
        ro = torch.ones((), dtype=torch.int32, device=device)
        roff = torch.zeros((), dtype=torch.int32, device=device)
        nv = torch.full((), T, dtype=torch.int32, device=device)
        load = torch.zeros(num_physical, dtype=torch.int32, device=device)

        def route_a():
            w, ids, _ = torch.ops._C_ascend.moe_gating_top_k(
                logits, k=K, k_group=K_GROUP, group_count=GROUP_COUNT,
                group_select_mode=1, renorm=1, norm_type=1, out_flag=False,
                routed_scaling_factor=SCALING, eps=EPS, bias_opt=bias)
            phys = map_to_physical_triton(ids, table)
            in_local = (phys >= local_start) & (phys < local_start + local_count)
            counts = torch.bincount((phys[in_local] - local_start).flatten(), minlength=local_count).to(torch.int32)
            record_expert_tokens_triton(counts, load, ro, 1, local_start)

        def route_c(record_enabled):
            return gating_map_record(
                logits, bias, table, record_enabled, nv, load, local_start,
                local_count, k=K, k_group=K_GROUP, group_count=GROUP_COUNT,
                routed_scaling_factor=SCALING, eps=EPS, norm_type=1, renorm=True)

        a = time_fn(route_a, iters=args.iters)
        c_on = time_fn(lambda: route_c(ro), iters=args.iters)
        c_off = time_fn(lambda: route_c(roff), iters=args.iters)
        print(f"T={T:>5}: A={a:.4f} ms | C(on)={c_on:.4f} ({a / c_on:.2f}x) | "
              f"C(off)={c_off:.4f} ({a / c_off:.2f}x) | atomic cost={(c_on - c_off):.4f} ms")


if __name__ == "__main__":
    main()
