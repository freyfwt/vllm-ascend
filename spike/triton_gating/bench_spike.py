# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Spike benchmark: original 3-launch stack vs fused Triton variants.

Routes compared (PR1 design doc §1.3):

  A  CANN moe_gating_top_k + Triton map + Triton record   (current, 3 launches)
  B  CANN moe_gating_top_k + Triton fused map+record      (2 launches)
  C  Triton fused gating+map+record                       (1 launch, spike)

A's expert_tokens come from the MoE operator in production; here they are
synthesized once with bincount and cached, so A is not penalized by the
synthesis cost. Parity checks: physical ids must match exactly, weights
within tolerance, recorded load integer-equal; a padding case proves that
variant C excludes padding rows while A cannot.

Usage (inside the NPU container, repo root on PYTHONPATH):
  python spike/triton_gating/bench_spike.py [--iters 100] [--warmup 20]
      [--shapes 64,1024,6144] [--num-warps 4] [--out results.json]
"""

import argparse
import json
import statistics

import torch
import torch_npu  # noqa: F401

from vllm.triton_utils import tl, triton  # noqa: F401  (env probe needs tl/triton)

import vllm_ascend  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401  (lazy native ext; registers _C_ascend ops)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.eplb import build_expert_replica_routing_table
from vllm_ascend.ops.triton.eplb import (
    map_to_physical_triton,
    record_expert_tokens_triton,
)

from spike.triton_gating.fused_kernels import (
    gating_map_record,
    map_record,
    reference_gating_torch,
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


def probe_env() -> dict:
    info = {
        "torch": torch.__version__,
        "triton": getattr(triton, "__version__", "unknown"),
        "device": torch.npu.get_device_name(0),
        "tl.sigmoid": hasattr(tl, "sigmoid"),
        "tl.static_range": hasattr(tl, "static_range"),
        "tl.atomic_add": hasattr(tl, "atomic_add"),
        "tl.argmax": hasattr(tl, "argmax"),
        "tl.sort": hasattr(tl, "sort"),
        "tl.topk": hasattr(tl, "topk"),
        "tl.histogram": hasattr(tl, "histogram"),
        "cann_gating_op": False,
    }
    try:
        torch.ops._C_ascend.moe_gating_top_k
        info["cann_gating_op"] = True
    except (AttributeError, RuntimeError):
        pass
    return info


def make_setup(num_tokens: int, device) -> dict:
    num_physical = NUM_LOGICAL + NUM_REDUNDANT
    gen = torch.Generator(device="cpu").manual_seed(7)
    l2p = torch.full((NUM_LOGICAL, 2), -1, dtype=torch.int64)
    rep = torch.ones(NUM_LOGICAL, dtype=torch.int64)
    for i in range(NUM_REDUNDANT):
        l2p[i, 1] = NUM_LOGICAL + i
        rep[i] = 2
    table = build_expert_replica_routing_table(
        l2p.to(device), rep.to(device), EP_RANK
    ).contiguous()
    local_count = num_physical // EP_SIZE
    local_start = EP_RANK * local_count

    logits = torch.randn(num_tokens, NUM_LOGICAL, generator=gen).to(device)
    bias = (torch.randn(NUM_LOGICAL, generator=gen) * 0.5).to(device)
    record_enabled = torch.ones((), dtype=torch.int32, device=device)
    num_valid = torch.full((), num_tokens, dtype=torch.int32, device=device)
    zeros = lambda: torch.zeros(num_physical, dtype=torch.int32, device=device)
    return dict(
        num_tokens=num_tokens,
        table=table, local_start=local_start, local_count=local_count,
        logits=logits, bias=bias, record_enabled=record_enabled,
        num_valid=num_valid, load_a=zeros(), load_b=zeros(), load_c=zeros(),
        num_physical=num_physical,
    )


def run_baseline_gating(s):
    return DeviceOperator.moe_gating_top_k(
        s["logits"],
        k=K,
        k_group=K_GROUP,
        group_count=GROUP_COUNT,
        group_select_mode=1,
        renorm=1,
        norm_type=1,  # sigmoid
        out_flag=False,
        routed_scaling_factor=SCALING,
        eps=EPS,
        bias_opt=s["bias"],
    )


def synthesize_counts(physical: torch.Tensor, s: dict) -> torch.Tensor:
    in_local = (physical >= s["local_start"]) & (
        physical < s["local_start"] + s["local_count"]
    )
    return torch.bincount(
        (physical[in_local] - s["local_start"]).flatten(),
        minlength=s["local_count"],
    ).to(torch.int32)


def route_a(s, counts: torch.Tensor | None = None):
    """Current 3-launch route; returns (weights, physical ids)."""
    weights, ids_logical, _ = run_baseline_gating(s)
    physical = map_to_physical_triton(ids_logical, s["table"])
    if counts is None:
        counts = synthesize_counts(physical, s)
    record_expert_tokens_triton(
        counts, s["load_a"], s["record_enabled"], 1, s["local_start"]
    )
    return weights, physical


def route_b(s, ids_logical: torch.Tensor):
    """CANN gating + fused Triton map+record (2 launches)."""
    weights, _, _ = run_baseline_gating(s)
    physical = map_record(
        ids_logical,
        s["table"],
        s["record_enabled"],
        s["num_valid"],
        s["load_b"],
        s["local_start"],
        s["local_count"],
    )
    return weights, physical


def route_c(s, num_warps: int):
    """Full-fusion Triton kernel (1 launch)."""
    return gating_map_record(
        s["logits"],
        s["bias"],
        s["table"],
        s["record_enabled"],
        s["num_valid"],
        s["load_c"],
        s["local_start"],
        s["local_count"],
        k=K,
        k_group=K_GROUP,
        group_count=GROUP_COUNT,
        routed_scaling_factor=SCALING,
        eps=EPS,
        num_warps=num_warps,
    )


def time_fn(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": samples[len(samples) // 2],
        "min_ms": samples[0],
        "p99_ms": samples[max(int(len(samples) * 0.99) - 1, 0)],
    }


def parity(s, w_a, ids_a, w_c, ids_c) -> dict:
    mismatches = (ids_a != ids_c).nonzero()
    ref_w, ref_ids = reference_gating_torch(
        s["logits"], s["bias"], K, K_GROUP, GROUP_COUNT, SCALING, EPS
    )
    return {
        "ids_equal": bool((ids_a == ids_c).all()),
        "id_mismatch_count": int(len(mismatches)),
        "id_mismatch_examples": mismatches[:5].tolist(),
        "weight_max_abs_diff": (w_a - w_c).abs().max().item(),
        "weight_ok": bool(torch.allclose(w_a, w_c, rtol=1e-5, atol=1e-6)),
        "load_equal": bool((s["load_a"] == s["load_c"]).all()),
        "torch_reference_ids_match_cann": bool((ref_ids.to(ids_a.device) == ids_a).all()),
        "torch_reference_weight_diff": (ref_w.to(w_a.device) - w_a).abs().max().item(),
    }


def padding_check(s, ids_c_full: torch.Tensor) -> dict:
    """C with num_valid < T must exclude padding rows from the recorded load."""
    pad_rows = max(s["num_tokens"] // 8, 1)
    load_full = s["load_c"].clone()
    s["load_c"].zero_()
    s["num_valid"].fill_(s["num_tokens"] - pad_rows)
    route_c(s, num_warps=4)
    torch.npu.synchronize()

    pad_mask = torch.zeros(s["num_tokens"], dtype=torch.bool, device=ids_c_full.device)
    pad_mask[-pad_rows:] = True
    in_local = (
        (ids_c_full >= s["local_start"])
        & (ids_c_full < s["local_start"] + s["local_count"])
        & pad_mask[:, None]
    )
    expected_removed = torch.bincount(
        (ids_c_full[in_local] - s["local_start"]).flatten(),
        minlength=s["local_count"],
    ).to(torch.int32)
    removed = load_full - s["load_c"]

    s["num_valid"].fill_(s["num_tokens"])
    s["load_c"].copy_(load_full)
    return {"padding_excluded": bool((removed == expected_removed).all())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="64,1024,6144")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--out", default="spike_triton_gating_results.json")
    args = parser.parse_args()

    device = "npu:0"
    torch.npu.set_device(0)
    results = {"env": probe_env(), "config": vars(args), "cases": {}}
    print(json.dumps(results["env"], indent=2))
    if not results["env"]["cann_gating_op"]:
        raise SystemExit("FATAL: torch.ops._C_ascend.moe_gating_top_k unavailable")

    keys = ("A_gating", "A_map", "A_record", "A_total", "B_map_record",
            "B_total", "C_total")
    for num_tokens in [int(t) for t in args.shapes.split(",")]:
        s = make_setup(num_tokens, device)
        case: dict = {}

        # Parity: one untimed run per route, then compare.
        w_a, ids_a = route_a(s)
        counts = synthesize_counts(ids_a, s)
        w_b, ids_b = route_b(s, ids_a)
        w_c, ids_c = route_c(s, args.num_warps)
        torch.npu.synchronize()
        case["parity_a_vs_c"] = parity(s, w_a, ids_a, w_c, ids_c)
        case["parity_b_ids_vs_a"] = bool((ids_b == ids_a).all())
        case["parity_load_b_vs_a"] = bool((s["load_b"] == s["load_a"]).all())
        case["padding"] = padding_check(s, ids_c)

        # Cached intermediates so timing loops do not re-synthesize counts.
        ids_logical = run_baseline_gating(s)[1]
        torch.npu.synchronize()

        case["A_gating"] = time_fn(lambda: run_baseline_gating(s), args.warmup, args.iters)
        case["A_map"] = time_fn(
            lambda: map_to_physical_triton(ids_logical, s["table"]), args.warmup, args.iters
        )
        case["A_record"] = time_fn(
            lambda: record_expert_tokens_triton(
                counts, s["load_a"], s["record_enabled"], 1, s["local_start"]
            ),
            args.warmup, args.iters,
        )

        def route_a_timed():
            run_baseline_gating(s)
            map_to_physical_triton(ids_logical, s["table"])
            record_expert_tokens_triton(
                counts, s["load_a"], s["record_enabled"], 1, s["local_start"]
            )

        def route_b_timed():
            run_baseline_gating(s)
            map_record(
                ids_logical, s["table"], s["record_enabled"], s["num_valid"],
                s["load_b"], s["local_start"], s["local_count"],
            )

        case["A_total"] = time_fn(route_a_timed, args.warmup, args.iters)
        case["B_map_record"] = time_fn(
            lambda: map_record(
                ids_logical, s["table"], s["record_enabled"], s["num_valid"],
                s["load_b"], s["local_start"], s["local_count"],
            ),
            args.warmup, args.iters,
        )
        case["B_total"] = time_fn(route_b_timed, args.warmup, args.iters)
        case["C_total"] = time_fn(
            lambda: route_c(s, args.num_warps), args.warmup, args.iters
        )

        case["speedup_C_vs_A"] = case["A_total"]["p50_ms"] / case["C_total"]["p50_ms"]
        case["speedup_C_vs_B"] = case["B_total"]["p50_ms"] / case["C_total"]["p50_ms"]
        results["cases"][f"T={num_tokens}"] = case
        print(f"\n=== T={num_tokens} ===")
        for key in keys:
            print(f"{key:>14}: p50={case[key]['p50_ms']:.4f} ms  "
                  f"mean={case[key]['mean_ms']:.4f}  min={case[key]['min_ms']:.4f}")
        print(f"{'C vs A':>14}: x{case['speedup_C_vs_A']:.2f}   "
              f"{'C vs B':>7}: x{case['speedup_C_vs_B']:.2f}")
        print("parity:", json.dumps({k: v for k, v in case.items()
                                     if k.startswith("parity") or k == "padding"}))

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
