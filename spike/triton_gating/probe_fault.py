# SPDX-License-Identifier: Apache-2.0
"""Bisect the v3 kernel runtime fault: test each suspicious op at bench shapes."""

import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl


def try_kernel(name, fn, grid, *args, **kw):
    try:
        fn[grid](*args, **kw)
        torch.npu.synchronize()
        print(f"{name}: OK")
        return True
    except Exception as exc:
        print(f"{name}: FAIL {str(exc).splitlines()[0][:140]}")
        return False


@triton.jit
def _pack(x, inv_idx, BITS: tl.constexpr):
    u = x.to(tl.int32, bitcast=True)
    mapped = tl.where(u < 0, ~u, u ^ (1 << 31))
    return (mapped.to(tl.int64) & 0xFFFFFFFF) << BITS | inv_idx.to(tl.int64)


@triton.jit
def _k_topk_i64_2d(x_ptr, o_ptr, R: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + row * N + offs)
    p = _pack(x, 511 - offs, 9)
    v = tl.topk(p, K, dim=1)
    tl.store(o_ptr + row * K + tl.arange(0, K), v)


@triton.jit
def _k_topk_i64_2d_nopack(x_ptr, o_ptr, R: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + row * N + offs)
    v = tl.topk(x, K, dim=1)
    tl.store(o_ptr + row * K + tl.arange(0, K), v)


@triton.jit
def _k_atomic_2d(load_ptr, phys_ptr, R: tl.constexpr, K: tl.constexpr, LC: tl.constexpr, START: tl.constexpr):
    row = tl.program_id(0)
    karange = tl.arange(0, K)
    phys = tl.load(phys_ptr + row * K + karange)  # may contain -1 and valid ids
    local = phys - START
    hit = (local >= 0) & (local < LC)
    tl.atomic_add(load_ptr + START + local, 1, mask=hit)


@triton.jit
def _k_gather_load(table_ptr, idx_ptr, o_ptr, R: tl.constexpr, K: tl.constexpr, E: tl.constexpr, TR: tl.constexpr):
    row = tl.program_id(0)
    karange = tl.arange(0, K)
    e = tl.load(idx_ptr + row * K + karange)
    rows = row * 97 % TR
    phys = tl.load(table_ptr + (rows % TR) * E + e, mask=e >= 0, other=-1)
    tl.store(o_ptr + row * K + karange, phys)


def main():
    torch.npu.set_device(0)
    R, N, K, LC, START, E, TR = 16, 256, 8, 64, 36, 256, 1024
    x = torch.randn(R, N, device="npu:0")
    x[:, :8] += 3.0
    packed_out = torch.empty(R, K, dtype=torch.int64, device="npu:0")
    try_kernel("topk packed i64 2d (16,256) k8", _k_topk_i64_2d, (R,), x, packed_out, R, N, K)

    plain_out = torch.empty(R, K, dtype=torch.int64, device="npu:0")
    try_kernel("topk plain i64 2d", _k_topk_i64_2d_nopack, (R,), x, plain_out, R, N, K)

    if try_kernel("verify packed ids", _k_topk_i64_2d, (R,), x, packed_out, R, N, K):
        ids = (511 - (packed_out & 511)).cpu()
        ref = torch.topk(x.cpu(), K, dim=1).indices
        print("packed topk ids match fp32 topk:", bool((ids == ref).all()))

    load = torch.zeros(LC * 2, dtype=torch.int32, device="npu:0")
    phys = torch.cat([
        torch.randint(START, START + LC, (R * K - 4,), dtype=torch.int32),
        torch.tensor([-1, -1, START + LC + 5, -1], dtype=torch.int32),
    ]).to("npu:0")
    try_kernel("atomic 2d with masked negative lanes", _k_atomic_2d, (R,), load, phys, R, K, LC, START)
    torch.npu.synchronize()
    print("atomic load sum:", int(load.sum()))

    table = torch.randint(0, 300, (TR, E), dtype=torch.int32, device="npu:0")
    idx = torch.randint(-1, E, (R, K), dtype=torch.int32, device="npu:0")
    try_kernel("2d gather load with mask", _k_gather_load, (R,), table, idx, torch.empty(R, K, dtype=torch.int32, device="npu:0"), R, K, E, TR)


if __name__ == "__main__":
    main()
