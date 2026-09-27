# SPDX-License-Identifier: Apache-2.0
"""Probe: 3D reductions and 2D gather/atomic at the fused-kernel tile shapes."""

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
        print(f"{name}: FAIL {str(exc).splitlines()[0][:130]}")
        return False


@triton.jit
def _k_group3d(x_ptr, o_val_ptr, o_idx_ptr, TS: tl.constexpr, GC: tl.constexpr, GS: tl.constexpr):
    tok0 = tl.program_id(0) * TS
    trows = tok0 + tl.arange(0, TS)
    x = tl.load(x_ptr + trows[:, None, None] * (GC * GS) + tl.arange(0, GC)[None, :, None] * GS + tl.arange(0, GS)[None, None, :])
    x3 = tl.reshape(x, (TS, GC, GS))
    m1 = tl.max(x3, axis=2)  # (TS, GC)
    i1 = tl.argmax(x3, axis=2)  # (TS, GC) local index
    gmax = tl.where(tl.arange(0, GS)[None, None, :] != i1[:, :, None], x3, float("-inf"))
    m2 = tl.max(gmax, axis=2)
    tl.store(o_val_ptr + trows[:, None] * GC + tl.arange(0, GC)[None, :], m1 + m2)
    tl.store(o_idx_ptr + trows[:, None] * GC + tl.arange(0, GC)[None, :], i1)


@triton.jit
def _k_gather2d(table_ptr, idx_ptr, o_ptr, TS: tl.constexpr, K: tl.constexpr, E: tl.constexpr, TR: tl.constexpr):
    tok0 = tl.program_id(0) * TS
    trows = tok0 + tl.arange(0, TS)
    tmask = trows < 4096
    karange = tl.arange(0, K)
    e = tl.load(idx_ptr + trows[:, None] * K + karange[None, :], mask=tmask[:, None], other=0)
    phys = tl.load(table_ptr + (trows % TR)[:, None] * E + e, mask=tmask[:, None], other=-1)
    tl.store(o_ptr + trows[:, None] * K + karange[None, :], phys, mask=tmask[:, None])


@triton.jit
def _k_atomic2d(load_ptr, idx_ptr, TS: tl.constexpr, K: tl.constexpr, LC: tl.constexpr, START: tl.constexpr):
    tok0 = tl.program_id(0) * TS
    trows = tok0 + tl.arange(0, TS)
    karange = tl.arange(0, K)
    e = tl.load(idx_ptr + trows[:, None] * K + karange[None, :])
    local = e - START
    hit = (local >= 0) & (local < LC)
    tl.atomic_add(load_ptr + START + local, 1, mask=hit)


def main():
    torch.npu.set_device(0)
    TS, GC, GS, K, E, TR, LC, START = 16, 8, 32, 8, 256, 1024, 64, 36
    x = torch.randn(64, GC * GS, device="npu:0")
    oval = torch.empty(64, GC, device="npu:0")
    oidx = torch.empty(64, GC, dtype=torch.int32, device="npu:0")
    ok3d = try_kernel("3d max/argmax axis2 (16,8,32)", _k_group3d, (4,), x, oval, oidx, TS, GC, GS)
    if ok3d:
        torch.npu.synchronize()
        xcpu = x.cpu().view(64, GC, GS)
        ref = xcpu.topk(2, dim=2).values.sum(-1)
        print("  3d top2sum matches torch:", bool(torch.allclose(oval.cpu(), ref, atol=1e-5)))

    table = torch.randint(0, 288, (TR, E), dtype=torch.int32, device="npu:0")
    idx = torch.randint(0, E, (4096, K), dtype=torch.int32, device="npu:0")
    out = torch.empty(4096, K, dtype=torch.int32, device="npu:0")
    if try_kernel("2d gather load (16,8)", _k_gather2d, (256,), table, idx, out, TS, K, E, TR):
        torch.npu.synchronize()
        ref = table.cpu()[(torch.arange(4096) % TR)[:, None], idx.cpu()]
        print("  2d gather matches torch:", bool((out.cpu() == ref).all()))

    load = torch.zeros(LC + 8, dtype=torch.int32, device="npu:0")
    e = torch.randint(START, START + LC, (4096, K), dtype=torch.int32, device="npu:0")
    if try_kernel("2d atomic (16,8)", _k_atomic2d, (256,), load, e, TS, K, LC, START):
        torch.npu.synchronize()
        expect = torch.bincount((e.cpu() - START).flatten(), minlength=LC + 8).to(torch.int32)
        print("  2d atomic matches bincount:", bool((load.cpu() == expect).all()))


if __name__ == "__main__":
    main()
