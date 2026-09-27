# SPDX-License-Identifier: Apache-2.0
"""Capability probe for triton-ascend 3.2.x: topk/sort dtypes, axes, bitcast."""

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
        msg = str(exc).split("\n")[0][:160]
        print(f"{name}: FAIL {msg}")
        return False


@triton.jit
def _k_topk_f32_2d(x_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + row * N + offs)
    v = tl.topk(x, K, dim=0)
    tl.store(o_ptr + row * K + tl.arange(0, K), v)


@triton.jit
def _k_sort_i64(x_ptr, o_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    v = tl.sort(x, descending=True)
    tl.store(o_ptr + offs, v)


@triton.jit
def _k_topk_i64(x_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    v = tl.topk(x, K)
    tl.store(o_ptr + tl.arange(0, K), v)


@triton.jit
def _k_sort_f32_2d_axis1(x_ptr, o_ptr, R: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + row * N + offs)
    v = tl.sort(x, descending=True)
    tl.store(o_ptr + row * N + offs, v)


@triton.jit
def _k_bitcast_map(x_ptr, o_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    u = x.to(tl.int32, bitcast=True)
    mapped = tl.where(u < 0, ~u, u ^ (1 << 31))
    packed = (mapped.to(tl.int64) & 0xFFFFFFFF) << 9 | (offs & 511).to(tl.int64)
    tl.store(o_ptr + offs, packed)


@triton.jit
def _k_argmax_axis1(x_ptr, o_ptr, R: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + row * N + offs)
    i = tl.argmax(x, axis=0, tie_break_left=True)
    tl.store(o_ptr + row, i)


def main():
    device = "npu:0"
    torch.npu.set_device(0)
    N, K, R = 256, 8, 4
    xf = torch.randn(R, N, device=device)
    x64 = torch.randint(-(2**40), 2**40, (N,), dtype=torch.int64, device=device)
    out = torch.empty(4, K, dtype=torch.float32, device=device)

    try_kernel("topk f32 2d rows", _k_topk_f32_2d, (R,), xf, out, N, K)
    try_kernel("sort i64", _k_sort_i64, (1,), x64, torch.empty(N, dtype=torch.int64, device=device), N)
    try_kernel("topk i64", _k_topk_i64, (1,), x64, torch.empty(K, dtype=torch.int64, device=device), N, K)
    try_kernel("sort f32 2d axis1", _k_sort_f32_2d_axis1, (R,), xf, torch.empty(R, N, device=device), R, N)
    try_kernel("bitcast pack i64", _k_bitcast_map, (1,), xf[0], torch.empty(N, dtype=torch.int64, device=device), N)
    try_kernel("argmax axis0 2d", _k_argmax_axis1, (R,), xf, torch.empty(R, dtype=torch.int64, device=device), R, N)

    # numeric check of the packing trick end to end
    packed = torch.empty(N, dtype=torch.int64, device=device)
    _k_bitcast_map[(1,)](xf[0], packed, N)
    torch.npu.synchronize()
    order_ref = xf[0].argsort(descending=True)
    order_packed = torch.argsort(packed, descending=True)
    print("packed order matches fp32 argsort:", bool((order_ref == order_packed).all()))


if __name__ == "__main__":
    main()
