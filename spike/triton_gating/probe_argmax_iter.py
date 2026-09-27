# SPDX-License-Identifier: Apache-2.0
"""Minimal repro: iterative tl.argmax top-k on 2D tiles."""

import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl


@triton.jit
def _k_argmax(x_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    cand = tl.load(x_ptr + row * N + offs)
    karange = tl.arange(0, K)
    e_all = tl.zeros((K,), dtype=tl.int32)
    for j in tl.static_range(K):
        e_j = tl.argmax(cand, axis=0).to(tl.int32)
        e_all = tl.where(karange == j, e_j, e_all)
        cand = tl.where(offs == e_j, float("-inf"), cand)
    tl.store(o_ptr + karange, e_all)


@triton.jit
def _k_maxmin(x_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    cand = tl.load(x_ptr + row * N + offs)
    karange = tl.arange(0, K)
    e_all = tl.zeros((K,), dtype=tl.int32)
    for j in tl.static_range(K):
        v = tl.max(cand, axis=0)
        e_j = tl.min(tl.where(offs == v, offs, N), axis=0).to(tl.int32)
        e_all = tl.where(karange == j, e_j, e_all)
        cand = tl.where(offs == e_j, float("-inf"), cand)
    tl.store(o_ptr + karange, e_all)


def main():
    torch.npu.set_device(0)
    T, N, K = 16, 256, 8
    gen = torch.Generator(device="cpu").manual_seed(9)
    x = torch.randn(T, N, generator=gen).to(torch.float32).to("npu:0")
    ref = torch.topk(x.cpu(), K, dim=1).indices
    for name, fn in (("argmax", _k_argmax), ("max+min", _k_maxmin)):
        out = torch.empty(T, K, dtype=torch.int32, device="npu:0")
        fn[(T,)](x, out, N, K)
        torch.npu.synchronize()
        print(f"{name}: ids match topk =", bool((out.cpu() == ref).all()),
              "| row0:", out.cpu()[0].tolist())


if __name__ == "__main__":
    main()
