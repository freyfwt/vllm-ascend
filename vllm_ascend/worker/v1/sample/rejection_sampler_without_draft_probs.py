#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import torch
from vllm.triton_utils import tl, triton

PLACEHOLDER_TOKEN_ID = -1


@triton.jit
def _compute_global_lse(
    local_max_ptr,
    local_max_stride,
    local_sumexp_ptr,
    local_sumexp_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    mask = blocks < vocab_num_blocks
    maxes = tl.load(
        local_max_ptr + logit_idx * local_max_stride + blocks,
        mask=mask,
        other=float("-inf"),
    )
    sumexps = tl.load(
        local_sumexp_ptr + logit_idx * local_sumexp_stride + blocks,
        mask=mask,
        other=0.0,
    )
    global_max = tl.max(maxes, axis=0)
    return global_max + tl.log(tl.sum(sumexps * tl.exp(maxes - global_max)))


@triton.jit
def _target_block_stats_kernel(
    target_local_argmax_ptr,
    target_local_argmax_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    target_logits_ptr,
    target_logits_stride,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    temperature_ptr,
    vocab_size,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
):
    logit_idx = tl.program_id(0)
    local_pos = tl.load(expanded_local_pos_ptr + logit_idx)
    if local_pos >= num_speculative_steps:
        return

    req_idx = tl.load(expanded_idx_mapping_ptr + logit_idx)
    temperature = tl.load(temperature_ptr + req_idx).to(tl.float32)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        target_logits_ptr + logit_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    if temperature == 0.0:
        value, idx = tl.max(logits, axis=0, return_indices=True)
        tl.store(
            target_local_argmax_ptr + logit_idx * target_local_argmax_stride + block_idx,
            block_idx * BLOCK_SIZE + idx,
        )
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            value,
        )
    else:
        block_max = tl.max(logits, axis=0)
        block_sumexp = tl.where(
            block_max > float("-inf"),
            tl.sum(tl.exp(logits - block_max)),
            0.0,
        )
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            block_max,
        )
        tl.store(
            target_local_sumexp_ptr + logit_idx * target_local_sumexp_stride + block_idx,
            block_sumexp,
        )


@triton.jit
def _probabilistic_rejection_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    target_logits_ptr,
    target_logits_stride,
    target_local_argmax_ptr,
    target_local_argmax_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_sampled_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    temperature_ptr,
    accept_uniform_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)

    num_sampled = 0
    accepted = True
    num_draft_tokens = end_idx - start_idx - 1
    for i in range(NUM_SPECULATIVE_STEPS):
        if accepted and i < num_draft_tokens:
            logit_idx = start_idx + i
            draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1)
            if temperature == 0.0:
                blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
                mask = blocks < vocab_num_blocks
                local_max = tl.load(
                    target_local_max_ptr + logit_idx * target_local_max_stride + blocks,
                    mask=mask,
                    other=float("-inf"),
                )
                max_block_idx = tl.argmax(local_max, axis=0)
                target_argmax = tl.load(
                    target_local_argmax_ptr + logit_idx * target_local_argmax_stride + max_block_idx
                )
                accepted &= target_argmax == draft_sampled
                tl.store(sampled_ptr + req_idx * sampled_stride + i, target_argmax)
            else:
                target_logit = tl.load(target_logits_ptr + logit_idx * target_logits_stride + draft_sampled).to(
                    tl.float32
                )
                target_lse = _compute_global_lse(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_sumexp_ptr,
                    target_local_sumexp_stride,
                    logit_idx,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                u = tl.load(accept_uniform_ptr + logit_idx).to(tl.float32)
                accepted &= target_logit - target_lse > tl.log(u)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            num_sampled += accepted
    tl.store(num_sampled_ptr + req_idx, num_sampled)


@triton.jit
def _resample_kernel(
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    resampled_local_max_ptr,
    resampled_local_max_stride,
    target_logits_ptr,
    target_logits_stride,
    num_sampled_ptr,
    cu_num_logits_ptr,
    expanded_idx_mapping_ptr,
    draft_sampled_ptr,
    temperature_ptr,
    resample_gumbel_ptr,
    resample_gumbel_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    resample_idx = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + resample_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temperature == 0.0 and not is_bonus:
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        target_logits_ptr + resample_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    if not is_bonus:
        rejected_draft = tl.load(draft_sampled_ptr + resample_token_idx + 1)
        logits = tl.where(block != rejected_draft, logits, float("-inf"))
    if temperature != 0.0:
        logits += tl.load(
            resample_gumbel_ptr + req_idx * resample_gumbel_stride + block,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)

    value, idx = tl.max(logits, axis=0, return_indices=True)
    tl.store(
        resampled_local_argmax_ptr + req_idx * resampled_local_argmax_stride + block_idx,
        block_idx * BLOCK_SIZE + idx,
    )
    tl.store(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block_idx,
        value,
    )


@triton.jit
def _insert_resampled_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    resampled_local_max_ptr,
    resampled_local_max_stride,
    resample_num_blocks,
    cu_num_logits_ptr,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    PADDED_RESAMPLE_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + num_sampled
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)
    if temperature == 0.0 and not is_bonus:
        return

    blocks = tl.arange(0, PADDED_RESAMPLE_NUM_BLOCKS)
    mask = blocks < resample_num_blocks
    local_max = tl.load(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + blocks,
        mask=mask,
        other=float("-inf"),
    )
    max_block_idx = tl.argmax(local_max, axis=0)
    resampled = tl.load(resampled_local_argmax_ptr + req_idx * resampled_local_argmax_stride + max_block_idx)
    tl.store(sampled_ptr + req_idx * sampled_stride + num_sampled, resampled)


def rejection_sample_without_draft_probs(
    target_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    positions: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    accept_uniform: torch.Tensor,
    resample_gumbel: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del positions
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape

    vocab_block_size = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, vocab_block_size)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    target_local_argmax = torch.empty((num_logits, vocab_num_blocks), dtype=torch.int64, device=target_logits.device)
    target_local_max = torch.empty((num_logits, vocab_num_blocks), dtype=torch.float32, device=target_logits.device)
    target_local_sumexp = torch.empty_like(target_local_max)
    _target_block_stats_kernel[(num_logits, vocab_num_blocks)](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_speculative_steps,
        BLOCK_SIZE=vocab_block_size,
    )

    sampled = torch.full(
        (num_reqs, num_speculative_steps + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=target_logits.device,
    )
    num_sampled = torch.empty((num_reqs,), dtype=torch.int32, device=target_logits.device)
    _probabilistic_rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_logits,
        target_logits.stride(0),
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        cu_num_logits,
        idx_mapping,
        temperature,
        accept_uniform,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        NUM_SPECULATIVE_STEPS=num_speculative_steps,
        num_warps=1,
    )

    resample_block_size = 1024
    resample_num_blocks = triton.cdiv(vocab_size, resample_block_size)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = torch.empty(
        (num_reqs, resample_num_blocks), dtype=torch.int64, device=target_logits.device
    )
    resampled_local_max = torch.empty((num_reqs, resample_num_blocks), dtype=torch.float32, device=target_logits.device)
    _resample_kernel[(num_reqs, resample_num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        resample_gumbel,
        resample_gumbel.stride(0),
        vocab_size,
        BLOCK_SIZE=resample_block_size,
    )
    _insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        resample_num_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
    )
    return sampled, num_sampled
