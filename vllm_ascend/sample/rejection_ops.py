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


class RejectionWorkspace:
    def __init__(self) -> None:
        self._buffers: dict[str, torch.Tensor] = {}

    def prepare(
        self,
        target_logits: torch.Tensor,
        num_reqs: int,
        num_logits: int,
        vocab_num_blocks: int,
        recovery_blocks: int,
        num_speculative_steps: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        target_local_argmax = self._buffer(
            "target_local_argmax",
            (num_logits, vocab_num_blocks),
            torch.int64,
            target_logits.device,
        )
        target_local_max = self._buffer(
            "target_local_max",
            (num_logits, vocab_num_blocks),
            torch.float32,
            target_logits.device,
        )
        target_local_sumexp = self._buffer(
            "target_local_sumexp",
            (num_logits, vocab_num_blocks),
            torch.float32,
            target_logits.device,
        )
        sampled = self._buffer(
            "sampled",
            (num_reqs, num_speculative_steps + 1),
            torch.int32,
            target_logits.device,
        )
        sampled.fill_(PLACEHOLDER_TOKEN_ID)
        num_sampled = self._buffer(
            "num_sampled",
            (num_reqs,),
            torch.int32,
            target_logits.device,
        )
        recovery_local_argmax = self._buffer(
            "recovery_local_argmax",
            (num_reqs, recovery_blocks),
            torch.int64,
            target_logits.device,
        )
        recovery_local_max = self._buffer(
            "recovery_local_max",
            (num_reqs, recovery_blocks),
            torch.float32,
            target_logits.device,
        )
        return (
            target_local_argmax,
            target_local_max,
            target_local_sumexp,
            sampled,
            num_sampled,
            recovery_local_argmax,
            recovery_local_max,
        )

    def _buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        buffer = self._buffers.get(name)
        if (
            buffer is None
            or buffer.dtype != dtype
            or buffer.device != device
            or any(buffer.shape[i] < shape[i] for i in range(len(shape)))
        ):
            buffer = torch.empty(shape, dtype=dtype, device=device)
            self._buffers[name] = buffer
        return buffer[tuple(slice(0, dim) for dim in shape)]


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
def _target_stats_kernel(
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
    draft_tokens_ptr,
    draft_probs_ptr,
    draft_probs_stride,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    temperature_ptr,
    acceptance_uniform_ptr,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_PROBS: tl.constexpr,
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
            draft_tokens = tl.load(draft_tokens_ptr + logit_idx + 1)
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
                accepted &= target_argmax == draft_tokens
                tl.store(sampled_ptr + req_idx * sampled_stride + i, target_argmax)
            else:
                target_logit = tl.load(target_logits_ptr + logit_idx * target_logits_stride + draft_tokens).to(
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
                u = tl.load(acceptance_uniform_ptr + logit_idx).to(tl.float32)
                target_log_prob = target_logit - target_lse
                if HAS_DRAFT_PROBS:
                    draft_prob = tl.load(draft_probs_ptr + logit_idx * draft_probs_stride + draft_tokens).to(tl.float32)
                    accepted &= target_log_prob > tl.log(u) + tl.log(draft_prob)
                else:
                    accepted &= target_log_prob > tl.log(u)
                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_tokens)
            num_sampled += accepted
    tl.store(num_sampled_ptr + req_idx, num_sampled)


@triton.jit
def _recovery_kernel(
    recovery_local_argmax_ptr,
    recovery_local_argmax_stride,
    recovery_local_max_ptr,
    recovery_local_max_stride,
    target_logits_ptr,
    target_logits_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    num_sampled_ptr,
    cu_num_logits_ptr,
    expanded_idx_mapping_ptr,
    draft_tokens_ptr,
    draft_probs_ptr,
    draft_probs_stride,
    temperature_ptr,
    recovery_gumbel_ptr,
    recovery_gumbel_stride,
    vocab_size,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    recovery_idx = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    recovery_token_idx = start_idx + recovery_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + recovery_token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    is_bonus = recovery_token_idx == end_idx - 1
    if temperature == 0.0 and not is_bonus:
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        target_logits_ptr + recovery_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    if not is_bonus:
        if HAS_DRAFT_PROBS:
            target_lse = _compute_global_lse(
                target_local_max_ptr,
                target_local_max_stride,
                target_local_sumexp_ptr,
                target_local_sumexp_stride,
                recovery_token_idx,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            )
            target_log_probs = logits - target_lse
            draft_probs = tl.load(
                draft_probs_ptr + recovery_token_idx * draft_probs_stride + block,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            draft_log_probs = tl.log(draft_probs)
            ratio = tl.exp(draft_log_probs - target_log_probs)
            logits = tl.where(
                ratio < 1.0,
                target_log_probs + tl.log(1.0 - ratio),
                float("-inf"),
            )
        else:
            rejected_draft = tl.load(draft_tokens_ptr + recovery_token_idx + 1)
            logits = tl.where(block != rejected_draft, logits, float("-inf"))
    if temperature != 0.0:
        logits += tl.load(
            recovery_gumbel_ptr + req_idx * recovery_gumbel_stride + block,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)

    value, idx = tl.max(logits, axis=0, return_indices=True)
    tl.store(
        recovery_local_argmax_ptr + req_idx * recovery_local_argmax_stride + block_idx,
        block_idx * BLOCK_SIZE + idx,
    )
    tl.store(
        recovery_local_max_ptr + req_idx * recovery_local_max_stride + block_idx,
        value,
    )


@triton.jit
def _insert_recovery_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    recovery_local_argmax_ptr,
    recovery_local_argmax_stride,
    recovery_local_max_ptr,
    recovery_local_max_stride,
    recovery_blocks,
    cu_num_logits_ptr,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    PADDED_RECOVERY_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    recovery_token_idx = start_idx + num_sampled
    req_state_idx = tl.load(expanded_idx_mapping_ptr + recovery_token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    is_bonus = recovery_token_idx == end_idx - 1
    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)
    if temperature == 0.0 and not is_bonus:
        return

    blocks = tl.arange(0, PADDED_RECOVERY_BLOCKS)
    mask = blocks < recovery_blocks
    local_max = tl.load(
        recovery_local_max_ptr + req_idx * recovery_local_max_stride + blocks,
        mask=mask,
        other=float("-inf"),
    )
    max_block_idx = tl.argmax(local_max, axis=0)
    recovery = tl.load(recovery_local_argmax_ptr + req_idx * recovery_local_argmax_stride + max_block_idx)
    tl.store(sampled_ptr + req_idx * sampled_stride + num_sampled, recovery)


def rejection_sample(
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    cu_num_logits: torch.Tensor,
    positions: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    acceptance_uniform: torch.Tensor,
    recovery_gumbel: torch.Tensor,
    num_speculative_steps: int,
    workspace: RejectionWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del positions
    assert draft_probs is None or draft_probs.is_contiguous()
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    assert draft_probs is None or draft_probs.shape == target_logits.shape

    vocab_block_size = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, vocab_block_size)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    recovery_block_size = 1024
    recovery_blocks = triton.cdiv(vocab_size, recovery_block_size)
    padded_recovery_blocks = triton.next_power_of_2(recovery_blocks)
    if workspace is None:
        workspace = RejectionWorkspace()
    (
        target_local_argmax,
        target_local_max,
        target_local_sumexp,
        sampled,
        num_sampled,
        recovery_local_argmax,
        recovery_local_max,
    ) = workspace.prepare(
        target_logits,
        num_reqs,
        num_logits,
        vocab_num_blocks,
        recovery_blocks,
        num_speculative_steps,
    )
    _target_stats_kernel[(num_logits, vocab_num_blocks)](
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
        draft_tokens,
        draft_probs,
        0 if draft_probs is None else draft_probs.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        acceptance_uniform,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_PROBS=draft_probs is not None,
        NUM_SPECULATIVE_STEPS=num_speculative_steps,
        num_warps=1,
    )

    _recovery_kernel[(num_reqs, recovery_blocks)](
        recovery_local_argmax,
        recovery_local_argmax.stride(0),
        recovery_local_max,
        recovery_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_tokens,
        draft_probs,
        0 if draft_probs is None else draft_probs.stride(0),
        temperature,
        recovery_gumbel,
        recovery_gumbel.stride(0),
        vocab_size,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_PROBS=draft_probs is not None,
        BLOCK_SIZE=recovery_block_size,
    )
    _insert_recovery_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        recovery_local_argmax,
        recovery_local_argmax.stride(0),
        recovery_local_max,
        recovery_local_max.stride(0),
        recovery_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RECOVERY_BLOCKS=padded_recovery_blocks,
    )
    return sampled, num_sampled


sample_with_rejection = rejection_sample
