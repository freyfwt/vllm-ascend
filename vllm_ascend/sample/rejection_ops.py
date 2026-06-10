# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.probabilistic_rejection_sampler_utils import (
    _compute_block_max_and_sumexp,
    _compute_global_lse,
)

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
def _compute_block_stats_kernel(
    target_local_argmax_ptr,
    target_local_argmax_stride,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    target_logits_ptr,
    target_logits_stride,
    target_indices_ptr,
    target_indices_stride,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    temperature_ptr,
    vocab_size,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    HAS_TARGET_INDICES: tl.constexpr,
):
    logit_idx = tl.program_id(0)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    if draft_step_idx >= num_speculative_steps:
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    target_logits = tl.load(
        target_logits_ptr + logit_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    if temperature == 0.0:
        value, idx = tl.max(target_logits, axis=0, return_indices=True)
        argmax_idx = block_idx * BLOCK_SIZE + idx
        if HAS_TARGET_INDICES:
            argmax_idx = tl.load(target_indices_ptr + logit_idx * target_indices_stride + argmax_idx)
        tl.store(
            target_local_argmax_ptr + logit_idx * target_local_argmax_stride + block_idx,
            argmax_idx,
        )
        tl.store(
            target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
            value,
        )
        return

    target_max, target_sumexp = _compute_block_max_and_sumexp(target_logits)
    tl.store(
        target_local_max_ptr + logit_idx * target_local_max_stride + block_idx,
        target_max,
    )
    tl.store(
        target_local_sumexp_ptr + logit_idx * target_local_sumexp_stride + block_idx,
        target_sumexp,
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
    target_indices_ptr,
    target_indices_stride,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    temperature_ptr,
    acceptance_uniform_ptr,
    vocab_num_blocks,
    vocab_size,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    TARGET_INDEX_BLOCK_SIZE: tl.constexpr,
    HAS_TARGET_INDICES: tl.constexpr,
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
    draft_start_idx = start_idx - req_idx
    for i in range(NUM_SPECULATIVE_STEPS):
        if accepted and i < num_draft_tokens:
            logit_idx = start_idx + i
            draft_tokens = tl.load(draft_tokens_ptr + draft_start_idx + i)
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
                if HAS_TARGET_INDICES:
                    candidates = tl.arange(0, TARGET_INDEX_BLOCK_SIZE)
                    candidate_mask = candidates < vocab_size
                    target_ids = tl.load(
                        target_indices_ptr + logit_idx * target_indices_stride + candidates,
                        mask=candidate_mask,
                        other=-1,
                    )
                    target_logits = tl.load(
                        target_logits_ptr + logit_idx * target_logits_stride + candidates,
                        mask=candidate_mask & (target_ids == draft_tokens),
                        other=float("-inf"),
                    ).to(tl.float32)
                    target_logit = tl.max(target_logits, axis=0)
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
    num_sampled_ptr,
    cu_num_logits_ptr,
    expanded_idx_mapping_ptr,
    draft_tokens_ptr,
    target_indices_ptr,
    target_indices_stride,
    temperature_ptr,
    recovery_gumbel_ptr,
    recovery_gumbel_stride,
    vocab_size,
    HAS_TARGET_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    recovery_idx = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    if recovery_idx > num_draft_tokens:
        return
    recovery_token_idx = start_idx + recovery_idx
    req_state_idx = tl.load(expanded_idx_mapping_ptr + recovery_token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    is_bonus = recovery_idx == num_draft_tokens
    if temperature == 0.0 and not is_bonus:
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    token_ids = block
    if HAS_TARGET_INDICES:
        token_ids = tl.load(
            target_indices_ptr + recovery_token_idx * target_indices_stride + block,
            mask=mask,
            other=-1,
        )
    logits = tl.load(
        target_logits_ptr + recovery_token_idx * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    if not is_bonus:
        draft_start_idx = start_idx - req_idx
        rejected_draft = tl.load(draft_tokens_ptr + draft_start_idx + recovery_idx)
        logits = tl.where(token_ids != rejected_draft, logits, float("-inf"))
    if temperature != 0.0:
        logits += tl.load(
            recovery_gumbel_ptr + req_idx * recovery_gumbel_stride + block,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)

    value, idx = tl.max(logits, axis=0, return_indices=True)
    argmax_idx = block_idx * BLOCK_SIZE + idx
    if HAS_TARGET_INDICES:
        argmax_idx = tl.load(target_indices_ptr + recovery_token_idx * target_indices_stride + argmax_idx)
    tl.store(
        recovery_local_argmax_ptr + req_idx * recovery_local_argmax_stride + block_idx,
        argmax_idx,
    )
    tl.store(
        recovery_local_max_ptr + req_idx * recovery_local_max_stride + block_idx,
        value,
    )


@triton.jit
def _insert_resampled_or_bonus_kernel(
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
    num_draft_tokens = end_idx - start_idx - 1

    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)
    recovery_token_idx = start_idx + num_sampled
    req_state_idx = tl.load(expanded_idx_mapping_ptr + recovery_token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    is_bonus = num_sampled == num_draft_tokens
    if temperature == 0.0 and not is_bonus:
        return

    blocks = tl.arange(0, PADDED_RECOVERY_BLOCKS)
    mask = blocks < recovery_blocks
    recovery_local_max = tl.load(
        recovery_local_max_ptr + req_idx * recovery_local_max_stride + blocks,
        mask=mask,
        other=float("-inf"),
    )
    recovery_block_idx = tl.argmax(recovery_local_max, axis=0)
    recovered = tl.load(
        recovery_local_argmax_ptr + req_idx * recovery_local_argmax_stride + recovery_block_idx,
    )
    tl.store(
        sampled_ptr + req_idx * sampled_stride + num_sampled,
        recovered,
    )


def rejection_sample(
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    target_indices: torch.Tensor | None,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    acceptance_uniform: torch.Tensor,
    recovery_gumbel: torch.Tensor,
    num_speculative_steps: int,
    workspace: RejectionWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from draft+bonus logits rows using compact draft token ids."""
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
    assert target_indices is None or target_indices.shape == target_logits.shape
    assert draft_tokens.ndim == 1
    assert draft_tokens.shape[0] == num_logits - num_reqs

    vocab_block_size = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, vocab_block_size)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    target_index_block_size = triton.next_power_of_2(vocab_size) if target_indices is not None else 1
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
    _compute_block_stats_kernel[(num_logits, vocab_num_blocks)](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        target_indices,
        0 if target_indices is None else target_indices.stride(0),
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_speculative_steps,
        BLOCK_SIZE=vocab_block_size,
        HAS_TARGET_INDICES=target_indices is not None,
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
        target_indices,
        0 if target_indices is None else target_indices.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        acceptance_uniform,
        vocab_num_blocks,
        vocab_size,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        TARGET_INDEX_BLOCK_SIZE=target_index_block_size,
        HAS_TARGET_INDICES=target_indices is not None,
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
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_tokens,
        target_indices,
        0 if target_indices is None else target_indices.stride(0),
        temperature,
        recovery_gumbel,
        recovery_gumbel.stride(0),
        vocab_size,
        HAS_TARGET_INDICES=target_indices is not None,
        BLOCK_SIZE=recovery_block_size,
    )
    _insert_resampled_or_bonus_kernel[(num_reqs,)](
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
