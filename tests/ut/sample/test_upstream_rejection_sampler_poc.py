import time
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

QWEN_VOCAB_SIZE = 151936
REPRESENTATIVE_BATCH_SIZES = (1, 4, 16, 64, 128)
NUM_SPEC_TOKENS = 4
WARMUP_ITERS = 5
NUM_ITERS = 20
HIGH_LOGIT = 40.0
LOW_LOGIT = -40.0


class _ReadyEvent:
    def synchronize(self):
        pass


def _npu_available() -> bool:
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _npu_available(), reason="NPU performance POC test")
@pytest.mark.parametrize(
    "num_reqs",
    REPRESENTATIVE_BATCH_SIZES,
    ids=[f"bs{batch_size}" for batch_size in REPRESENTATIVE_BATCH_SIZES],
)
def test_upstream_rejection_sampler_poc_perf_against_ascend_golden(num_reqs):
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.sample import rejection_sampler as rejection_sampler_module
    from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
    from vllm_ascend.sample.sampler import AscendSampler, _apply_top_k_top_p_pytorch
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    device = torch.device("npu")
    init_device_properties_triton()
    import_module("vllm_ascend.patch.worker.patch_v2.patch_triton")
    num_spec_tokens = NUM_SPEC_TOKENS
    vocab_size = QWEN_VOCAB_SIZE

    num_draft_tokens_np = torch.full((num_reqs,), num_spec_tokens, dtype=torch.int32).numpy()
    num_logits_per_req = num_draft_tokens_np + 1
    cu_num_sampled_np = num_logits_per_req.cumsum(dtype="int32")
    cu_num_draft_np = num_draft_tokens_np.cumsum(dtype="int32")
    total_num_logits = int(cu_num_sampled_np[-1])
    total_num_draft_tokens = int(cu_num_draft_np[-1])

    logits = torch.full((total_num_logits, vocab_size), LOW_LOGIT, device=device)
    draft_logits = torch.full(
        (num_reqs, num_spec_tokens, vocab_size),
        LOW_LOGIT,
        device=device,
    )
    input_ids = torch.zeros(total_num_logits, dtype=torch.int32, device=device)
    draft_token_ids = torch.empty(total_num_draft_tokens, dtype=torch.int32, device=device)
    bonus_token_ids = torch.empty(num_reqs, dtype=torch.int32, device=device)

    target_logits_indices = []
    bonus_logits_indices = []
    draft_offset = 0
    for req_idx in range(num_reqs):
        row_base = req_idx * (num_spec_tokens + 1)
        for step in range(num_spec_tokens):
            row = row_base + step
            token_id = (req_idx * 97 + step * 31 + 7) % vocab_size
            input_ids[row + 1] = token_id
            draft_token_ids[draft_offset] = token_id
            logits[row, token_id] = HIGH_LOGIT
            draft_logits[req_idx, step, token_id] = HIGH_LOGIT
            target_logits_indices.append(row)
            draft_offset += 1
        bonus_row = row_base + num_spec_tokens
        bonus_id = (req_idx * 193 + 11) % vocab_size
        logits[bonus_row, bonus_id] = HIGH_LOGIT
        bonus_token_ids[req_idx] = bonus_id
        bonus_logits_indices.append(bonus_row)

    spec_decode_metadata = SpecDecodeMetadata(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens_np.tolist(),
        cu_num_draft_tokens=torch.from_numpy(cu_num_draft_np).to(device),
        cu_num_sampled_tokens=torch.from_numpy(cu_num_sampled_np).to(device),
        target_logits_indices=torch.tensor(target_logits_indices, dtype=torch.int64, device=device),
        bonus_logits_indices=torch.tensor(bonus_logits_indices, dtype=torch.int64, device=device),
        logits_indices=torch.arange(total_num_logits, dtype=torch.int64, device=device),
    )
    sampling_metadata = SamplingMetadata(
        temperature=torch.ones(num_reqs, dtype=torch.float32, device=device),
        all_greedy=False,
        all_random=True,
        top_p=torch.full((num_reqs,), 0.95, dtype=torch.float32, device=device),
        top_k=torch.full((num_reqs,), 32, dtype=torch.int32, device=device),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(num_reqs, dtype=torch.float32, device=device),
        presence_penalties=torch.zeros(num_reqs, dtype=torch.float32, device=device),
        repetition_penalties=torch.ones(num_reqs, dtype=torch.float32, device=device),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        spec_token_ids=[[] for _ in range(num_reqs)],
    )

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = device
    runner.speculative_config = SimpleNamespace(
        rejection_sample_method="probabilistic",
        num_speculative_tokens=num_spec_tokens,
    )
    runner.input_batch = SimpleNamespace(
        sampling_metadata=sampling_metadata,
        num_reqs=num_reqs,
        top_k_cpu=torch.full((num_reqs,), 32, dtype=torch.int32),
        update_async_output_token_ids=lambda: None,
    )
    runner.input_ids = SimpleNamespace(gpu=input_ids)
    runner.positions = torch.arange(total_num_logits, dtype=torch.int64, device=device)
    runner.sampler = AscendSampler()
    runner.sampler.topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_pytorch
    runner.rejection_sampler = AscendRejectionSampler(runner.sampler)
    old_bonus_q = torch.empty((num_reqs, vocab_size), dtype=torch.float32, device=device)
    old_bonus_q.exponential_()
    runner.sampler.topk_topp_sampler.set_q_event(old_bonus_q, _ReadyEvent())
    old_uniform_probs = torch.full(
        (total_num_draft_tokens,),
        0.5,
        dtype=torch.float64,
        device=device,
    )
    old_recovered_q = torch.empty((num_reqs, vocab_size), dtype=torch.float32, device=device)
    old_recovered_q.exponential_()
    runner.upstream_rejection_sampler_poc_seeds = torch.arange(
        1,
        num_reqs + 1,
        dtype=torch.int64,
        device=device,
    )
    runner.upstream_rejection_sampler_poc_accept_uniform = torch.full(
        (total_num_logits,),
        0.5,
        dtype=torch.float32,
        device=device,
    )
    runner.upstream_rejection_sampler_poc_resample_gumbel = torch.empty(
        (num_reqs, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    runner.upstream_rejection_sampler_poc_resample_gumbel.exponential_()
    runner.upstream_rejection_sampler_poc_resample_gumbel.log_().neg_()
    runner.draft_logits_poc = draft_logits

    config = SimpleNamespace(
        enable_async_exponential=True,
        enable_reduce_sample=False,
    )

    def generate_uniform_probs_async_poc(*args, **kwargs):
        return old_uniform_probs

    def sample_recovered_tokens_async_poc(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
        use_block_verify=False,
        target_indices=None,
        global_vocab_size=None,
        enable_reduce_sampling=False,
    ):
        batch_size = len(num_draft_tokens)
        vocab_size = target_probs.shape[-1]
        recovered_token_ids = torch.empty_like(draft_token_ids)
        rejection_sampler_module.sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
            recovered_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            target_indices,
            old_recovered_q[:batch_size, :vocab_size],
            vocab_size,
            global_vocab_size if global_vocab_size is not None else vocab_size,
            NO_DRAFT_PROBS=draft_probs is None,
            BLOCK_VERIFY=use_block_verify,
            ENABLE_REDUCE_SAMPLING=enable_reduce_sampling,
            SUB_BLOCK=512,
            multibuffer=False,
        )
        return recovered_token_ids

    def run_old():
        runner.use_upstream_rejection_sampler_poc = False
        return runner._sample(logits.clone(), spec_decode_metadata).sampled_token_ids

    def run_new():
        runner.use_upstream_rejection_sampler_poc = True
        return runner._sample(logits.clone(), spec_decode_metadata).sampled_token_ids

    def measure(fn):
        for _ in range(WARMUP_ITERS):
            fn()
        torch.npu.synchronize()
        start = time.perf_counter()
        out = None
        for _ in range(NUM_ITERS):
            out = fn()
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000 / NUM_ITERS
        return elapsed_ms, out

    with (
        patch("vllm_ascend.sample.sampler.get_ascend_config", return_value=config),
        patch("vllm_ascend.sample.rejection_sampler.get_ascend_config", return_value=config),
        patch("vllm_ascend.sample.rejection_sampler.apply_top_k_top_p", new=_apply_top_k_top_p_pytorch),
        patch(
            "vllm_ascend.sample.rejection_sampler.generate_uniform_probs",
            new=generate_uniform_probs_async_poc,
        ),
        patch(
            "vllm_ascend.sample.rejection_sampler.sample_recovered_tokens",
            new=sample_recovered_tokens_async_poc,
        ),
        patch("vllm_ascend.worker.model_runner_v1.get_ascend_config", return_value=config),
        patch("vllm_ascend.worker.model_runner_v1.apply_top_k_top_p", new=_apply_top_k_top_p_pytorch),
        patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable", return_value=False),
    ):
        old_ms, old_out = measure(run_old)
        new_ms, new_out = measure(run_new)

    assert torch.equal(new_out, old_out)
    print(
        "upstream_rejection_sampler_poc "
        f"batch_size={num_reqs} vocab_size={vocab_size} "
        f"warmup_iters={WARMUP_ITERS} num_iters={NUM_ITERS} "
        "old_async_random=True new_external_random=True "
        f"old_ms={old_ms:.3f} new_ms={new_ms:.3f} "
        f"speedup={old_ms / new_ms:.3f}x"
    )
