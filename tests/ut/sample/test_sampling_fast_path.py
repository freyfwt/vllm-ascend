from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm_ascend.sample.sampler import AscendTopKTopPSampler, gumbel_sample


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return bool(torch.npu.is_available()) and torch.npu.device_count() > 0


def _sampling_metadata(device, max_num_logprobs=2):
    from vllm.v1.sample.metadata import SamplingMetadata

    return SamplingMetadata(
        temperature=torch.ones(2, dtype=torch.float32, device=device),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=max_num_logprobs,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(2, dtype=torch.float32, device=device),
        presence_penalties=torch.zeros(2, dtype=torch.float32, device=device),
        repetition_penalties=torch.ones(2, dtype=torch.float32, device=device),
        output_token_ids=[[], []],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=SimpleNamespace(non_argmax_invariant=[], argmax_invariant=[]),
    )


class _FakeBonusSampler:
    logprobs_mode = "raw_logprobs"

    def __init__(self, bonus_token_ids):
        self.bonus_token_ids = bonus_token_ids.view(-1, 1).to(torch.int32)

    def __call__(
        self,
        logits,
        sampling_metadata,
        predict_bonus_token=False,
        logprobs_mode_override=None,
    ):
        from vllm.v1.outputs import LogprobsTensors, SamplerOutput

        return SamplerOutput(
            sampled_token_ids=self.bonus_token_ids,
            logprobs_tensors=LogprobsTensors(
                torch.empty(0, dtype=torch.int32, device=logits.device),
                logits.to(torch.float32).clone(),
                torch.empty(0, dtype=torch.int32, device=logits.device),
            ),
        )

    @staticmethod
    def compute_logprobs(logits):
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def gather_logprobs(logprobs, num_logprobs, token_ids):
        from vllm.v1.outputs import LogprobsTensors

        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)
        token_ranks = (logprobs > token_logprobs).sum(dim=-1)
        return LogprobsTensors(
            torch.cat((token_ids, topk_indices), dim=1).to(torch.int32),
            torch.cat((token_logprobs, topk_logprobs), dim=1),
            token_ranks,
        )


def _fake_rejection_sampler(logprobs_mode="raw_logprobs"):
    sampler = AscendRejectionSampler.__new__(AscendRejectionSampler)
    sampler.sampler = SimpleNamespace(logprobs_mode=logprobs_mode)
    sampler.is_processed_logprobs_mode = logprobs_mode.startswith("processed")
    sampler.is_logits_logprobs_mode = logprobs_mode.endswith("logits")
    return sampler


def test_gumbel_sample_does_not_mutate_logits():
    logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
    original = logits.clone()
    gumbel = torch.tensor([[2.0, 0.0], [0.0, 4.0]])

    sampled = gumbel_sample(logits, gumbel)

    assert sampled.tolist() == [0, 1]
    assert logits.tolist() == original.tolist()


def test_topk_topp_sampler_uses_gumbel_without_polluting_processed_logits():
    sampler = AscendTopKTopPSampler(logprobs_mode="processed_logits")
    sampler.apply_top_k_top_p = lambda logits, k, p: logits
    logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
    gumbel = torch.tensor([[2.0, 0.0], [0.0, 4.0]])

    with (
        patch("vllm_ascend.sample.sampler.get_ascend_config") as get_config,
        patch.object(sampler, "_get_gumbel", return_value=gumbel),
    ):
        get_config.return_value = SimpleNamespace(
            enable_reduce_sample=False,
            enable_async_exponential=False,
        )
        sampled, logits_to_return = sampler.forward_native(logits, {}, None, None)

    assert sampled.tolist() == [0, 1]
    assert logits_to_return.tolist() == [[0.0, 1.0], [3.0, 0.0]]


def test_topk_topp_sampler_maps_reduce_sample_candidates_to_token_ids():
    sampler = AscendTopKTopPSampler(logprobs_mode="processed_logits")
    sampler.top_k = 2
    cand_logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
    cand_idx = torch.tensor([[11, 12], [21, 22]])
    gumbel = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    sampler.apply_top_k_top_p = lambda logits, k, p, top_k: (cand_logits, cand_idx)

    with (
        patch("vllm_ascend.sample.sampler.get_ascend_config") as get_config,
        patch.object(sampler, "_get_gumbel", return_value=gumbel),
    ):
        get_config.return_value = SimpleNamespace(
            enable_reduce_sample=True,
            enable_async_exponential=False,
        )
        sampled, logits_to_return = sampler.forward_native(torch.empty_like(cand_logits), {}, None, None)

    assert sampled.tolist() == [11, 22]
    assert logits_to_return.tolist() == cand_logits.tolist()


def test_rejection_sampler_optimized_gate_keeps_async_exponential_fallback():
    sampling_metadata = SimpleNamespace(all_greedy=False)
    sampler = _fake_rejection_sampler()

    with patch("vllm_ascend.sample.rejection_sampler.get_ascend_config") as get_config:
        get_config.return_value = SimpleNamespace(
            enable_async_exponential=True,
            enable_reduce_sample=False,
        )
        assert not sampler._can_use_optimized_rejection(
            sampling_metadata,
        )


def test_rejection_sampler_optimized_gate_accepts_logprobs_and_reduce_sample():
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        max_num_logprobs=1,
        logprob_token_ids={0: [7]},
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=SimpleNamespace(non_argmax_invariant=[], argmax_invariant=[]),
    )
    sampler = _fake_rejection_sampler()

    with patch("vllm_ascend.sample.rejection_sampler.get_ascend_config") as get_config:
        get_config.return_value = SimpleNamespace(
            enable_async_exponential=False,
            enable_reduce_sample=True,
        )
        assert sampler._can_use_optimized_rejection(
            sampling_metadata,
        )


def test_rejection_sampler_optimized_gate_rejects_all_greedy():
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=SimpleNamespace(non_argmax_invariant=[], argmax_invariant=[]),
    )
    sampler = _fake_rejection_sampler()

    with patch("vllm_ascend.sample.rejection_sampler.get_ascend_config") as get_config:
        get_config.return_value = SimpleNamespace(
            enable_async_exponential=False,
            enable_reduce_sample=False,
        )
        assert not sampler._can_use_optimized_rejection(
            sampling_metadata,
        )


@pytest.mark.skipif(not _npu_available(), reason="requires NPU")
def test_topk_topp_sampler_npu_preserves_reduce_sample_logprobs():
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    sampler = AscendTopKTopPSampler(logprobs_mode="processed_logprobs")
    sampler.top_k = 2
    cand_logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]], dtype=torch.float32, device=device)
    cand_idx = torch.tensor([[11, 12], [21, 22]], dtype=torch.int64, device=device)
    gumbel = torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=torch.float32, device=device)
    sampler.apply_top_k_top_p = lambda logits, k, p, top_k: (cand_logits, cand_idx)

    with (
        patch("vllm_ascend.sample.sampler.get_ascend_config") as get_config,
        patch.object(sampler, "_get_gumbel", return_value=gumbel),
    ):
        get_config.return_value = SimpleNamespace(
            enable_reduce_sample=True,
            enable_async_exponential=False,
        )
        sampled, logits_to_return = sampler.forward_native(
            torch.empty((2, 4), dtype=torch.float32, device=device),
            {},
            None,
            None,
        )

    torch.npu.synchronize()
    assert sampled.cpu().tolist() == [11, 22]
    torch.testing.assert_close(
        logits_to_return.cpu(),
        cand_logits.log_softmax(dim=-1, dtype=torch.float32).cpu(),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not _npu_available(), reason="requires NPU")
def test_rejection_sampler_npu_matches_fallback_with_reduce_sample_logprobs():
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    torch.npu.set_device(0)
    init_device_properties_triton()
    device = torch.device("npu:0")

    vocab_size = 32
    logits = torch.full((4, vocab_size), -20.0, dtype=torch.float32, device=device)
    logits[0, 11] = 40.0
    logits[0, 12] = 1.0
    logits[1, 14] = 35.0
    logits[2, 21] = 42.0
    logits[2, 22] = 2.0
    logits[3, 23] = 36.0

    candidate_logits_all = torch.tensor(
        [
            [40.0, 1.0, -2.0],
            [35.0, 1.0, -2.0],
            [42.0, 2.0, -3.0],
            [36.0, 1.0, -2.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    candidate_indices_all = torch.tensor(
        [
            [11, 12, 13],
            [14, 12, 13],
            [21, 22, 24],
            [23, 22, 24],
        ],
        dtype=torch.int64,
        device=device,
    )
    candidate_logits_draft = candidate_logits_all[[0, 2]]
    candidate_indices_draft = candidate_indices_all[[0, 2]]

    metadata = SpecDecodeMetadata(
        draft_token_ids=torch.tensor([11, 21], dtype=torch.int32, device=device),
        num_draft_tokens=[1, 1],
        cu_num_draft_tokens=torch.tensor([1, 2], dtype=torch.int32, device=device),
        cu_num_sampled_tokens=torch.tensor([2, 4], dtype=torch.int32, device=device),
        target_logits_indices=torch.tensor([0, 2], dtype=torch.int32, device=device),
        bonus_logits_indices=torch.tensor([1, 3], dtype=torch.int32, device=device),
        logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device),
    )
    sampling_metadata = _sampling_metadata(device)
    bonus_token_ids = torch.tensor([14, 23], dtype=torch.int32, device=device)

    def fake_apply_sampling_constraints(logits, cu_num_draft_tokens, sampling_metadata, top_k):
        if logits.shape[0] == candidate_logits_all.shape[0]:
            return candidate_logits_all.clone(), candidate_indices_all
        return candidate_logits_draft.clone(), candidate_indices_draft

    config = SimpleNamespace(enable_reduce_sample=True, enable_async_exponential=False)
    fallback_config = SimpleNamespace(enable_reduce_sample=True, enable_async_exponential=True)
    uniform = torch.full((2,), 1e-4, dtype=torch.float64, device=device)
    acceptance_uniform = torch.full((2,), 1e-4, dtype=torch.float32, device=device)
    recovery_gumbel = torch.zeros((2, 3), dtype=torch.float32, device=device)

    optimized_sampler = AscendRejectionSampler(_FakeBonusSampler(bonus_token_ids))
    fallback_sampler = AscendRejectionSampler(_FakeBonusSampler(bonus_token_ids))

    with (
        patch("vllm_ascend.sample.rejection_sampler.get_ascend_config", return_value=config),
        patch("vllm_ascend.sample.rejection_sampler.apply_sampling_constraints", fake_apply_sampling_constraints),
        patch.object(
            optimized_sampler,
            "_get_rejection_random",
            return_value=(acceptance_uniform, recovery_gumbel),
        ),
    ):
        optimized = optimized_sampler(
            metadata,
            None,
            logits.clone(),
            sampling_metadata,
        )

    with (
        patch("vllm_ascend.sample.rejection_sampler.get_ascend_config", return_value=fallback_config),
        patch("vllm_ascend.sample.rejection_sampler.apply_sampling_constraints", fake_apply_sampling_constraints),
        patch("vllm_ascend.sample.rejection_sampler.generate_uniform_probs", return_value=uniform),
    ):
        fallback = fallback_sampler(
            metadata,
            None,
            logits.clone(),
            sampling_metadata,
        )

    torch.npu.synchronize()
    assert optimized.sampled_token_ids.cpu().tolist() == [[11, 14], [21, 23]]
    assert optimized.sampled_token_ids.cpu().tolist() == fallback.sampled_token_ids.cpu().tolist()
    assert optimized.logprobs_tensors is not None
    assert fallback.logprobs_tensors is not None
    torch.testing.assert_close(
        optimized.logprobs_tensors.logprob_token_ids.cpu(),
        fallback.logprobs_tensors.logprob_token_ids.cpu(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        optimized.logprobs_tensors.logprobs.cpu(),
        fallback.logprobs_tensors.logprobs.cpu(),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        optimized.logprobs_tensors.selected_token_ranks.cpu(),
        fallback.logprobs_tensors.selected_token_ranks.cpu(),
        rtol=0,
        atol=0,
    )
