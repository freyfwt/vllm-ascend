from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.sample.sampling_bridge import (
    FastSampler,
    NoiseManager,
    SamplingNoise,
    apply_regular_sampling_params,
    sample_logits,
)


def test_sample_logits_modes():
    logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
    gumbel = torch.tensor([[2.0, 0.0], [0.0, 4.0]])

    assert sample_logits(logits, torch.ones(2), gumbel).tolist() == [0, 1]
    assert sample_logits(logits, torch.tensor([0.0, 1.0]), gumbel).tolist() == [1, 1]
    assert sample_logits(logits, torch.empty(0), gumbel, all_random=True).tolist() == [0, 1]
    assert sample_logits(logits, torch.zeros(2), None, all_greedy=True).tolist() == [1, 0]


def test_sample_logits_inplace_gumbel_is_opt_in():
    logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
    original = logits.clone()
    gumbel = torch.tensor([[2.0, 0.0], [0.0, 4.0]])

    out = sample_logits(logits, torch.empty(0), gumbel, all_random=True)
    assert out.tolist() == [0, 1]
    assert logits.tolist() == original.tolist()

    out = sample_logits(
        logits,
        torch.empty(0),
        gumbel,
        all_random=True,
        add_gumbel_inplace=True,
    )
    assert out.tolist() == [0, 1]
    assert logits.tolist() == (original + gumbel).tolist()


def test_apply_regular_sampling_params_reuses_ascend_sampler_processors():
    sampler = _FakeSampler()
    metadata = _metadata(
        temperature=torch.tensor([2.0, 2.0]),
        all_greedy=False,
        all_random=True,
        top_k=torch.tensor([2, 2], dtype=torch.int32),
        top_p=torch.tensor([1.0, 1.0]),
    )
    logits = torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=torch.float16)

    with patch(
        "vllm_ascend.sample.sampling_bridge.npu_apply_top_k_top_p",
        side_effect=lambda logits, top_k, top_p: logits.add_(3.0),
    ) as apply_top_k_top_p:
        processed = apply_regular_sampling_params(logits, sampler, metadata)

    assert processed.dtype == torch.float32
    assert processed.tolist() == [[4.5, 5.5], [6.5, 7.5]]
    assert sampler.predict_bonus_token_calls == [False]
    apply_top_k_top_p.assert_called_once()
    _, top_k, top_p = apply_top_k_top_p.call_args.args
    assert top_k is metadata.top_k
    assert top_p is metadata.top_p


def test_apply_regular_sampling_params_all_greedy_skips_random_only_processors():
    sampler = _FakeSampler()
    metadata = _metadata(
        temperature=torch.zeros(2),
        all_greedy=True,
        all_random=False,
        top_k=None,
        top_p=None,
    )
    logits = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

    with patch("vllm_ascend.sample.sampling_bridge.npu_apply_top_k_top_p") as apply_top_k_top_p:
        processed = apply_regular_sampling_params(logits, sampler, metadata, predict_bonus_token=True)

    assert processed.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert sampler.predict_bonus_token_calls == [True]
    assert metadata.logitsprocs.argmax_invariant[0].calls == 0
    apply_top_k_top_p.assert_not_called()


def test_fast_sampler_regular_returns_sampler_output_without_bridge_state():
    fast_sampler = FastSampler(
        max_num_reqs=2,
        device=torch.device("cpu"),
        speculative_config=None,
    )
    sampler = _FakeSampler()
    metadata = _metadata(
        temperature=torch.ones(2),
        all_greedy=False,
        all_random=True,
        top_k=None,
        top_p=None,
    )
    logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
    gumbel = torch.tensor([[2.0, 0.0], [0.0, 4.0]])

    with patch(
        "vllm_ascend.sample.sampling_bridge.npu_apply_top_k_top_p",
        side_effect=lambda logits, top_k, top_p: logits,
    ):
        output = fast_sampler.sample_regular(logits, sampler, metadata, gumbel)

    assert output.sampled_token_ids.tolist() == [[0], [1]]
    assert output.logprobs_tensors is None


def test_noise_manager_all_greedy_returns_empty_noise_without_buffers():
    manager = NoiseManager.__new__(NoiseManager)
    metadata = _metadata(
        temperature=torch.zeros(2),
        all_greedy=True,
        all_random=False,
        top_k=None,
        top_p=None,
    )

    regular_noise = manager.prepare_regular_noise(2, 8, metadata)
    spec_noise = manager.prepare_spec_noise(4, 2, 8, [1, 1], metadata)

    assert regular_noise == SamplingNoise()
    assert spec_noise == SamplingNoise()


class _FakeArgmaxInvariantProcessor:
    def __init__(self):
        self.calls = 0

    def apply(self, logits):
        self.calls += 1
        return logits.add_(1.0)


class _FakeSampler:
    def __init__(self):
        self.predict_bonus_token_calls = []

    def apply_logits_processors(self, logits, sampling_metadata, predict_bonus_token):
        del sampling_metadata
        self.predict_bonus_token_calls.append(predict_bonus_token)
        return logits.add_(1.0)

    @staticmethod
    def apply_temperature(logits, temperature, all_random):
        del all_random
        return logits.div_(temperature.unsqueeze(1))


def _metadata(
    *,
    temperature,
    all_greedy,
    all_random,
    top_k,
    top_p,
):
    return SimpleNamespace(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=all_random,
        top_k=top_k,
        top_p=top_p,
        generators={},
        logitsprocs=SimpleNamespace(
            argmax_invariant=[_FakeArgmaxInvariantProcessor()],
        ),
    )
