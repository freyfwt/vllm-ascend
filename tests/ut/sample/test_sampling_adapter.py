import unittest
from types import SimpleNamespace

import torch

from vllm_ascend.worker.v1.sample.adapter import (
    sample_processed_logits,
    temperature_for_sampling,
)


def _metadata(**overrides):
    values = dict(
        temperature=torch.ones(2, dtype=torch.float32),
        all_greedy=False,
        all_random=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestSamplingAdapter(unittest.TestCase):
    def test_sample_processed_logits_uses_external_gumbel(self):
        processed_logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
        sampling_gumbel = torch.tensor([[2.0, 0.0], [0.0, 4.0]])

        sampled = sample_processed_logits(
            processed_logits,
            _metadata(),
            sampling_gumbel,
            greedy_tokens=None,
        )

        self.assertEqual(sampled.tolist(), [0, 1])

    def test_sample_processed_logits_preserves_greedy_rows(self):
        processed_logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
        sampling_gumbel = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
        greedy_tokens = torch.tensor([1, 0])

        sampled = sample_processed_logits(
            processed_logits,
            _metadata(
                temperature=torch.tensor([0.0, 1.0], dtype=torch.float32),
                all_random=False,
            ),
            sampling_gumbel,
            greedy_tokens,
        )

        self.assertEqual(sampled.tolist(), [1, 1])

    def test_temperature_for_sampling_returns_greedy_temperature_when_absent(self):
        temperature = temperature_for_sampling(
            _metadata(temperature=None, all_greedy=True, all_random=False),
            num_reqs=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(temperature.tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
