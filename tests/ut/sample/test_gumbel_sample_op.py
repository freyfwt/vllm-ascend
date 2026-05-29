import unittest

import torch

from tests.ut.base import TestBase


def _npu_custom_gumbel_available() -> bool:
    if not hasattr(torch, "npu"):
        return False
    try:
        if not torch.npu.is_available():
            return False
        namespace = getattr(torch.ops, "_C_ascend", None)
        return namespace is not None and hasattr(namespace, "npu_gumbel_sample")
    except RuntimeError:
        return False


@unittest.skipUnless(
    _npu_custom_gumbel_available(),
    "NPU and custom gumbel sampling op are required",
)
class TestGumbelSampleOp(TestBase):
    def test_greedy_rows_match_argmax(self):
        device = torch.device("npu:0")
        logits = torch.tensor(
            [[1.0, 9.0, 0.0], [3.0, 2.0, 8.0]],
            dtype=torch.float32,
            device=device,
        )
        idx_mapping = torch.arange(2, dtype=torch.int32, device=device)
        temperature = torch.zeros(2, dtype=torch.float32, device=device)
        seeds = torch.tensor([11, 17], dtype=torch.int64, device=device)
        positions = torch.tensor([0, 1], dtype=torch.int64, device=device)

        out = torch.ops._C_ascend.npu_gumbel_sample(logits, idx_mapping, temperature, seeds, positions, True)

        self.assertEqual(out.dtype, torch.int32)
        self.assertTrue(torch.equal(out, logits.argmax(dim=-1).to(torch.int32)))

    def test_deterministic_for_same_counter_inputs(self):
        device = torch.device("npu:0")
        logits = torch.randn(4, 32, dtype=torch.float32, device=device)
        idx_mapping = torch.arange(4, dtype=torch.int32, device=device)
        temperature = torch.ones(4, dtype=torch.float32, device=device)
        seeds = torch.arange(100, 104, dtype=torch.int64, device=device)
        positions = torch.arange(4, dtype=torch.int64, device=device)

        out1 = torch.ops._C_ascend.npu_gumbel_sample(logits, idx_mapping, temperature, seeds, positions, True)
        out2 = torch.ops._C_ascend.npu_gumbel_sample(logits, idx_mapping, temperature, seeds, positions, True)

        self.assertTrue(torch.equal(out1, out2))

    def test_apply_temperature_matches_predivided_logits(self):
        device = torch.device("npu:0")
        logits = torch.randn(3, 64, dtype=torch.float32, device=device)
        idx_mapping = torch.arange(3, dtype=torch.int32, device=device)
        temperature = torch.tensor([0.7, 1.0, 1.3], dtype=torch.float32, device=device)
        seeds = torch.arange(77, 80, dtype=torch.int64, device=device)
        positions = torch.arange(3, dtype=torch.int64, device=device)

        out_scaled_in_op = torch.ops._C_ascend.npu_gumbel_sample(
            logits, idx_mapping, temperature, seeds, positions, True
        )
        out_predivided = torch.ops._C_ascend.npu_gumbel_sample(
            logits / temperature.unsqueeze(1),
            idx_mapping,
            temperature,
            seeds,
            positions,
            False,
        )

        self.assertTrue(torch.equal(out_scaled_in_op, out_predivided))

    def test_top_k_candidate_path_returns_original_ids(self):
        device = torch.device("npu:0")
        logits = torch.tensor(
            [[0.1, 5.0, 1.0, 4.0], [3.0, 2.0, 8.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        idx_mapping = torch.arange(2, dtype=torch.int32, device=device)
        temperature = torch.ones(2, dtype=torch.float32, device=device)
        top_k = torch.tensor([2, 2], dtype=torch.int32, device=device)

        candidate_logits, candidate_ids, candidate_lens, status = torch.ops._C_ascend.npu_build_top_k_top_p_candidates(
            logits,
            idx_mapping,
            temperature,
            k=top_k,
            candidate_capacity=2,
            apply_temperature=True,
        )

        self.assertTrue(torch.equal(status.cpu(), torch.zeros(2, dtype=torch.int32)))
        self.assertTrue(torch.equal(candidate_lens.cpu(), torch.full((2,), 2, dtype=torch.int32)))
        self.assertEqual(candidate_ids.cpu().tolist(), [[1, 3], [2, 0]])

        seeds = torch.arange(7, 9, dtype=torch.int64, device=device)
        positions = torch.arange(2, dtype=torch.int64, device=device)
        out = torch.ops._C_ascend.npu_gumbel_sample_from_candidates(
            candidate_logits,
            candidate_ids,
            candidate_lens,
            idx_mapping,
            seeds,
            positions,
        )
        self.assertTrue(bool(torch.isin(out, candidate_ids.reshape(-1)).all().item()))


if __name__ == "__main__":
    unittest.main()
