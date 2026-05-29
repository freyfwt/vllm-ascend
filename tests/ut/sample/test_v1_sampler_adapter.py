import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.v1.outputs import SamplerOutput

from tests.ut.base import TestBase


def _metadata(**overrides):
    values = dict(
        temperature=torch.ones(3, dtype=torch.float32),
        generators={},
        max_num_logprobs=None,
        logprob_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        top_k=None,
        top_p=None,
        logitsprocs=SimpleNamespace(
            non_argmax_invariant=(),
            argmax_invariant=(),
        ),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _ctx(req_ids=("req0", "req1", "req2")):
    from vllm_ascend.worker.v1.sample.sampling_context import V1SamplingContext

    return V1SamplingContext.from_model_runner_inputs(
        num_reqs=len(req_ids),
        positions_at_logits=torch.arange(len(req_ids), dtype=torch.int64),
        input_ids_at_logits=torch.arange(100, 100 + len(req_ids), dtype=torch.int64),
        req_indices_at_logits=torch.arange(len(req_ids), dtype=torch.int32),
        device=torch.device("cpu"),
        req_ids=req_ids,
    )


def _ctx_without_req_ids():
    from vllm_ascend.worker.v1.sample.sampling_context import V1SamplingContext

    return V1SamplingContext.from_model_runner_inputs(
        num_reqs=2,
        positions_at_logits=torch.tensor([0, 1], dtype=torch.int64),
        input_ids_at_logits=torch.tensor([10, 11], dtype=torch.int64),
        req_indices_at_logits=torch.tensor([0, 1], dtype=torch.int32),
        device=torch.device("cpu"),
    )


class TestV1SamplerAdapter(TestBase):
    def test_returns_sampler_output_for_normal_decode(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter

        adapter = V1SamplerAdapter(
            max_num_reqs=4,
            device=torch.device("cpu"),
        )
        logits = torch.tensor(
            [
                [1.0, 9.0, 0.0],
                [3.0, 2.0, 8.0],
                [7.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        with patch(
            "vllm_ascend.worker.v2.sample.gumbel.gumbel_sample",
            return_value=torch.tensor([1, 2, 0], dtype=torch.int64),
        ) as mock_gumbel_sample:
            output = adapter(
                logits=logits,
                sampling_metadata=_metadata(temperature=None),
                ctx=_ctx(),
            )

        self.assertIsInstance(output, SamplerOutput)
        self.assertEqual(output.sampled_token_ids.dtype, torch.int32)
        self.assertEqual(output.sampled_token_ids.tolist(), [[1], [2], [0]])
        self.assertIsNone(output.logprobs_tensors)
        mock_gumbel_sample.assert_called_once()
        self.assertIs(mock_gumbel_sample.call_args.kwargs["logits"], logits)
        self.assertEqual(mock_gumbel_sample.call_args.kwargs["temperature"].tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(mock_gumbel_sample.call_args.kwargs["apply_temperature"])

    def test_rejects_phase_1b_unsupported_surfaces(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter
        from vllm_ascend.worker.v1.sample.sampling_context import V1SamplingContext

        adapter = V1SamplerAdapter(
            max_num_reqs=4,
            device=torch.device("cpu"),
        )
        normal_ctx = _ctx(("req0", "req1"))
        expanded_ctx = V1SamplingContext.from_model_runner_inputs(
            num_reqs=2,
            positions_at_logits=torch.tensor([0, 1, 0], dtype=torch.int64),
            input_ids_at_logits=torch.tensor([10, 11, 20], dtype=torch.int64),
            req_indices_at_logits=torch.tensor([0, 0, 1], dtype=torch.int32),
            device=torch.device("cpu"),
        )

        self.assertFalse(adapter.can_sample(_metadata(max_num_logprobs=1), normal_ctx))
        self.assertFalse(adapter.can_sample(_metadata(no_penalties=False), normal_ctx))
        self.assertFalse(adapter.can_sample(_metadata(), expanded_ctx))
        self.assertFalse(adapter.can_sample(_metadata(), _ctx_without_req_ids()))
        self.assertTrue(adapter.can_sample(_metadata(top_p=torch.tensor([0.9, 1.0])), normal_ctx))

        with self.assertRaisesRegex(TypeError, "top_p"):
            adapter._top_k_top_p_tensors(_metadata(top_p=[0.9, 1.0]), torch.device("cpu"))

    def test_applies_top_k_top_p_before_gumbel_sampling(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter

        adapter = V1SamplerAdapter(
            max_num_reqs=2,
            device=torch.device("cpu"),
        )
        logits = torch.randn(2, 8)
        filtered_logits = logits + 1.0
        top_k = torch.tensor([4, 8], dtype=torch.int32)
        top_p = torch.tensor([0.9, 1.0], dtype=torch.float32)

        with (
            patch(
                "vllm_ascend.worker.v2.sample.gumbel.apply_temperature",
            ) as mock_apply_temperature,
            patch(
                "vllm_ascend.sample.sampler.apply_top_k_top_p",
                return_value=filtered_logits,
            ) as mock_top_k_top_p,
            patch(
                "vllm_ascend.worker.v2.sample.gumbel.gumbel_sample",
                return_value=torch.tensor([1, 2], dtype=torch.int64),
            ) as mock_gumbel_sample,
        ):
            output = adapter(
                logits=logits,
                sampling_metadata=_metadata(top_k=top_k, top_p=top_p),
                ctx=_ctx(("req0", "req1")),
            )

        self.assertEqual(output.sampled_token_ids.tolist(), [[1], [2]])
        mock_apply_temperature.assert_called_once()
        self.assertIs(mock_apply_temperature.call_args.args[0], logits)
        mock_top_k_top_p.assert_called_once_with(logits, top_k, top_p)
        self.assertIs(mock_gumbel_sample.call_args.kwargs["logits"], filtered_logits)
        self.assertFalse(mock_gumbel_sample.call_args.kwargs["apply_temperature"])

    def test_prefers_compact_top_k_top_p_when_available(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter

        adapter = V1SamplerAdapter(
            max_num_reqs=2,
            device=torch.device("cpu"),
        )
        logits = torch.randn(2, 8)
        top_k = torch.tensor([4, 8], dtype=torch.int32)
        top_p = torch.tensor([0.9, 1.0], dtype=torch.float32)

        with (
            patch(
                "vllm_ascend.worker.v2.sample.gumbel.can_use_compact_top_k_top_p_sample",
                return_value=True,
            ) as mock_can_use_compact,
            patch(
                "vllm_ascend.worker.v2.sample.gumbel.compact_top_k_top_p_sample",
                return_value=torch.tensor([3, 5], dtype=torch.int32),
            ) as mock_compact_sample,
            patch(
                "vllm_ascend.worker.v2.sample.gumbel.apply_temperature",
            ) as mock_apply_temperature,
            patch("vllm_ascend.sample.sampler.apply_top_k_top_p") as mock_top_k_top_p,
            patch("vllm_ascend.worker.v2.sample.gumbel.gumbel_sample") as mock_gumbel_sample,
        ):
            output = adapter(
                logits=logits,
                sampling_metadata=_metadata(top_k=top_k, top_p=top_p),
                ctx=_ctx(("req0", "req1")),
            )

        self.assertEqual(output.sampled_token_ids.tolist(), [[3], [5]])
        mock_can_use_compact.assert_called_once()
        mock_compact_sample.assert_called_once()
        self.assertIs(mock_compact_sample.call_args.kwargs["logits"], logits)
        self.assertIs(mock_compact_sample.call_args.kwargs["k"], top_k)
        self.assertIs(mock_compact_sample.call_args.kwargs["p"], top_p)
        mock_apply_temperature.assert_not_called()
        mock_top_k_top_p.assert_not_called()
        mock_gumbel_sample.assert_not_called()

    def test_temperature_none_maps_to_greedy_gumbel_temperature(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter

        adapter = V1SamplerAdapter(
            max_num_reqs=2,
            device=torch.device("cpu"),
        )

        temp = adapter._temperature_for_sampling(_metadata(temperature=None), _ctx(("req0", "req1")))

        self.assertEqual(temp.tolist(), [0.0, 0.0])
        self.assertEqual(temp.dtype, torch.float32)

    def test_temperature_must_be_tensor_when_present(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter

        adapter = V1SamplerAdapter(
            max_num_reqs=2,
            device=torch.device("cpu"),
        )

        with self.assertRaisesRegex(TypeError, "temperature"):
            adapter._temperature_for_sampling(_metadata(temperature=1.0), _ctx(("req0", "req1")))

    def test_sampling_tensors_must_already_be_on_expected_device(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter

        adapter = V1SamplerAdapter(
            max_num_reqs=2,
            device=torch.device("meta"),
        )
        with self.assertRaisesRegex(ValueError, "temperature"):
            adapter._temperature_for_sampling(
                _metadata(temperature=torch.ones(2, dtype=torch.float32)),
                _ctx(("req0", "req1")),
            )
        with self.assertRaisesRegex(ValueError, "top_k"):
            adapter._top_k_top_p_tensors(
                _metadata(top_k=torch.ones(2, dtype=torch.int32)),
                torch.device("meta"),
            )

    def test_requires_request_ids_for_seed_cache(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter

        adapter = V1SamplerAdapter(
            max_num_reqs=2,
            device=torch.device("cpu"),
        )

        with self.assertRaisesRegex(ValueError, "request IDs"):
            adapter._compute_seeds(_metadata(), _ctx_without_req_ids())

    def test_seed_cache_follows_request_id_when_slot_changes(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter

        adapter = V1SamplerAdapter(
            max_num_reqs=3,
            device=torch.device("cpu"),
        )
        seeds = adapter._compute_seeds(_metadata(), _ctx(("req0", "req1", "req2"))).clone()

        moved_seeds = adapter._compute_seeds(
            _metadata(),
            _ctx(("req2", "req0", "req1")),
        )

        self.assertEqual(moved_seeds[0].item(), seeds[2].item())
        self.assertEqual(moved_seeds[1].item(), seeds[0].item())
        self.assertEqual(moved_seeds[2].item(), seeds[1].item())

    def test_compute_seeds_returns_device_slice_after_cpu_aggregation(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter

        adapter = V1SamplerAdapter(
            max_num_reqs=2,
            device=torch.device("meta"),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(12345)

        seeds = adapter._compute_seeds(
            _metadata(generators={0: generator}),
            _ctx(("req0", "req1")),
        )

        self.assertEqual(seeds.device.type, "meta")
        self.assertEqual(adapter._seeds_cpu[0].item(), 12345)
        self.assertEqual(adapter._seeds_cpu[:2].shape, (2,))


if __name__ == "__main__":
    unittest.main()
