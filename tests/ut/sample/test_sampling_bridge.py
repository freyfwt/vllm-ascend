import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vllm_ascend.sample.sampling_bridge import (
    GpuBatchView,
    SamplingBridge,
    _DeviceBackedTensor,
    _DeviceStagedWriteTensor,
    sample_processed_logits,
)


class TestSamplingBridge(unittest.TestCase):
    def test_sample_processed_logits_uses_external_gumbel(self):
        processed_logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
        sampling_gumbel = torch.tensor([[2.0, 0.0], [0.0, 4.0]])

        sampled = sample_processed_logits(
            processed_logits,
            torch.ones(2, dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.int32),
            sampling_gumbel,
        )

        self.assertEqual(sampled.tolist(), [0, 1])

    def test_sample_processed_logits_preserves_greedy_rows(self):
        processed_logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
        sampling_gumbel = torch.tensor([[2.0, 0.0], [0.0, 4.0]])

        sampled = sample_processed_logits(
            processed_logits,
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.int32),
            sampling_gumbel,
        )

        self.assertEqual(sampled.tolist(), [1, 1])

    def test_sample_processed_logits_skips_mixed_logic_for_all_random(self):
        processed_logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
        sampling_gumbel = torch.tensor([[2.0, 0.0], [0.0, 4.0]])

        sampled = sample_processed_logits(
            processed_logits,
            torch.empty(0, dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.int32),
            sampling_gumbel,
            all_random=True,
        )

        self.assertEqual(sampled.tolist(), [0, 1])

    def test_sample_regular_uses_gpu_sampler_processing(self):
        bridge = SamplingBridge.__new__(SamplingBridge)
        bridge.sampler = _FakeSampler(torch.tensor([1.0, 1.0]))
        input_batch = SimpleNamespace(
            positions=torch.arange(2, dtype=torch.int64),
            input_ids=torch.tensor([3, 4], dtype=torch.int32),
            logits_indices=torch.tensor([0, 1], dtype=torch.int32),
            expanded_idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
            idx_mapping_np=np.array([0, 1], dtype=np.int32),
            expanded_local_pos=torch.zeros(2, dtype=torch.int32),
        )

        output = bridge.sample_regular(
            torch.tensor([[0.0, 1.0], [3.0, 0.0]]),
            input_batch,
            torch.tensor([[2.0, 0.0], [0.0, 4.0]]),
        )

        self.assertEqual(output.sampled_token_ids.tolist(), [[0], [1]])
        self.assertEqual(len(bridge.sampler.apply_calls), 1)

    def test_sample_regular_processes_logits_in_place_when_raw_is_not_needed(self):
        bridge = SamplingBridge.__new__(SamplingBridge)
        bridge.sampler = _FakeInplaceSampler(torch.tensor([1.0, 1.0]))
        input_batch = SimpleNamespace(
            positions=torch.arange(2, dtype=torch.int64),
            input_ids=torch.tensor([3, 4], dtype=torch.int32),
            logits_indices=torch.tensor([0, 1], dtype=torch.int32),
            expanded_idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
            idx_mapping_np=np.array([0, 1], dtype=np.int32),
            expanded_local_pos=torch.zeros(2, dtype=torch.int32),
        )
        logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])

        bridge.sample_regular(
            logits,
            input_batch,
            torch.zeros_like(logits),
            preserve_original_logits=False,
        )

        self.assertEqual(logits.tolist(), [[1.0, 2.0], [4.0, 1.0]])

    def test_prepare_syncs_request_states_from_mrv1_cpu_mirror(self):
        with (
            patch(
                "vllm_ascend.sample.sampling_bridge.RequestState",
                _FakeRequestState,
            ),
            patch(
                "vllm_ascend.sample.sampling_bridge.GpuSampler",
                _FakeGpuSampler,
            ),
        ):
            bridge = SamplingBridge(
                max_num_reqs=4,
                max_model_len=8,
                max_num_batched_tokens=4,
                vocab_size=16,
                device=torch.device("cpu"),
                logprobs_mode="raw_logprobs",
                speculative_config=None,
            )
            input_batch = SimpleNamespace(
                req_ids=["req0", "req1"],
                req_id_to_index={"req0": 0, "req1": 1},
                num_prompt_tokens=np.array([2, 1], dtype=np.int32),
                num_tokens_no_spec=np.array([3, 2], dtype=np.int32),
                num_computed_tokens_cpu=np.array([3, 2], dtype=np.int32),
                token_ids_cpu=np.array(
                    [[10, 11, 12, 0], [20, 21, 0, 0]],
                    dtype=np.int32,
                ),
                is_token_ids=np.ones((2, 4), dtype=bool),
            )
            requests = {
                "req0": SimpleNamespace(sampling_params=SimpleNamespace(name="p0")),
                "req1": SimpleNamespace(sampling_params=SimpleNamespace(name="p1")),
            }

            gpu_input = bridge.prepare(
                input_batch,
                requests,
                torch.tensor([10, 20], dtype=torch.int32),
                torch.arange(2, dtype=torch.int64),
                torch.tensor([0, 1], dtype=torch.int32),
                None,
            )

        self.assertIsNotNone(gpu_input)
        self.assertEqual(
            bridge.req_states.added,
            [
                ("req0", 2, [10, 11, 12], 3),
                ("req1", 1, [20, 21], 2),
            ],
        )
        self.assertEqual(
            bridge.sampler.added,
            [
                (0, 2, "p0"),
                (1, 1, "p1"),
            ],
        )
        self.assertEqual(gpu_input.idx_mapping_np.tolist(), [0, 1])

    def test_update_requests_is_incremental_for_stable_batch(self):
        with (
            patch(
                "vllm_ascend.sample.sampling_bridge.RequestState",
                _FakeRequestState,
            ),
            patch(
                "vllm_ascend.sample.sampling_bridge.GpuSampler",
                _FakeGpuSampler,
            ),
        ):
            bridge = SamplingBridge(
                max_num_reqs=4,
                max_model_len=8,
                max_num_batched_tokens=4,
                vocab_size=16,
                device=torch.device("cpu"),
                logprobs_mode="raw_logprobs",
                speculative_config=None,
            )
            input_batch = _make_fake_input_batch(["req0", "req1"])
            requests = _make_fake_requests(["req0", "req1"])

            self.assertTrue(bridge.update_requests(input_batch, requests))
            self.assertTrue(bridge.update_requests(input_batch, requests))

        self.assertEqual(len(bridge.req_states.added), 2)
        self.assertEqual(bridge.req_states.removed, [])
        self.assertEqual(bridge.sampler.applied_writes, 1)

    def test_update_requests_removes_only_departed_requests(self):
        with (
            patch(
                "vllm_ascend.sample.sampling_bridge.RequestState",
                _FakeRequestState,
            ),
            patch(
                "vllm_ascend.sample.sampling_bridge.GpuSampler",
                _FakeGpuSampler,
            ),
        ):
            bridge = SamplingBridge(
                max_num_reqs=4,
                max_model_len=8,
                max_num_batched_tokens=4,
                vocab_size=16,
                device=torch.device("cpu"),
                logprobs_mode="raw_logprobs",
                speculative_config=None,
            )
            bridge.update_requests(
                _make_fake_input_batch(["req0", "req1"]),
                _make_fake_requests(["req0", "req1", "req2"]),
            )
            bridge.update_requests(
                _make_fake_input_batch(["req1", "req2"]),
                _make_fake_requests(["req0", "req1", "req2"]),
            )

        self.assertEqual(bridge.req_states.removed, ["req0"])
        self.assertEqual([item[0] for item in bridge.req_states.added], ["req0", "req1", "req2"])

    def test_bind_batch_regular_reuses_runner_tensors(self):
        view = GpuBatchView(max_num_reqs=4, device=torch.device("cpu"))
        req_ids = ["req0", "req1"]
        idx_mapping = torch.tensor([3, 2], dtype=torch.int32)
        idx_mapping_np = np.array([3, 2], dtype=np.int32)
        input_ids = torch.tensor([10, 11], dtype=torch.int32)
        positions = torch.tensor([5, 6], dtype=torch.int64)
        logits_indices = torch.tensor([0, 1], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
        query_start_loc_np = np.array([0, 1, 2], dtype=np.int32)
        seq_lens = torch.tensor([6, 7], dtype=torch.int32)
        seq_lens_cpu_upper_bound = torch.tensor([6, 7], dtype=torch.int32)

        batch = view.bind_regular(
            req_ids,
            idx_mapping,
            idx_mapping_np,
            input_ids,
            positions,
            logits_indices,
            query_start_loc,
            query_start_loc_np,
            seq_lens,
            seq_lens_cpu_upper_bound,
        )

        self.assertIs(batch, view.input_batch)
        self.assertIs(batch.input_ids, input_ids)
        self.assertIs(batch.positions, positions)
        self.assertIs(batch.logits_indices, logits_indices)
        self.assertIs(batch.seq_lens, seq_lens)
        self.assertIs(batch.expanded_idx_mapping, idx_mapping)
        self.assertEqual(batch.expanded_local_pos.tolist(), [0, 0])
        self.assertEqual(batch.cu_num_logits.tolist(), [0, 1, 2])

    def test_bind_batch_spec_uses_metadata_and_kernel_mapping(self):
        view = GpuBatchView(max_num_reqs=4, device=torch.device("cpu"))
        metadata = SimpleNamespace(
            num_draft_tokens=[2, 1],
            logits_indices=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
            cu_num_sampled_tokens=torch.tensor([3, 5], dtype=torch.int32),
        )
        with patch(
            "vllm_ascend.sample.sampling_bridge._fill_spec_decode_mapping_kernel",
            _FakeSpecMappingKernel(),
        ):
            batch = view.bind_spec(
                ["req0", "req1"],
                torch.tensor([4, 5], dtype=torch.int32),
                np.array([4, 5], dtype=np.int32),
                metadata,
                torch.arange(5, dtype=torch.int32),
                torch.arange(5, dtype=torch.int64),
                torch.tensor([0, 3, 5], dtype=torch.int32),
                np.array([0, 3, 5], dtype=np.int32),
                torch.tensor([8, 9], dtype=torch.int32),
                torch.tensor([8, 9], dtype=torch.int32),
                num_speculative_steps=2,
            )

        self.assertIs(batch.logits_indices, metadata.logits_indices)
        self.assertEqual(batch.cu_num_logits.tolist(), [0, 3, 5])
        self.assertEqual(batch.cu_num_logits_np.tolist(), [0, 3, 5])
        self.assertEqual(batch.expanded_idx_mapping.tolist(), [4, 4, 4, 5, 5])
        self.assertEqual(batch.expanded_local_pos.tolist(), [0, 1, 2, 0, 1])

    def test_sample_spec_returns_num_sampled(self):
        bridge = SamplingBridge.__new__(SamplingBridge)
        bridge.rejection_sampler = _FakeRejectionSampler()
        output = bridge.sample_spec(
            torch.empty(0),
            SimpleNamespace(),
            torch.empty(0),
            torch.empty(0),
        )

        self.assertEqual(output.sampled_token_ids.tolist(), [[7, -1]])
        self.assertEqual(output.num_sampled.tolist(), [1])

    def test_device_backed_tensor_copies_cpu_mirror_to_device_tensor(self):
        _DeviceBackedTensor.set_device(torch.device("cpu"))
        tensor = _DeviceBackedTensor(4, torch.int32)
        tensor.np[:] = [1, 2, 3, 4]

        tensor.copy_to_uva(2)

        self.assertEqual(tensor.gpu.tolist(), [1, 2, 0, 0])

    def test_device_staged_write_tensor_applies_rows_on_device(self):
        tensor = _DeviceStagedWriteTensor(
            (2, 4),
            torch.int32,
            torch.device("cpu"),
        )
        tensor.stage_write(0, 1, (idx for idx in [3, 4]))
        tensor.stage_write(1, 0, [5])

        tensor.apply_write()

        self.assertEqual(tensor.gpu.tolist(), [[0, 3, 4, 0], [5, 0, 0, 0]])
        self.assertEqual(tensor._staged_write_indices, [])


class _FakeSampler:
    def __init__(self, temperature):
        self.sampling_states = SimpleNamespace(temperature=_FakeTemperature(temperature))
        self.apply_calls = []

    def apply_sampling_params(self, *args):
        self.apply_calls.append(args)
        return args[0].to(torch.float32)


class _FakeInplaceSampler:
    def __init__(self, temperature):
        self.num_speculative_tokens = 1
        self.logit_bias_state = _FakeAddOneLogitBiasState()
        self.penalties_state = _FakeNoopState()
        self.bad_words_state = _FakeNoopState()
        self.sampling_states = _FakeSamplingStates(temperature)


class _FakeAddOneLogitBiasState:
    @staticmethod
    def apply_logit_bias(logits, *args):
        del args
        logits.add_(1.0)


class _FakeNoopState:
    @staticmethod
    def apply_penalties(*args):
        del args

    @staticmethod
    def apply_bad_words(*args):
        del args


class _FakeSamplingStates:
    def __init__(self, temperature):
        self.temperature = _FakeTemperature(temperature)

    @staticmethod
    def apply_temperature(*args):
        del args

    @staticmethod
    def apply_min_p(*args):
        del args

    @staticmethod
    def apply_top_k_top_p(logits, *args):
        del args
        return logits


class _FakeRequestState:
    def __init__(self, **kwargs):
        del kwargs
        self.req_id_to_index = {}
        self.num_speculative_steps = 0
        self.added = []
        self.removed = []
        self.applied_writes = 0

    def remove_request(self, req_id):
        if req_id not in self.req_id_to_index:
            return False
        self.removed.append(req_id)
        self.req_id_to_index.pop(req_id)
        return True

    def add_request(self, req_id, prompt_len, all_token_ids, num_computed_tokens):
        req_idx = len(self.req_id_to_index)
        self.req_id_to_index[req_id] = req_idx
        self.added.append((req_id, prompt_len, all_token_ids, num_computed_tokens))

    def apply_staged_writes(self):
        self.applied_writes += 1


class _FakeGpuSampler:
    def __init__(self, **kwargs):
        del kwargs
        self.added = []
        self.sampling_states = SimpleNamespace(temperature=_FakeTemperature(torch.ones(4)))
        self.applied_writes = 0

    def add_request(self, req_idx, prompt_len, sampling_params):
        self.added.append((req_idx, prompt_len, sampling_params.name))

    def apply_staged_writes(self):
        self.applied_writes += 1


class _FakeTemperature:
    def __init__(self, temperature):
        self.gpu = temperature
        self.np = temperature.detach().cpu().numpy()


class _FakeSpecMappingKernel:
    def __getitem__(self, grid):
        del grid

        def launch(
            cu_num_logits,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            **kwargs,
        ):
            del kwargs
            for req_idx in range(idx_mapping.shape[0]):
                start = int(cu_num_logits[req_idx])
                end = int(cu_num_logits[req_idx + 1])
                expanded_idx_mapping[start:end] = idx_mapping[req_idx]
                expanded_local_pos[start:end] = torch.arange(end - start, dtype=torch.int32)

        return launch


class _FakeRejectionSampler:
    def sample_with_prefetched_noise(self, *args):
        del args
        return (
            torch.tensor([[7, -1]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
        )


def _make_fake_input_batch(req_ids):
    return SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        num_prompt_tokens=np.array([2, 1, 1], dtype=np.int32),
        num_tokens_no_spec=np.array([3, 2, 2], dtype=np.int32),
        num_computed_tokens_cpu=np.array([3, 2, 2], dtype=np.int32),
        token_ids_cpu=np.array(
            [[10, 11, 12, 0], [20, 21, 0, 0], [30, 31, 0, 0]],
            dtype=np.int32,
        ),
        is_token_ids=np.ones((3, 4), dtype=bool),
    )


def _make_fake_requests(req_ids):
    return {req_id: SimpleNamespace(sampling_params=SimpleNamespace(name=f"p{i}")) for i, req_id in enumerate(req_ids)}


if __name__ == "__main__":
    unittest.main()
