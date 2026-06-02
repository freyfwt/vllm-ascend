import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vllm_ascend.sample.sampling_bridge import (
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

    def remove_request(self, req_id):
        self.req_id_to_index.pop(req_id, None)
        return True

    def add_request(self, req_id, prompt_len, all_token_ids, num_computed_tokens):
        req_idx = len(self.req_id_to_index)
        self.req_id_to_index[req_id] = req_idx
        self.added.append((req_id, prompt_len, all_token_ids, num_computed_tokens))

    def apply_staged_writes(self):
        pass


class _FakeGpuSampler:
    def __init__(self, **kwargs):
        del kwargs
        self.added = []
        self.sampling_states = SimpleNamespace(temperature=_FakeTemperature(torch.ones(4)))

    def add_request(self, req_idx, prompt_len, sampling_params):
        self.added.append((req_idx, prompt_len, sampling_params.name))

    def apply_staged_writes(self):
        pass


class _FakeTemperature:
    def __init__(self, temperature):
        self.gpu = temperature
        self.np = temperature.detach().cpu().numpy()


if __name__ == "__main__":
    unittest.main()
