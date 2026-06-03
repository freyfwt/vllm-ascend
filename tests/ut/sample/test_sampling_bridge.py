from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vllm_ascend.sample.sampling_bridge import (
    GpuBatchView,
    SamplingBridge,
    _DeviceBackedTensor,
    _DeviceStagedWriteTensor,
    sample_logits,
)


def test_sample_logits_modes():
    logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
    gumbel = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    mapping = torch.tensor([0, 1], dtype=torch.int32)

    assert sample_logits(logits, torch.ones(2), mapping, gumbel).tolist() == [0, 1]
    assert sample_logits(logits, torch.tensor([0.0, 1.0]), mapping, gumbel).tolist() == [1, 1]
    assert sample_logits(logits, torch.empty(0), mapping, gumbel, all_random=True).tolist() == [0, 1]


def test_sample_regular_can_reuse_or_mutate_logits():
    bridge = SamplingBridge.__new__(SamplingBridge)
    bridge._batch_view = GpuBatchView(2, torch.device("cpu"))
    bridge.sampler = _FakeSampler(torch.ones(2))
    input_batch = _sample_input_batch()

    output = bridge.sample_regular(
        torch.tensor([[0.0, 1.0], [3.0, 0.0]]),
        input_batch,
        torch.tensor([[2.0, 0.0], [0.0, 4.0]]),
    )

    assert output.sampled_token_ids.tolist() == [[0], [1]]
    assert output.num_sampled.tolist() == [1, 1]
    assert len(bridge.sampler.apply_calls) == 1

    bridge.sampler = _FakeInplaceSampler(torch.ones(2))
    logits = torch.tensor([[0.0, 1.0], [3.0, 0.0]])
    bridge.sample_regular(logits, input_batch, torch.zeros_like(logits), preserve_original_logits=False)
    assert logits.tolist() == [[1.0, 2.0], [4.0, 1.0]]


def test_update_requests_is_incremental_and_removes_departed_requests():
    with _patched_bridge_deps():
        bridge = _make_bridge()
        bridge.update_requests(_make_fake_input_batch(["req0", "req1"]), _make_fake_requests(["req0", "req1"]))
        bridge.update_requests(_make_fake_input_batch(["req0", "req1"]), _make_fake_requests(["req0", "req1"]))
        bridge.update_requests(_make_fake_input_batch(["req1", "req2"]), _make_fake_requests(["req0", "req1", "req2"]))

    assert [item[0] for item in bridge.req_states.added] == ["req0", "req1", "req2"]
    assert bridge.req_states.removed == ["req0"]
    assert bridge.sampler.applied_writes == 2


def test_bind_batch_regular_and_spec_reuse_runner_tensors():
    view = GpuBatchView(max_num_reqs=4, device=torch.device("cpu"))
    req_ids = ["req0", "req1"]
    regular = view.bind_regular(
        req_ids,
        torch.tensor([3, 2], dtype=torch.int32),
        np.array([3, 2], dtype=np.int32),
        torch.tensor([10, 11], dtype=torch.int32),
        torch.tensor([5, 6], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        np.array([0, 1, 2], dtype=np.int32),
        torch.tensor([6, 7], dtype=torch.int32),
        torch.tensor([6, 7], dtype=torch.int32),
    )

    assert regular is view.input_batch
    assert regular.expanded_idx_mapping.tolist() == [3, 2]
    assert regular.expanded_local_pos.tolist() == [0, 0]
    assert regular.cu_num_logits.tolist() == [0, 1, 2]

    metadata = SimpleNamespace(
        num_draft_tokens=[2, 1],
        logits_indices=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([3, 5], dtype=torch.int32),
    )
    with patch("vllm_ascend.sample.sampling_bridge.expand_idx_mapping", _fake_expand_idx_mapping):
        spec = view.bind_spec(
            req_ids,
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

    assert spec.logits_indices is metadata.logits_indices
    assert spec.cu_num_logits_np.tolist() == [0, 3, 5]
    assert spec.expanded_idx_mapping.tolist() == [4, 4, 4, 5, 5]
    assert spec.expanded_local_pos.tolist() == [0, 1, 2, 0, 1]


def test_device_backed_and_staged_tensors_copy_to_device():
    _DeviceBackedTensor.set_device(torch.device("cpu"))
    backed = _DeviceBackedTensor(4, torch.int32)
    backed.np[:] = [1, 2, 3, 4]
    backed.copy_to_uva(2)
    assert backed.gpu.tolist() == [1, 2, 0, 0]

    staged = _DeviceStagedWriteTensor((2, 4), torch.int32, torch.device("cpu"))
    staged.stage_write(0, 1, (idx for idx in [3, 4]))
    staged.stage_write(1, 0, [5])
    staged.apply_write()
    assert staged.gpu.tolist() == [[0, 3, 4, 0], [5, 0, 0, 0]]
    assert staged._staged_write_indices == []


def _patched_bridge_deps():
    return patch.multiple(
        "vllm_ascend.sample.sampling_bridge",
        RequestState=_FakeRequestState,
        GpuSampler=_FakeGpuSampler,
    )


def _make_bridge():
    return SamplingBridge(
        max_num_reqs=4,
        max_model_len=8,
        max_num_batched_tokens=4,
        vocab_size=16,
        device=torch.device("cpu"),
        logprobs_mode="raw_logprobs",
        speculative_config=None,
    )


def _sample_input_batch():
    return SimpleNamespace(
        positions=torch.arange(2, dtype=torch.int64),
        input_ids=torch.tensor([3, 4], dtype=torch.int32),
        logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        expanded_local_pos=torch.zeros(2, dtype=torch.int32),
        num_reqs=2,
    )


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
        self.logit_bias_state = SimpleNamespace(apply_logit_bias=lambda logits, *args: logits.add_(1.0))
        self.penalties_state = SimpleNamespace(apply_penalties=lambda *args: None)
        self.bad_words_state = SimpleNamespace(apply_bad_words=lambda *args: None)
        self.sampling_states = SimpleNamespace(
            temperature=_FakeTemperature(temperature),
            apply_temperature=lambda *args: None,
            apply_min_p=lambda *args: None,
            apply_top_k_top_p=lambda logits, *args: logits,
        )


class _FakeRequestState:
    def __init__(self, **kwargs):
        del kwargs
        self.req_id_to_index = {}
        self.num_speculative_steps = 0
        self.added = []
        self.removed = []

    def remove_request(self, req_id):
        if req_id not in self.req_id_to_index:
            return False
        self.req_id_to_index.pop(req_id)
        self.removed.append(req_id)
        return True

    def add_request(self, req_id, prompt_len, all_token_ids, num_computed_tokens):
        self.req_id_to_index[req_id] = len(self.req_id_to_index)
        self.added.append((req_id, prompt_len, all_token_ids, num_computed_tokens))

    @staticmethod
    def apply_staged_writes():
        return None


class _FakeGpuSampler:
    def __init__(self, **kwargs):
        del kwargs
        self.added = []
        self.applied_writes = 0

    def add_request(self, req_idx, prompt_len, sampling_params):
        self.added.append((req_idx, prompt_len, sampling_params.name))

    def apply_staged_writes(self):
        self.applied_writes += 1


class _FakeTemperature:
    def __init__(self, temperature):
        self.gpu = temperature
        self.np = temperature.detach().cpu().numpy()


def _fake_expand_idx_mapping(idx_mapping, total_num_logits, cu_num_logits, max_expand_len):
    del max_expand_len
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = idx_mapping.new_empty(total_num_logits)
    for req_idx in range(idx_mapping.shape[0]):
        start = int(cu_num_logits[req_idx])
        end = int(cu_num_logits[req_idx + 1])
        expanded_idx_mapping[start:end] = idx_mapping[req_idx]
        expanded_local_pos[start:end] = torch.arange(end - start, dtype=torch.int32)
    return expanded_idx_mapping, expanded_local_pos


def _make_fake_input_batch(req_ids):
    return SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        num_prompt_tokens=np.array([2, 1, 1], dtype=np.int32),
        num_tokens_no_spec=np.array([3, 2, 2], dtype=np.int32),
        num_computed_tokens_cpu=np.array([3, 2, 2], dtype=np.int32),
        token_ids_cpu=np.array([[10, 11, 12, 0], [20, 21, 0, 0], [30, 31, 0, 0]], dtype=np.int32),
        is_token_ids=np.ones((3, 4), dtype=bool),
    )


def _make_fake_requests(req_ids):
    return {req_id: SimpleNamespace(sampling_params=SimpleNamespace(name=f"p{i}")) for i, req_id in enumerate(req_ids)}
