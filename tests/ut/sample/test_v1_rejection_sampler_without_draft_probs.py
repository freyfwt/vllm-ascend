import pytest
import torch


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch.npu.is_available()


@pytest.mark.skipif(not _npu_available(), reason="requires NPU")
def test_rejection_sample_without_draft_probs_accepts_and_recovers():
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.worker.v1.sample.rejection_sampler_without_draft_probs import (
        rejection_sample_without_draft_probs,
    )

    torch.npu.set_device(0)
    init_device_properties_triton()
    device = torch.device("npu:0")

    target_logits = torch.tensor(
        [
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
            [0.0, 5.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    sampled, num_sampled = rejection_sample_without_draft_probs(
        target_logits=target_logits,
        draft_sampled=torch.tensor([0, 1, 0, 2], dtype=torch.int32, device=device),
        cu_num_logits=torch.tensor([0, 2, 4], dtype=torch.int32, device=device),
        positions=torch.arange(4, dtype=torch.int64, device=device),
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32, device=device),
        expanded_idx_mapping=torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device),
        expanded_local_pos=torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device),
        temperature=torch.tensor([1.0, 1.0], dtype=torch.float32, device=device),
        accept_uniform=torch.tensor([0.05, 0.0, 0.95, 0.0], dtype=torch.float32, device=device),
        resample_gumbel=torch.zeros((2, 4), dtype=torch.float32, device=device),
        num_speculative_steps=1,
    )

    torch.npu.synchronize()
    assert sampled.cpu().tolist() == [[1, 2], [3, -1]]
    assert num_sampled.cpu().tolist() == [2, 1]
