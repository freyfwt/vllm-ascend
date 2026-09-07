from types import SimpleNamespace

import numpy as np
import torch

from vllm_ascend.distributed.eplb.policy.stair_types import BalanceScore, LayerPlan
from vllm_ascend.distributed.eplb.stair_runtime import StairModelRuntime


def _runtime():
    model = SimpleNamespace(num_physical_experts=4, num_logical_experts=4, num_moe_layers=1)
    state = SimpleNamespace(
        model=model,
        expert_load_pass=torch.tensor([[2, 3, 4, 5]], dtype=torch.int32),
        physical_to_logical_map=torch.tensor([[0, 1, 2, 3]]),
    )
    return StairModelRuntime.create("model", state, window_size=2, num_ranks=2, device=torch.device("cpu"))


def test_model_runtime_records_execution_and_resets_hot_state():
    runtime = _runtime()
    runtime.note_execution(True)
    runtime.record_step(7)
    assert runtime.ring.chronological_metadata() == ((True, True),)
    assert runtime.model_state.expert_load_pass.sum() == 0
    assert not runtime.executed


def test_model_runtime_discards_dummy_execution_metadata():
    runtime = _runtime()
    runtime.note_execution(True)
    runtime.discard_step()
    assert not runtime.executed
    assert not runtime.has_prefill
    assert runtime.model_state.expert_load_pass.sum() == 0


def test_model_runtime_updates_anchor_only_on_commit():
    runtime = _runtime()
    placement = runtime.placements()[0]
    layer = LayerPlan(
        0,
        placement.copy(),
        placement.copy(),
        np.array([[0, 0], [1, 1]]),
        np.array([[0, 1], [0, 1]]),
        0,
        BalanceScore(1.2, 1.2, 1.2),
        BalanceScore(1.0, 1.0, 1.0),
    )
    runtime.commit(layer)
    assert runtime.placement_epochs.tolist() == [1]
    assert runtime.accepted_score_tuple() == (1.0,)


def test_model_runtime_counts_overlapping_samples_once():
    runtime = _runtime()
    runtime.accept_snapshot((3, 4))
    runtime.accept_snapshot((4, 5))
    assert runtime.snapshot_sequence == 2
    assert runtime.sample_sequence == 3
    assert runtime.last_sampled_outer_step == 5
