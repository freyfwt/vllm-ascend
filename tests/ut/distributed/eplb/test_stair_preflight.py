from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.stair_preflight import validate_stair_model


class Communicator:
    add_send = add_recv = execute = set_stream = lambda *args: None


def _state(shape=(2, 4)):
    weights = [[torch.empty(shape)] for _ in range(2)]
    moe_layers = [
        SimpleNamespace(
            eplb_state=SimpleNamespace(
                expert_replica_routing_table=torch.empty(1024, 3, dtype=torch.int32),
                refresh_expert_replica_routing_table=lambda: None,
            )
        )
        for _ in range(2)
    ]
    return SimpleNamespace(
        model=SimpleNamespace(
            num_physical_experts=4,
            num_logical_experts=3,
            num_moe_layers=2,
            expert_weights=weights,
            moe_layers=moe_layers,
        ),
        expert_buffer=[torch.empty(shape)],
        communicator=Communicator(),
    )


def test_preflight_computes_complete_staging_size():
    assert validate_stair_model(_state(), StairConfig(), 2) == 2 * 4 * 4 + 2 * 15


def test_preflight_rejects_pair_cap_above_local_slots():
    with pytest.raises(ValueError, match="pair transfer cap"):
        validate_stair_model(_state(), StairConfig(max_expert_transfers_per_rank_pair=3), 2)


def test_preflight_rejects_small_staging_limit():
    with pytest.raises(ValueError, match="above limit"):
        validate_stair_model(_state(), StairConfig(max_staged_bytes_per_rank=1), 2)


def test_preflight_rejects_cross_layer_schema_change():
    state = _state()
    state.model.expert_weights[1] = [torch.empty(2, 5)]
    with pytest.raises(ValueError, match="schema changed"):
        validate_stair_model(state, StairConfig(), 2)


def test_preflight_rejects_unstable_routing_shape():
    state = _state()
    state.model.moe_layers[1].eplb_state.expert_replica_routing_table = torch.empty(1, 3)
    with pytest.raises(ValueError, match="graph-stable"):
        validate_stair_model(state, StairConfig(), 2)
