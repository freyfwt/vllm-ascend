# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.distributed.eplb.planner_protocol import PlannerModelRegistration
from vllm_ascend.distributed.eplb.planner_shared_memory import StairSharedInputs
from vllm_ascend.distributed.eplb.policy.stair_types import StairTopology


def _registration() -> PlannerModelRegistration:
    return PlannerModelRegistration(
        model_id="main",
        load_shape=(2, 1, 4),
        mapping_shape=(1, 6),
        num_replicas=6,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        topology=StairTopology.contiguous(2, 1),
        layer_expert_bytes=None,
    )


def test_shared_inputs_double_buffer_and_validate_digest():
    owner = StairSharedInputs.create(_registration())
    reader = StairSharedInputs.attach(owner.spec)
    load = torch.arange(8, dtype=torch.float64).reshape(2, 1, 4)
    mapping = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long)
    try:
        first_slot, first_sequence, first_digest = owner.write(load, mapping)
        second_slot, second_sequence, second_digest = owner.write(load + 1, mapping)

        assert first_slot != second_slot
        assert second_sequence == first_sequence + 1
        first_load, first_mapping = reader.read(first_slot, first_digest)
        second_load, _ = reader.read(second_slot, second_digest)
        torch.testing.assert_close(first_load, load)
        torch.testing.assert_close(first_mapping, mapping)
        torch.testing.assert_close(second_load, load + 1)
        with pytest.raises(RuntimeError, match="digest mismatch"):
            reader.read(first_slot, second_digest)
    finally:
        reader.close()
        owner.close()


def test_shared_inputs_reject_shape_drift():
    owner = StairSharedInputs.create(_registration())
    try:
        with pytest.raises(ValueError, match="load shape changed"):
            owner.write(torch.ones((1, 1, 4)), torch.ones((1, 6), dtype=torch.long))
    finally:
        owner.close()
