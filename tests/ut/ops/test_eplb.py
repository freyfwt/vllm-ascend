# SPDX-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

from vllm_ascend.ops.fused_moe.eplb import (
    EPLB_LOOKUP_NUM_ROWS,
    build_physical_id_lookup,
    map_to_physical,
    normalize_local_expert_load,
)


def _eplb_inputs():
    logical_to_physical_map = torch.tensor(
        [[0, 4, -1], [1, 5, -1], [2, -1, -1]],
        dtype=torch.int64,
    )
    logical_replica_count = torch.tensor([2, 2, 1], dtype=torch.int64)
    return logical_to_physical_map, logical_replica_count


def test_build_physical_id_lookup_applies_rank_and_expert_offsets():
    logical_map, replica_count = _eplb_inputs()

    rank0_lookup = build_physical_id_lookup(logical_map, replica_count, ep_rank=0)
    rank1_lookup = build_physical_id_lookup(logical_map, replica_count, ep_rank=1)

    assert rank0_lookup.shape == (EPLB_LOOKUP_NUM_ROWS, 3)
    assert rank0_lookup.dtype == torch.int32
    torch.testing.assert_close(rank0_lookup[0], torch.tensor([0, 5, 2], dtype=torch.int32))
    torch.testing.assert_close(rank1_lookup[0], torch.tensor([4, 1, 2], dtype=torch.int32))
    torch.testing.assert_close(rank0_lookup[1], rank1_lookup[0])


def test_map_to_physical_uses_periodic_rows():
    logical_map, replica_count = _eplb_inputs()
    topk_ids = torch.zeros((EPLB_LOOKUP_NUM_ROWS + 1, 2), dtype=torch.int64)
    topk_ids[:, 1] = 1

    physical_ids = map_to_physical(topk_ids, logical_map, replica_count, ep_rank=0)

    assert physical_ids[0, 0] == 0
    assert physical_ids[EPLB_LOOKUP_NUM_ROWS - 1, 0] == 4
    assert physical_ids[EPLB_LOOKUP_NUM_ROWS, 0] == physical_ids[0, 0]
    assert physical_ids[EPLB_LOOKUP_NUM_ROWS, 1] == physical_ids[0, 1]


def test_normalize_local_expert_load_preserves_non_cumulative_counts():
    local_load = normalize_local_expert_load(
        expert_tokens=torch.tensor([3, 5], dtype=torch.int64),
        group_list_type=1,
        num_local_physical_experts=2,
    )

    torch.testing.assert_close(local_load, torch.tensor([3, 5], dtype=torch.int64))


def test_normalize_local_expert_load_converts_cumulative_group_list():
    local_load = normalize_local_expert_load(
        expert_tokens=torch.tensor([2, 7], dtype=torch.int64),
        group_list_type=0,
        num_local_physical_experts=2,
    )

    torch.testing.assert_close(local_load, torch.tensor([2, 5], dtype=torch.int64))
