# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

# Import registers the custom op used by the production router path.
import vllm_ascend.ops.fused_moe.eplb  # noqa: F401


def test_map_to_physical_and_record_triton_gates_load_collection():
    routing_table = torch.tensor(
        [[0, 3], [2, 1], [0, 3], [2, 1]],
        dtype=torch.int32,
        device="npu",
    )
    topk_ids = torch.tensor(
        [[0, 1], [0, 1], [0, 1], [0, 1]],
        dtype=torch.int32,
        device="npu",
    )
    expert_load = torch.zeros(4, dtype=torch.int32, device="npu")

    physical_ids = torch.ops.vllm.ascend_eplb_map_to_physical_and_record(
        topk_ids,
        routing_table,
        expert_load,
        record_enabled=torch.tensor(True, device="npu"),
        num_unpadded_tokens=torch.tensor(3, dtype=torch.int32, device="npu"),
    )

    torch.testing.assert_close(
        physical_ids.cpu(),
        torch.tensor([[0, 3], [2, 1], [0, 3], [2, 1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        expert_load.cpu(),
        torch.tensor([2, 1, 1, 2], dtype=torch.int32),
    )

    torch.ops.vllm.ascend_eplb_map_to_physical_and_record(
        topk_ids,
        routing_table,
        expert_load,
        record_enabled=torch.tensor(False, device="npu"),
        num_unpadded_tokens=torch.tensor(4, dtype=torch.int32, device="npu"),
    )
    torch.testing.assert_close(
        expert_load.cpu(),
        torch.tensor([2, 1, 1, 2], dtype=torch.int32),
    )
