# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Startup checks for STAIR model and transfer invariants."""

from collections.abc import Sequence
from typing import Any

import torch

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.transfer_adapter import validate_transfer_capability
from vllm_ascend.ops.fused_moe.eplb import EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS


def _tensor_schema(tensors: Sequence[torch.Tensor], slots: int) -> tuple:
    schema = []
    for tensor in tensors:
        if tensor.layout != torch.strided or tensor.ndim == 0 or tensor.shape[0] != slots:
            raise ValueError("STAIR requires dense expert tensors with a leading local-slot dimension")
        schema.append((str(tensor.dtype), tuple(tensor.shape), tuple(tensor.stride()), tensor.device.type))
    return tuple(schema)


def validate_stair_model(model_state: Any, config: StairConfig, num_ranks: int) -> int:
    """Validate fixed tensor schema and return full per-rank staging bytes."""
    model = model_state.model
    physical = model.num_physical_experts
    logical = model.num_logical_experts
    if physical % num_ranks:
        raise ValueError("STAIR physical experts must divide evenly across EP ranks")
    slots = physical // num_ranks
    if not logical <= physical <= logical * num_ranks:
        raise ValueError("STAIR requires E <= P <= E * EP ranks")
    if config.max_expert_transfers_per_rank_pair > slots:
        raise ValueError("STAIR rank-pair transfer cap cannot exceed local expert slots")
    validate_transfer_capability(model_state.communicator)

    layers = list(model.expert_weights)
    if len(layers) != model.num_moe_layers or not layers:
        raise ValueError("STAIR expert tensor layers do not match the model layer count")
    reference = _tensor_schema(layers[0], slots)
    if _tensor_schema(model_state.expert_buffer, slots) != reference:
        raise ValueError("STAIR staging buffer does not match the expert tensor schema")
    for layer_idx, tensors in enumerate(layers[1:], start=1):
        if _tensor_schema(tensors, slots) != reference:
            raise ValueError(f"STAIR expert tensor schema changed at layer {layer_idx}")

    expected_routing_shape = (EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS, logical)
    moe_layers = list(model.moe_layers)
    if len(moe_layers) != model.num_moe_layers:
        raise ValueError("STAIR routing layers do not match the model layer count")
    for layer_idx, layer in enumerate(moe_layers):
        layer_state = getattr(layer, "eplb_state", None)
        routing = getattr(layer_state, "expert_replica_routing_table", None)
        if routing is None or tuple(routing.shape) != expected_routing_shape:
            raise ValueError(f"STAIR routing shape is not graph-stable at layer {layer_idx}")
        if not callable(getattr(layer_state, "refresh_expert_replica_routing_table", None)):
            raise ValueError(f"STAIR routing refresh capability is missing at layer {layer_idx}")

    metadata_bytes = slots * (3 * 1 + 8 + 4)
    staged_bytes = sum(tensor.numel() * tensor.element_size() for tensor in model_state.expert_buffer)
    staged_bytes += metadata_bytes
    limit = config.max_staged_bytes_per_rank
    if limit != "auto" and staged_bytes > limit:
        raise ValueError(f"STAIR staging requires {staged_bytes} bytes per rank, above limit {limit}")
    return staged_bytes
