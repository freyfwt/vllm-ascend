# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Typed request and result payloads for the STAIR planner protocol."""

import struct
from dataclasses import dataclass

import numpy as np

from vllm_ascend.distributed.eplb.policy.stair_types import BalanceScore, LayerPlan, RebalancePlan

_REQUEST = struct.Struct("<BI5Q32s32s32s")
_PLAN = struct.Struct("<4Q32s32sI")
_LAYER = struct.Struct("<iq6d")


@dataclass(frozen=True)
class PlanRequest:
    model_id: str
    slot: int
    bin_count: int
    slot_sequence: int
    planning_round: int
    snapshot_sequence: int
    sample_sequence: int
    stats_schema_epoch: int
    input_digest: str
    key_digest: str
    config_digest: str


def _encode_string(value: str) -> bytes:
    encoded = value.encode()
    if len(encoded) > 65535:
        raise ValueError("STAIR wire string is too long")
    return struct.pack("<H", len(encoded)) + encoded


def _decode_string(payload: bytes, cursor: int = 0) -> tuple[str, int]:
    (length,) = struct.unpack_from("<H", payload, cursor)
    cursor += 2
    return payload[cursor : cursor + length].decode(), cursor + length


def encode_request(request: PlanRequest) -> bytes:
    fixed = _REQUEST.pack(
        request.slot,
        request.bin_count,
        request.slot_sequence,
        request.planning_round,
        request.snapshot_sequence,
        request.sample_sequence,
        request.stats_schema_epoch,
        bytes.fromhex(request.input_digest),
        bytes.fromhex(request.key_digest),
        bytes.fromhex(request.config_digest),
    )
    return _encode_string(request.model_id) + fixed


def decode_request(payload: bytes) -> PlanRequest:
    model_id, cursor = _decode_string(payload)
    if len(payload) != cursor + _REQUEST.size:
        raise RuntimeError("STAIR plan request length mismatch")
    values = _REQUEST.unpack_from(payload, cursor)
    return PlanRequest(model_id, *values[:7], *(value.hex() for value in values[7:]))


def encode_plan(plan: RebalancePlan) -> bytes:
    payload = bytearray(_encode_string(plan.model_id))
    payload.extend(
        _PLAN.pack(
            plan.planning_round,
            plan.snapshot_sequence,
            plan.sample_sequence,
            plan.stats_schema_epoch,
            bytes.fromhex(plan.config_digest),
            bytes.fromhex(plan.topology_digest),
            len(plan.layers),
        )
    )
    for layer in plan.layers:
        payload.extend(
            _LAYER.pack(
                layer.layer_idx,
                layer.base_placement_epoch,
                layer.current_score.mean,
                layer.current_score.p95,
                layer.current_score.maximum,
                layer.candidate_score.mean,
                layer.candidate_score.p95,
                layer.candidate_score.maximum,
            )
        )
        for value in (layer.old_placement, layer.new_placement, layer.source_rank, layer.source_slot):
            payload.extend(np.asarray(value, dtype="<i4").tobytes(order="C"))
    payload.extend(bytes.fromhex(plan.digest()))
    return bytes(payload)


def decode_plan(payload: bytes, placement_shape: tuple[int, int]) -> RebalancePlan:
    model_id, cursor = _decode_string(payload)
    values = _PLAN.unpack_from(payload, cursor)
    cursor += _PLAN.size
    planning_round, snapshot_sequence, sample_sequence, schema_epoch = values[:4]
    config_digest, topology_digest, layer_count = values[4], values[5], values[6]
    array_bytes = int(np.prod(placement_shape)) * 4
    layers = []
    for _ in range(layer_count):
        layer_values = _LAYER.unpack_from(payload, cursor)
        cursor += _LAYER.size
        arrays = []
        for _ in range(4):
            end = cursor + array_bytes
            arrays.append(np.frombuffer(payload[cursor:end], dtype="<i4").copy().reshape(placement_shape))
            cursor = end
        layers.append(
            LayerPlan(
                int(layer_values[0]),
                *arrays,
                int(layer_values[1]),
                BalanceScore(*layer_values[2:5]),
                BalanceScore(*layer_values[5:8]),
            )
        )
    if len(payload) != cursor + 32:
        raise RuntimeError("STAIR plan result length mismatch")
    plan = RebalancePlan(
        model_id,
        planning_round,
        snapshot_sequence,
        sample_sequence,
        schema_epoch,
        config_digest.hex(),
        topology_digest.hex(),
        tuple(layers),
    )
    if plan.digest() != payload[cursor:].hex():
        raise RuntimeError("STAIR plan digest mismatch")
    return plan
