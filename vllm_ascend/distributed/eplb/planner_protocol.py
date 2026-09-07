# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Versioned, pickle-free local wire protocol for the STAIR planner."""

import struct
from dataclasses import dataclass
from enum import IntEnum
from multiprocessing.connection import Connection

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.planner_shared_memory import SharedSnapshotShape, SharedSnapshotSpec
from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology

PROTOCOL_VERSION = 1
_MAGIC = b"STAIRPLN"
_HEADER = struct.Struct("<8sHHII")
_SPEC = struct.Struct("<iQQ5Q5I")
_CONFIG = struct.Struct("<Id?d?ddIdddIIIId")

class WireOp(IntEnum):
    INITIALIZE = 1
    ACK = 2
    PLAN = 3
    RESULT = 4
    SHUTDOWN = 5
    ERROR = 6

@dataclass(frozen=True)
class PlannerRegistration:
    model_id: str
    shared: SharedSnapshotSpec
    topology: RankTopology
    config: StairConfig

def crc32c(payload: bytes) -> int:
    value = 0xFFFFFFFF
    for byte in payload:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
    return value ^ 0xFFFFFFFF


def send_frame(connection: Connection, operation: WireOp, payload: bytes = b"") -> None:
    header = _HEADER.pack(_MAGIC, PROTOCOL_VERSION, int(operation), len(payload), crc32c(payload))
    connection.send_bytes(header + payload)


def receive_frame(connection: Connection) -> tuple[WireOp, bytes]:
    frame = connection.recv_bytes()
    if len(frame) < _HEADER.size:
        raise RuntimeError("Truncated STAIR planner frame")
    magic, version, operation, length, checksum = _HEADER.unpack_from(frame)
    payload = frame[_HEADER.size :]
    if magic != _MAGIC or version != PROTOCOL_VERSION:
        raise RuntimeError("STAIR planner protocol header mismatch")
    if len(payload) != length or crc32c(payload) != checksum:
        raise RuntimeError("STAIR planner frame checksum mismatch")
    try:
        return WireOp(operation), payload
    except ValueError as error:
        raise RuntimeError(f"Unknown STAIR planner operation {operation}") from error


def _encode_config(config: StairConfig) -> bytes:
    return _CONFIG.pack(
        config.sample_size,
        config.z_score,
        config.use_covariance,
        config.imbalance_threshold,
        config.hysteresis_enabled,
        config.hysteresis_relative,
        config.hysteresis_absolute,
        config.max_expert_transfers_per_rank_pair,
        config.min_relative_score_improvement,
        config.min_absolute_score_improvement,
        config.p95_regression_tolerance,
        config.experimental_flash_tree_depth,
        config.experimental_flash_tree_width,
        config.experimental_max_candidates_per_layer,
        config.experimental_lpt_max_backtracks,
        config.experimental_score_tie_tolerance,
    )


def _decode_config(payload: bytes) -> StairConfig:
    values = _CONFIG.unpack(payload)
    names = (
        "sample_size",
        "z_score",
        "use_covariance",
        "imbalance_threshold",
        "hysteresis_enabled",
        "hysteresis_relative",
        "hysteresis_absolute",
        "max_expert_transfers_per_rank_pair",
        "min_relative_score_improvement",
        "min_absolute_score_improvement",
        "p95_regression_tolerance",
        "experimental_flash_tree_depth",
        "experimental_flash_tree_width",
        "experimental_max_candidates_per_layer",
        "experimental_lpt_max_backtracks",
        "experimental_score_tie_tolerance",
    )
    return StairConfig(**dict(zip(names, values)))


def encode_registration(registration: PlannerRegistration) -> bytes:
    model_id = registration.model_id.encode()
    if len(model_id) > 65535:
        raise ValueError("STAIR model id is too long")
    spec, shape = registration.shared, registration.shared.shape
    fixed = _SPEC.pack(
        spec.fd,
        spec.size,
        spec.slot_stride,
        *spec.offsets,
        shape.max_bins,
        shape.num_layers,
        shape.num_experts,
        shape.num_ranks,
        shape.slots_per_rank,
    )
    topology = struct.pack(
        f"<{shape.num_ranks * 2}i",
        *registration.topology.node_by_ep_rank,
        *registration.topology.eplb_rank_by_ep_rank,
    )
    return struct.pack("<H", len(model_id)) + model_id + fixed + topology + _encode_config(registration.config)


def decode_registration(payload: bytes) -> PlannerRegistration:
    (model_length,) = struct.unpack_from("<H", payload)
    cursor = 2
    model_id = payload[cursor : cursor + model_length].decode()
    cursor += model_length
    values = _SPEC.unpack_from(payload, cursor)
    cursor += _SPEC.size
    fd, size, stride, *rest = values
    offsets, dimensions = tuple(rest[:5]), tuple(rest[5:])
    shape = SharedSnapshotShape(*dimensions)
    count = shape.num_ranks * 2
    topology_values = struct.unpack_from(f"<{count}i", payload, cursor)
    cursor += count * 4
    topology = RankTopology(tuple(topology_values[: shape.num_ranks]), tuple(topology_values[shape.num_ranks :]))
    config = _decode_config(payload[cursor : cursor + _CONFIG.size])
    if cursor + _CONFIG.size != len(payload):
        raise RuntimeError("STAIR registration has trailing bytes")
    spec = SharedSnapshotSpec(fd, size, stride, offsets, shape)
    return PlannerRegistration(model_id, spec, topology, config)
