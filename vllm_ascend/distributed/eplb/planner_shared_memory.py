# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Anonymous double-buffered storage for STAIR planner snapshots."""

import hashlib
import mmap
import os
import tempfile
from dataclasses import dataclass

import numpy as np


_ALIGNMENT = 64


def _align(value: int) -> int:
    return (value + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


@dataclass(frozen=True)
class SharedSnapshotShape:
    max_bins: int
    num_layers: int
    num_experts: int
    num_ranks: int
    slots_per_rank: int


@dataclass(frozen=True)
class SharedSnapshotSpec:
    fd: int
    size: int
    slot_stride: int
    offsets: tuple[int, int, int, int, int]
    shape: SharedSnapshotShape


class SharedSnapshotBuffer:
    """Own or attach to two fixed-layout snapshot slots."""

    def __init__(self, spec: SharedSnapshotSpec, *, owner: bool) -> None:
        self.spec = spec
        self._owner = owner
        self._memory = mmap.mmap(spec.fd, spec.size, access=mmap.ACCESS_WRITE)
        self._next_slot = 0
        self._sequence = 0

    @classmethod
    def create(cls, shape: SharedSnapshotShape) -> "SharedSnapshotBuffer":
        dimensions = (
            (shape.max_bins, shape.num_layers, shape.num_experts),
            (shape.max_bins,),
            (shape.num_layers, shape.num_ranks, shape.slots_per_rank),
            (shape.num_layers,),
            (shape.num_layers,),
        )
        dtypes = (np.dtype("<i8"), np.dtype("<i8"), np.dtype("<i4"), np.dtype("<i8"), np.dtype("<f8"))
        offsets: list[int] = []
        cursor = 0
        for dimensions_for_value, dtype in zip(dimensions, dtypes):
            offsets.append(cursor)
            cursor = _align(cursor + int(np.prod(dimensions_for_value)) * dtype.itemsize)
        slot_stride = cursor
        fd, path = tempfile.mkstemp(prefix="vllm-stair-")
        os.unlink(path)
        os.ftruncate(fd, slot_stride * 2)
        return cls(SharedSnapshotSpec(fd, slot_stride * 2, slot_stride, tuple(offsets), shape), owner=True)

    @classmethod
    def attach(cls, spec: SharedSnapshotSpec) -> "SharedSnapshotBuffer":
        return cls(spec, owner=False)

    def _arrays(self, slot: int) -> tuple[np.ndarray, ...]:
        if slot not in (0, 1):
            raise ValueError(f"Invalid STAIR shared-memory slot {slot}")
        shape = self.spec.shape
        dimensions = (
            (shape.max_bins, shape.num_layers, shape.num_experts),
            (shape.max_bins,),
            (shape.num_layers, shape.num_ranks, shape.slots_per_rank),
            (shape.num_layers,),
            (shape.num_layers,),
        )
        dtypes = ("<i8", "<i8", "<i4", "<i8", "<f8")
        base = slot * self.spec.slot_stride
        return tuple(
            np.ndarray(dimensions[index], dtype=dtypes[index], buffer=self._memory, offset=base + offset)
            for index, offset in enumerate(self.spec.offsets)
        )

    @staticmethod
    def _digest(arrays: tuple[np.ndarray, ...], bin_count: int) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray((bin_count,), dtype="<i4").tobytes())
        for index, value in enumerate(arrays):
            selected = value[:bin_count] if index < 2 else value
            digest.update(selected.tobytes(order="C"))
        return digest.hexdigest()

    def write(
        self,
        bin_sums: np.ndarray,
        bin_lengths: np.ndarray,
        placements: np.ndarray,
        placement_epochs: np.ndarray,
        accepted_scores: np.ndarray,
    ) -> tuple[int, int, str]:
        bin_count = len(bin_lengths)
        if bin_count == 0 or bin_count > self.spec.shape.max_bins or bin_sums.shape[0] != bin_count:
            raise ValueError("STAIR snapshot bin count is outside the registered range")
        slot = self._next_slot
        arrays = self._arrays(slot)
        values = (bin_sums, bin_lengths, placements, placement_epochs, accepted_scores)
        for index, (target, value) in enumerate(zip(arrays, values)):
            target.fill(np.nan if index == 4 else 0)
            selected = target[:bin_count] if index < 2 else target
            if selected.shape != value.shape:
                raise ValueError(f"STAIR shared snapshot field {index} changed shape")
            np.copyto(selected, value, casting="safe")
        self._sequence += 1
        self._next_slot = 1 - slot
        return slot, self._sequence, self._digest(arrays, bin_count)

    def read(self, slot: int, bin_count: int, expected_digest: str) -> tuple[np.ndarray, ...]:
        arrays = self._arrays(slot)
        if self._digest(arrays, bin_count) != expected_digest:
            raise RuntimeError("STAIR shared snapshot checksum mismatch")
        return tuple(value[:bin_count] if index < 2 else value for index, value in enumerate(arrays))

    def close(self) -> None:
        self._memory.close()
        if self._owner:
            os.close(self.spec.fd)
