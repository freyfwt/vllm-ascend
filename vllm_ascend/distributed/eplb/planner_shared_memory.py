# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Typed anonymous shared memory for STAIR planner inputs."""

from __future__ import annotations

import hashlib
import mmap
import os
import tempfile

import numpy as np
import torch

from vllm_ascend.distributed.eplb.planner_protocol import PlannerModelRegistration, PlannerModelSpec

_ALIGNMENT = 64
_SLOT_COUNT = 2
_LOAD_DTYPE: np.dtype[np.float64] = np.dtype(np.float64)
_MAPPING_DTYPE: np.dtype[np.int64] = np.dtype(np.int64)


def _align(value: int) -> int:
    return (value + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


class StairSharedInputs:
    """Double-buffered load and mapping arrays backed by one inherited FD."""

    def __init__(
        self,
        spec: PlannerModelSpec,
        *,
        owner: bool,
    ) -> None:
        self.spec = spec
        self._owner = owner
        self._mmap = mmap.mmap(spec.shared_fd, spec.shared_size, access=mmap.ACCESS_WRITE)
        self._next_slot = 0
        self._sequence = 0

    @classmethod
    def create(cls, registration: PlannerModelRegistration) -> StairSharedInputs:
        load_bytes = int(np.prod(registration.load_shape, dtype=np.int64)) * _LOAD_DTYPE.itemsize
        mapping_bytes = int(np.prod(registration.mapping_shape, dtype=np.int64)) * _MAPPING_DTYPE.itemsize
        load_offset = 0
        mapping_offset = _align(load_bytes)
        slot_stride = _align(mapping_offset + mapping_bytes)
        shared_size = _SLOT_COUNT * slot_stride
        if hasattr(os, "memfd_create"):
            shared_fd = os.memfd_create(f"vllm-stair-{registration.model_id}", flags=0)
        else:  # pragma: no cover - Linux is the production target.
            shared_fd, shared_path = tempfile.mkstemp(prefix="vllm-stair-")
            os.unlink(shared_path)
        os.ftruncate(shared_fd, shared_size)
        spec = PlannerModelSpec(
            registration=registration,
            shared_fd=shared_fd,
            shared_size=shared_size,
            slot_stride=slot_stride,
            load_offset=load_offset,
            mapping_offset=mapping_offset,
        )
        return cls(spec, owner=True)

    @classmethod
    def attach(cls, spec: PlannerModelSpec) -> StairSharedInputs:
        return cls(spec, owner=False)

    def _array(self, slot: int, *, load: bool) -> np.ndarray:
        if slot < 0 or slot >= _SLOT_COUNT:
            raise ValueError(f"Invalid STAIR shared-memory slot {slot}.")
        registration = self.spec.registration
        offset = slot * self.spec.slot_stride
        if load:
            offset += self.spec.load_offset
            return np.ndarray(registration.load_shape, dtype=_LOAD_DTYPE, buffer=self._mmap, offset=offset)
        offset += self.spec.mapping_offset
        return np.ndarray(registration.mapping_shape, dtype=_MAPPING_DTYPE, buffer=self._mmap, offset=offset)

    @staticmethod
    def _digest(load: np.ndarray, mapping: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(load.data)
        digest.update(mapping.data)
        return digest.hexdigest()[:24]

    def write(self, load: torch.Tensor, mapping: torch.Tensor) -> tuple[int, int, str]:
        """Publish one complete slot and return slot, sequence, and digest."""
        registration = self.spec.registration
        if tuple(load.shape) != registration.load_shape:
            raise ValueError(
                f"STAIR load shape changed for model {registration.model_id}: "
                f"expected {registration.load_shape}, got {tuple(load.shape)}."
            )
        if tuple(mapping.shape) != registration.mapping_shape:
            raise ValueError(
                f"STAIR mapping shape changed for model {registration.model_id}: "
                f"expected {registration.mapping_shape}, got {tuple(mapping.shape)}."
            )
        slot = self._next_slot
        load_view = self._array(slot, load=True)
        mapping_view = self._array(slot, load=False)
        np.copyto(load_view, load.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy())
        np.copyto(mapping_view, mapping.detach().to(device="cpu", dtype=torch.long).contiguous().numpy())
        self._sequence += 1
        self._next_slot = (slot + 1) % _SLOT_COUNT
        return slot, self._sequence, self._digest(load_view, mapping_view)

    def read(self, slot: int, expected_digest: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and expose one immutable-for-the-request shared slot."""
        load_view = self._array(slot, load=True)
        mapping_view = self._array(slot, load=False)
        actual_digest = self._digest(load_view, mapping_view)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"STAIR shared-memory digest mismatch for model {self.spec.registration.model_id}: "
                f"expected {expected_digest}, got {actual_digest}."
            )
        return torch.from_numpy(load_view), torch.from_numpy(mapping_view)

    def close(self) -> None:
        self._mmap.close()
        if self._owner:
            os.close(self.spec.shared_fd)
