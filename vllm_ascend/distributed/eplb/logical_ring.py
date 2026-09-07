# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Device-resident logical expert load history for STAIR."""

import hashlib

import torch


class LogicalLoadRing:
    """Store logical loads using the mapping that was active for each step."""

    def __init__(
        self,
        capacity: int,
        num_layers: int,
        num_logical_experts: int,
        device: torch.device,
    ) -> None:
        if min(capacity, num_layers, num_logical_experts) <= 0:
            raise ValueError("STAIR logical ring dimensions must be positive")
        self.capacity = capacity
        self.num_logical_experts = num_logical_experts
        self.values = torch.zeros(
            capacity,
            num_layers,
            num_logical_experts,
            dtype=torch.int64,
            device=device,
        )
        self.outer_step_keys: list[int | None] = [None] * capacity
        self.write_cursor = 0
        self.valid_size = 0
        self.sample_sequence = 0

    def record(
        self,
        physical_load: torch.Tensor,
        physical_to_logical: torch.Tensor,
        outer_step_key: int,
    ) -> None:
        if physical_load.shape != physical_to_logical.shape or physical_load.ndim != 2:
            raise ValueError("STAIR load and mapping must share [layers, physical experts]")
        if physical_load.shape[0] != self.values.shape[1]:
            raise ValueError("STAIR load layer count changed")
        target = self.values[self.write_cursor]
        target.zero_()
        target.scatter_add_(1, physical_to_logical.long(), physical_load.to(torch.int64))
        self.outer_step_keys[self.write_cursor] = outer_step_key
        self.write_cursor = (self.write_cursor + 1) % self.capacity
        self.valid_size = min(self.valid_size + 1, self.capacity)
        self.sample_sequence += 1

    def chronological(self) -> torch.Tensor:
        if self.valid_size == 0:
            return self.values[:0]
        start = (self.write_cursor - self.valid_size) % self.capacity
        if start + self.valid_size <= self.capacity:
            return self.values[start : start + self.valid_size]
        return torch.cat((self.values[start:], self.values[: self.write_cursor]), dim=0)

    def chronological_keys(self) -> tuple[int, ...]:
        if self.valid_size == 0:
            return ()
        start = (self.write_cursor - self.valid_size) % self.capacity
        keys = [self.outer_step_keys[(start + index) % self.capacity] for index in range(self.valid_size)]
        assert all(key is not None for key in keys)
        return tuple(key for key in keys if key is not None)

    def compressed_sums(self, sample_size: int) -> tuple[torch.Tensor, tuple[int, ...]]:
        if self.valid_size == 0 or sample_size <= 0:
            raise ValueError("STAIR compression requires recorded samples and positive sample_size")
        values = self.chronological()
        bins = min(sample_size, self.valid_size)
        boundaries = [index * self.valid_size // bins for index in range(bins + 1)]
        lengths = tuple(end - start for start, end in zip(boundaries[:-1], boundaries[1:]))
        sums = torch.stack([values[start:end].sum(dim=0, dtype=torch.int64) for start, end in zip(boundaries[:-1], boundaries[1:])])
        return sums, lengths

    def key_digest(self) -> str:
        payload = b"".join(key.to_bytes(8, "little", signed=True) for key in self.chronological_keys())
        return hashlib.sha256(payload).hexdigest()

    def clear(self) -> None:
        self.write_cursor = 0
        self.valid_size = 0
        self.outer_step_keys[:] = [None] * self.capacity
