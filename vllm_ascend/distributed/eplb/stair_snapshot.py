# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Build aligned, phase-filtered STAIR snapshots at planning boundaries."""

import hashlib
from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup, all_gather, all_reduce

from vllm_ascend.distributed.eplb.logical_ring import LogicalLoadRing


@dataclass(frozen=True)
class SynchronizedSnapshot:
    bin_sums: torch.Tensor | None
    bin_lengths: tuple[int, ...]
    key_digest: str
    selected_keys: tuple[int, ...]


def phase_mask(group_metadata: torch.Tensor, phase: str) -> torch.Tensor:
    """Classify each model step from rank-local executed/prefill metadata."""
    if group_metadata.ndim != 3 or group_metadata.shape[-1] != 2:
        raise ValueError("STAIR phase metadata must be [ranks, steps, 2]")
    executed = group_metadata[..., 0].bool()
    prefill = group_metadata[..., 1].bool() & executed
    group_executed = executed.any(dim=0)
    group_prefill = prefill.any(dim=0)
    if phase == "all":
        return group_executed
    if phase == "prefill":
        return group_executed & group_prefill
    if phase == "decode":
        return group_executed & ~group_prefill
    raise ValueError(f"Unsupported STAIR load collection phase {phase!r}")


def sync_snapshot(
    ring: LogicalLoadRing,
    *,
    phase: str,
    sample_size: int,
    cpu_group: ProcessGroup,
    device_group: ProcessGroup,
    group_rank: int,
) -> SynchronizedSnapshot | None:
    """Return one rank-zero CPU snapshot after aligned device-side reduction."""
    world_size = cpu_group.size()
    progress = torch.tensor((ring.valid_size, ring.sample_sequence), dtype=torch.int64)
    progress_by_rank = [torch.empty_like(progress) for _ in range(world_size)]
    all_gather(progress_by_rank, progress, group=cpu_group)
    if any(not torch.equal(item, progress) for item in progress_by_rank) or ring.valid_size == 0:
        return None

    keys = torch.tensor(ring.chronological_keys(), dtype=torch.int64)
    keys_by_rank = [torch.empty_like(keys) for _ in range(world_size)]
    all_gather(keys_by_rank, keys, group=cpu_group)
    if any(not torch.equal(item, keys) for item in keys_by_rank):
        return None

    local_metadata = torch.tensor(ring.chronological_metadata(), dtype=torch.int32)
    metadata_by_rank = [torch.empty_like(local_metadata) for _ in range(world_size)]
    all_gather(metadata_by_rank, local_metadata, group=cpu_group)
    selected = phase_mask(torch.stack(metadata_by_rank), phase)
    selected_indices = selected.nonzero().flatten().tolist()
    if not selected_indices:
        return None

    bin_sums, lengths = ring.compressed_selected_sums(selected_indices, sample_size)
    all_reduce(bin_sums, group=device_group)
    selected_keys = keys[selected]
    digest = hashlib.sha256(selected_keys.numpy().astype("<i8", copy=False).tobytes()).hexdigest()
    cpu_sums = bin_sums.cpu() if group_rank == 0 else None
    return SynchronizedSnapshot(cpu_sums, lengths, digest, tuple(selected_keys.tolist()))
