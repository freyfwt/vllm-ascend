# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Explicit EP, EPLB, global-rank, and host identity discovery."""

import hashlib
import socket
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vllm_ascend.distributed.eplb.policy.stair_types import RankTopology


def build_topology(
    ep_global_ranks: list[int],
    eplb_global_ranks: list[int],
    node_digests: list[bytes],
) -> RankTopology:
    if len(ep_global_ranks) != len(node_digests) or set(ep_global_ranks) != set(eplb_global_ranks):
        raise ValueError("STAIR requires EP and EPLB groups to contain the same global ranks")
    node_ids: dict[bytes, int] = {}
    node_by_ep_rank = tuple(node_ids.setdefault(value, len(node_ids)) for value in node_digests)
    eplb_rank_by_ep_rank = tuple(eplb_global_ranks.index(rank) for rank in ep_global_ranks)
    return RankTopology(node_by_ep_rank, eplb_rank_by_ep_rank)


def _host_digest() -> bytes:
    identity = socket.gethostname().encode()
    machine_id = Path("/etc/machine-id")
    if machine_id.is_file():
        identity += b":" + machine_id.read_bytes().strip()
    return hashlib.sha256(identity).digest()


def discover_topology(ep_group: Any, eplb_group: Any) -> RankTopology:
    """All-gather canonical host identity and validate the group bijection."""
    ep_cpu_group = ep_group.cpu_group
    ep_globals = dist.get_process_group_ranks(ep_group.device_group)
    eplb_globals = dist.get_process_group_ranks(eplb_group.device_group)
    local = torch.tensor(list(_host_digest()), dtype=torch.uint8)
    gathered = [torch.empty_like(local) for _ in ep_globals]
    dist.all_gather(gathered, local, group=ep_cpu_group)
    return build_topology(ep_globals, eplb_globals, [bytes(value.tolist()) for value in gathered])
