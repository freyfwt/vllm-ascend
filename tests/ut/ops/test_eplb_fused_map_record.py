# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Unit tests for the fused EPLB gating/mapping/recording route."""

import torch

from vllm_ascend.ops.triton import eplb_map_record
from vllm_ascend.ops.triton.eplb_map_record import (
    EPLB_FUSED_MAP_RECORD_MAX_TOKENS,
    gating_map_record,
)


def test_custom_ops_registered_with_fake_impl():
    """Importing the eplb ops module registers the fused ops on the vllm namespace."""
    import vllm_ascend.ops.fused_moe.eplb  # noqa: F401  (triggers registration)

    assert torch.ops.vllm.ascend_eplb_gating_top_k_map_record is not None
    assert torch.ops.vllm.ascend_eplb_hash_map_record is not None


def test_threshold_constant_in_batch_bucket_range():
    """The fused route only claims batches where it measured a win or parity."""
    assert 0 < EPLB_FUSED_MAP_RECORD_MAX_TOKENS <= 1024


def test_tokens_per_program_snaps_to_power_of_two():
    """Tile heights must be powers of two (backend parseSelect limitation)."""

    captured = {}

    class _FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                captured["ts"] = kwargs["TOKENS_PER_PROGRAM"]
                return torch.empty(0)

            return launch

    saved = eplb_map_record.gating_map_record_kernel
    eplb_map_record.gating_map_record_kernel = _FakeKernel()  # type: ignore[assignment]
    try:
        gating_map_record(
            torch.empty(1000, 256),
            None,
            torch.empty(1024, 256, dtype=torch.int32),
            torch.ones((), dtype=torch.int32),
            torch.ones((), dtype=torch.int32),
            torch.empty(288, dtype=torch.int32),
            local_expert_start=0,
            local_expert_count=32,
            k=8,
            k_group=4,
            group_count=8,
            routed_scaling_factor=2.5,
        )
    finally:
        eplb_map_record.gating_map_record_kernel = saved  # type: ignore[assignment]

    ts = captured["ts"]
    assert 2 <= ts <= 16
    assert ts & (ts - 1) == 0


def test_max_tokens_guard_matches_crossover():
    """T=512 measured parity and T=1024 a regression; the guard sits between."""
    assert EPLB_FUSED_MAP_RECORD_MAX_TOKENS in (256, 512)


def test_softmax_route_uses_global_normalization():
    """Softmax weights come from the full-expert softmax; renorm is optional."""
    from vllm_ascend.ops.triton import eplb_map_record as mod

    class _FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                captured["norm"] = kwargs["NORM_TYPE"]
                captured["renorm"] = kwargs["RENORM"]

            return launch

    captured = {}
    saved = mod.gating_map_record_kernel
    mod.gating_map_record_kernel = _FakeKernel()  # type: ignore[assignment]
    try:
        for norm, renorm in ((0, False), (0, True), (1, True)):
            mod.gating_map_record(
                torch.empty(64, 256),
                None,
                torch.empty(1024, 256, dtype=torch.int32),
                torch.ones((), dtype=torch.int32),
                torch.ones((), dtype=torch.int32),
                torch.empty(288, dtype=torch.int32),
                local_expert_start=0,
                local_expert_count=32,
                k=8,
                k_group=4,
                group_count=8,
                routed_scaling_factor=2.5,
                norm_type=norm,
                renorm=renorm,
            )
            assert captured["norm"] == norm
            assert captured["renorm"] == renorm
    finally:
        mod.gating_map_record_kernel = saved  # type: ignore[assignment]


def test_hash_route_lookup_domain():
    """tid2eid holds logical expert ids; the table maps them to physical."""
    from vllm_ascend.ops.triton import eplb_map_record as mod

    captured = {}

    class _FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                captured["k"] = kwargs["K"]
                captured["table_rows"] = kwargs["TABLE_ROWS"]

            return launch

    saved = mod.hash_map_record_kernel
    mod.hash_map_record_kernel = _FakeKernel()  # type: ignore[assignment]
    try:
        mod.hash_map_record(
            torch.empty(64, 256),
            torch.zeros(64, dtype=torch.int64),
            torch.zeros(4096, 8, dtype=torch.int32),
            torch.empty(1024, 256, dtype=torch.int32),
            torch.ones((), dtype=torch.int32),
            torch.ones((), dtype=torch.int32),
            torch.empty(288, dtype=torch.int32),
            local_expert_start=0,
            local_expert_count=32,
            k=8,
            routed_scaling_factor=2.5,
        )
    finally:
        mod.hash_map_record_kernel = saved  # type: ignore[assignment]

    assert captured["k"] == 8
    assert captured["table_rows"] == 1024
