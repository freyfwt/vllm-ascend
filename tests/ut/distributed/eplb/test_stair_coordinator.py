from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.stair_coordinator import PendingSnapshot, StairCoordinator


def test_coordinator_routes_execution_phase_to_registered_model():
    coordinator = StairCoordinator(StairConfig(), "decode", torch.device("cpu"), 4)
    runtime = SimpleNamespace(note_execution=lambda phase: setattr(runtime, "phase", phase))
    coordinator.models["key"] = runtime
    coordinator.note_execution("key", True)
    assert runtime.phase is True


class FakePlanner:
    def __init__(self):
        self.calls = []

    def submit(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return True


def test_pending_snapshots_use_stable_cross_model_arbitration():
    coordinator = object.__new__(StairCoordinator)
    coordinator.planner = FakePlanner()
    coordinator._submitted = None
    first = SimpleNamespace(model_id="b")
    second = SimpleNamespace(model_id="a")
    values = dict(
        bin_sums=np.zeros((1, 1, 1), dtype=np.int64),
        bin_lengths=np.ones(1, dtype=np.int64),
        placements=np.zeros((1, 1, 1), dtype=np.int32),
        placement_epochs=np.zeros(1, dtype=np.int64),
        accepted_scores=np.full(1, np.nan),
        planning_round=1,
        snapshot_sequence=1,
        sample_sequence=1,
        key_digest="11" * 32,
    )
    coordinator._pending = {
        "b": PendingSnapshot(runtime=first, **values),
        "a": PendingSnapshot(runtime=second, **values),
    }
    coordinator.stats_schema_epoch = 0
    coordinator._try_submit()
    assert coordinator.planner.calls[0][0][0] == "a"
    assert coordinator._submitted.runtime.model_id == "a"


def test_planner_failure_disables_before_transfer(monkeypatch):
    coordinator = StairCoordinator(StairConfig(planner_restart_limit=0), "all", torch.device("cpu"), 4)
    coordinator.topology = SimpleNamespace()
    coordinator.models = {"model": SimpleNamespace()}
    coordinator.poll_local_plan = lambda: (_ for _ in ()).throw(RuntimeError("failed"))
    group = SimpleNamespace(
        cpu_group=object(),
        device_group=SimpleNamespace(rank=lambda: 0),
    )
    monkeypatch.setattr("vllm_ascend.distributed.eplb.stair_coordinator.get_ep_group", lambda: group)
    monkeypatch.setattr("vllm_ascend.distributed.eplb.stair_coordinator.dist.get_global_rank", lambda *_: 0)

    def broadcast(value, **_kwargs):
        assert value[0] == -1

    monkeypatch.setattr("vllm_ascend.distributed.eplb.stair_coordinator.dist.broadcast", broadcast)
    with patch("vllm_ascend.distributed.eplb.stair_coordinator.logger.exception"):
        coordinator.poll_and_broadcast()
    assert coordinator.disabled


def test_planner_failure_restarts_before_disabling(monkeypatch):
    coordinator = StairCoordinator(StairConfig(), "all", torch.device("cpu"), 4)
    coordinator.topology = SimpleNamespace()
    coordinator.models = {"model": SimpleNamespace()}
    coordinator.poll_local_plan = lambda: (_ for _ in ()).throw(RuntimeError("failed"))
    coordinator._restart_planner = lambda: True
    group = SimpleNamespace(cpu_group=object(), device_group=SimpleNamespace(rank=lambda: 0))
    monkeypatch.setattr("vllm_ascend.distributed.eplb.stair_coordinator.get_ep_group", lambda: group)
    monkeypatch.setattr("vllm_ascend.distributed.eplb.stair_coordinator.dist.get_global_rank", lambda *_: 0)
    monkeypatch.setattr("vllm_ascend.distributed.eplb.stair_coordinator.dist.broadcast", lambda *_args, **_kwargs: None)
    coordinator.poll_and_broadcast()
    assert not coordinator.disabled


def test_startup_identity_rejects_rank_mismatch(monkeypatch):
    coordinator = StairCoordinator(StairConfig(), "all", torch.device("cpu"), 4)
    coordinator.topology = SimpleNamespace(digest=lambda: "topology")
    coordinator.models = {}
    group = MagicMock()
    group.size.return_value = 2

    def all_gather(outputs, local, **_kwargs):
        outputs[0].copy_(local)
        outputs[1].zero_()

    monkeypatch.setattr("vllm_ascend.distributed.eplb.stair_coordinator.dist.all_gather", all_gather)
    with pytest.raises(RuntimeError, match="differs across EP ranks"):
        coordinator._validate_startup_identity(group)
