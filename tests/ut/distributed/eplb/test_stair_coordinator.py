from types import SimpleNamespace

import torch

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.stair_coordinator import StairCoordinator


def test_coordinator_routes_execution_phase_to_registered_model():
    coordinator = StairCoordinator(StairConfig(), "decode", torch.device("cpu"), 4)
    runtime = SimpleNamespace(note_execution=lambda phase: setattr(runtime, "phase", phase))
    coordinator.models["key"] = runtime
    coordinator.note_execution("key", True)
    assert runtime.phase is True
