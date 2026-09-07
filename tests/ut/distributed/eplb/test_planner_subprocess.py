from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.planner_subprocess import apply_affinity


def test_planner_affinity_auto_is_a_noop():
    with patch(
        "vllm_ascend.distributed.eplb.planner_subprocess.os.sched_setaffinity",
        create=True,
    ) as setter:
        apply_affinity(SimpleNamespace(pid=7), StairConfig())
    setter.assert_not_called()


def test_planner_affinity_strict_propagates_failure():
    config = StairConfig(planner_cpu_set=[2], planner_affinity_strict=True)
    with (
        patch(
            "vllm_ascend.distributed.eplb.planner_subprocess.os.sched_setaffinity",
            side_effect=OSError("unavailable"),
            create=True,
        ),
        pytest.raises(RuntimeError, match="affinity failed"),
    ):
        apply_affinity(SimpleNamespace(pid=7), config)
