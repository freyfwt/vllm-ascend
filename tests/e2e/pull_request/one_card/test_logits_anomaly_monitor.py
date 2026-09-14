# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch

from vllm_ascend.worker.logits_anomaly_monitor import LogitsAnomalyMonitor


def test_logits_anomaly_monitor_on_npu():
    monitor = LogitsAnomalyMonitor()
    bad_logits = torch.tensor(
        [[float("nan"), float("inf"), float("-inf"), 0.0]],
        device="npu",
    )
    clean_logits = torch.zeros(1, 4, device="npu")

    with patch("vllm_ascend.worker.logits_anomaly_monitor.logger.warning") as warning:
        monitor.check(bad_logits, "rejection sampling")
        # Tests may synchronize to make the asynchronous copy deterministically
        # observable; the production check path itself never synchronizes.
        torch.npu.synchronize()
        monitor.check(clean_logits, "sampling")
        monitor.check(clean_logits, "sampling")

    warning.assert_called_once_with(
        "Detected %s in model logits before %s (reported %d iterations later).",
        "NaN, +Inf, -Inf",
        "rejection sampling",
        2,
    )
