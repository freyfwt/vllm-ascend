# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack, contextmanager, nullcontext
from unittest import TestCase
from unittest.mock import patch

import torch

from vllm_ascend.worker.logits_anomaly_monitor import LogitsAnomalyMonitor


class _FakeStream:
    def wait_stream(self, _stream) -> None:
        pass


class _FakeEvent:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.synchronize_calls = 0

    def record(self, _stream=None) -> None:
        pass

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self.ready = True


@contextmanager
def _monitor_context(events: list[_FakeEvent], ready: bool = True):
    def new_event():
        event = _FakeEvent(ready)
        events.append(event)
        return event

    with ExitStack() as stack:
        stack.enter_context(patch("vllm_ascend.worker.logits_anomaly_monitor.PIN_MEMORY", False))
        stack.enter_context(
            patch(
                "vllm_ascend.worker.logits_anomaly_monitor.torch.npu.Stream",
                return_value=_FakeStream(),
            )
        )
        stack.enter_context(
            patch(
                "vllm_ascend.worker.logits_anomaly_monitor.torch.npu.Event",
                side_effect=new_event,
            )
        )
        stack.enter_context(
            patch(
                "vllm_ascend.worker.logits_anomaly_monitor.torch.npu.current_stream",
                return_value=_FakeStream(),
            )
        )
        stack.enter_context(
            patch(
                "vllm_ascend.worker.logits_anomaly_monitor.torch.npu.stream",
                return_value=nullcontext(),
            )
        )
        yield


class TestLogitsAnomalyMonitor(TestCase):
    def test_clean_logits_do_not_warn(self):
        events: list[_FakeEvent] = []
        with _monitor_context(events), patch("vllm_ascend.worker.logits_anomaly_monitor.logger.warning") as warning:
            monitor = LogitsAnomalyMonitor()
            for _ in range(3):
                monitor.check(torch.zeros(1, 4), "sampling")

        warning.assert_not_called()

    def test_reports_each_anomaly_two_iterations_later(self):
        events: list[_FakeEvent] = []
        bad_logits = torch.tensor([[float("nan"), float("inf"), float("-inf"), 0.0]])
        clean_logits = torch.zeros(1, 4)

        with _monitor_context(events), patch("vllm_ascend.worker.logits_anomaly_monitor.logger.warning") as warning:
            monitor = LogitsAnomalyMonitor()
            monitor.check(bad_logits, "sampling")
            monitor.check(clean_logits, "sampling")
            warning.assert_not_called()

            monitor.check(clean_logits, "sampling")

        warning.assert_called_once_with(
            "Detected %s in model logits before %s (reported %d iterations later).",
            "NaN, +Inf, -Inf",
            "sampling",
            2,
        )
        self.assertTrue(all(event.synchronize_calls == 0 for event in events))

    def test_retains_busy_slots_without_synchronizing_or_overwriting(self):
        events: list[_FakeEvent] = []
        with (
            _monitor_context(events, ready=False),
            patch("vllm_ascend.worker.logits_anomaly_monitor.logger.warning") as warning,
        ):
            monitor = LogitsAnomalyMonitor()
            bad_logits = torch.tensor([[float("nan")]])
            clean_logits = torch.zeros(1, 1)
            monitor.check(bad_logits, "rejection sampling")
            monitor.check(clean_logits, "sampling")
            monitor.check(clean_logits, "sampling")

            self.assertEqual(len(events), 3)
            self.assertTrue(all(event.synchronize_calls == 0 for event in events))
            warning.assert_not_called()

            events[0].ready = True
            monitor.check(clean_logits, "sampling")

        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[2], "rejection sampling")
        self.assertEqual(warning.call_args.args[3], 3)

    def test_flush_reports_pending_result_at_shutdown(self):
        events: list[_FakeEvent] = []
        with (
            _monitor_context(events, ready=False),
            patch("vllm_ascend.worker.logits_anomaly_monitor.logger.warning") as warning,
        ):
            monitor = LogitsAnomalyMonitor()
            monitor.check(torch.tensor([[float("inf")]]), "sampling")
            monitor.flush()

        warning.assert_called_once()
        self.assertEqual(events[0].synchronize_calls, 1)
