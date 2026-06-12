import unittest
from unittest.mock import MagicMock, patch

from vllm_ascend.ascend_config import EplbPerfLogMode
from vllm_ascend.eplb.perf_logger import EplbPerfLogger


class TestEplbPerfLogger(unittest.TestCase):

    def test_configure_modes(self):
        perf_logger = EplbPerfLogger()

        perf_logger.configure(EplbPerfLogMode.DISABLED, 0)
        self.assertFalse(perf_logger.enabled)

        perf_logger.configure(EplbPerfLogMode.RANK0, 1)
        self.assertFalse(perf_logger.enabled)

        perf_logger.configure(EplbPerfLogMode.RANK0, 0)
        self.assertTrue(perf_logger.enabled)

        perf_logger.configure(EplbPerfLogMode.ALL_RANKS, 2)
        self.assertTrue(perf_logger.enabled)

    def test_log_npu_event_only_logs_ready_event(self):
        perf_logger = EplbPerfLogger()
        perf_logger.configure(EplbPerfLogMode.ALL_RANKS, 3)
        start_event = MagicMock()
        start_event.elapsed_time.return_value = 1.25
        end_event = MagicMock()

        end_event.query.return_value = False
        with patch("vllm_ascend.eplb.perf_logger.logger.info") as mock_info:
            perf_logger.log_npu_event("not_ready", 7, start_event, end_event)
        mock_info.assert_not_called()

        end_event.query.return_value = True
        with patch("vllm_ascend.eplb.perf_logger.logger.info") as mock_info:
            perf_logger.log_npu_event("ready", 7, start_event, end_event)
        mock_info.assert_called_once()
