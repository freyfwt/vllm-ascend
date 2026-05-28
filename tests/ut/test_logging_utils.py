import logging
import unittest

from vllm_ascend.logging_utils import (
    MICROSECOND_DATE_FORMAT,
    MICROSECOND_LOG_FORMAT,
    MicrosecondFormatter,
    configure_vllm_microsecond_logging,
)


class TestLoggingUtils(unittest.TestCase):
    def test_microsecond_formatter_uses_fine_grained_timestamp(self):
        formatter = MicrosecondFormatter(
            fmt=MICROSECOND_LOG_FORMAT,
            datefmt=MICROSECOND_DATE_FORMAT,
        )
        record = logging.LogRecord(
            name="vllm",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.created = 1780000000.123456

        self.assertTrue(formatter.formatTime(record, MICROSECOND_DATE_FORMAT).endswith(".123.456"))

    def test_configure_vllm_microsecond_logging_updates_existing_handlers(self):
        vllm_logger = logging.getLogger("vllm")
        original_handlers = vllm_logger.handlers[:]
        handler = logging.StreamHandler()

        try:
            vllm_logger.handlers = [handler]
            configure_vllm_microsecond_logging()

            self.assertIsInstance(handler.formatter, MicrosecondFormatter)
        finally:
            vllm_logger.handlers = original_handlers

    def test_configure_vllm_microsecond_logging_updates_propagated_handlers(self):
        vllm_logger = logging.getLogger("vllm")
        root_logger = logging.getLogger()
        original_vllm_handlers = vllm_logger.handlers[:]
        original_vllm_propagate = vllm_logger.propagate
        original_root_handlers = root_logger.handlers[:]
        handler = logging.StreamHandler()

        try:
            vllm_logger.handlers = []
            vllm_logger.propagate = True
            root_logger.handlers = [handler]
            configure_vllm_microsecond_logging()

            self.assertIsInstance(handler.formatter, MicrosecondFormatter)
        finally:
            vllm_logger.handlers = original_vllm_handlers
            vllm_logger.propagate = original_vllm_propagate
            root_logger.handlers = original_root_handlers
