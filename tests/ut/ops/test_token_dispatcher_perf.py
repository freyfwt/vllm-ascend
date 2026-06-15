import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather


class TestTokenDispatcherPerf(unittest.TestCase):

    def test_apply_expert_map_mask_records_perf_event(self):
        dispatcher = TokenDispatcherWithAllGather.__new__(TokenDispatcherWithAllGather)
        topk_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        topk_ids = torch.tensor([[0, 1], [2, 3]])
        expert_map = torch.tensor([0, -1, 1, -1])
        start_event = MagicMock()
        end_event = MagicMock()
        fake_npu = SimpleNamespace(Event=MagicMock(side_effect=[start_event, end_event]))

        with (
            patch("vllm_ascend.ops.fused_moe.token_dispatcher.torch.npu", fake_npu, create=True),
            patch("vllm_ascend.ops.fused_moe.token_dispatcher.eplb_perf_logger.enabled", True),
            patch("vllm_ascend.ops.fused_moe.token_dispatcher.eplb_perf_logger.cycle_round", 7),
        ):
            masked_weights = dispatcher._apply_expert_map_mask(topk_weights, topk_ids, expert_map)

        torch.testing.assert_close(masked_weights, torch.tensor([[0.5, 0.0], [0.3, 0.0]]))
        start_event.record.assert_called_once()
        end_event.record.assert_called_once()
        self.assertEqual(
            dispatcher._expert_map_perf_events,
            [("expert_map_mask_execute", 7, start_event, end_event)],
        )

    def test_log_ready_expert_map_perf_events_keeps_pending_event(self):
        dispatcher = TokenDispatcherWithAllGather.__new__(TokenDispatcherWithAllGather)
        ready_end_event = MagicMock(query=MagicMock(return_value=True))
        pending_end_event = MagicMock(query=MagicMock(return_value=False))
        ready_event = ("expert_map_mask_execute", 1, MagicMock(), ready_end_event)
        pending_event = ("expert_map_mask_execute", 2, MagicMock(), pending_end_event)
        dispatcher._expert_map_perf_events = [ready_event, pending_event]

        with patch("vllm_ascend.ops.fused_moe.token_dispatcher.eplb_perf_logger.log_npu_event") as mock_log:
            dispatcher._log_ready_expert_map_perf_events()

        mock_log.assert_called_once_with(ready_event)
        self.assertEqual(dispatcher._expert_map_perf_events, [pending_event])
