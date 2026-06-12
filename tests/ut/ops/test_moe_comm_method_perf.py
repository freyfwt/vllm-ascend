import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl


class TestMoECommMethodPerf(unittest.TestCase):

    def test_apply_log2phy_records_perf_event(self):
        comm_impl = AllGatherCommImpl.__new__(AllGatherCommImpl)
        topk_ids = torch.tensor([[0, 1], [2, 3]])
        log2phy = torch.tensor([3, 2, 1, 0])
        start_event = MagicMock()
        end_event = MagicMock()
        fake_npu = SimpleNamespace(Event=MagicMock(side_effect=[start_event, end_event]))

        with (
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.torch.npu", fake_npu, create=True),
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.eplb_perf_logger.enabled", True),
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.eplb_perf_logger.cycle_round", 7),
        ):
            routed_topk_ids = comm_impl._apply_log2phy(topk_ids, log2phy)

        torch.testing.assert_close(routed_topk_ids, torch.tensor([[3, 2], [1, 0]]))
        start_event.record.assert_called_once()
        end_event.record.assert_called_once()
        self.assertEqual(
            comm_impl._log2phy_perf_events,
            [("log2phy_route_execute", 7, start_event, end_event)],
        )

    def test_log_ready_log2phy_perf_events_keeps_pending_event(self):
        comm_impl = AllGatherCommImpl.__new__(AllGatherCommImpl)
        ready_end_event = MagicMock(query=MagicMock(return_value=True))
        pending_end_event = MagicMock(query=MagicMock(return_value=False))
        ready_event = ("log2phy_route_execute", 1, MagicMock(), ready_end_event)
        pending_event = ("log2phy_route_execute", 2, MagicMock(), pending_end_event)
        comm_impl._log2phy_perf_events = [ready_event, pending_event]

        with patch("vllm_ascend.ops.fused_moe.moe_comm_method.eplb_perf_logger.log_npu_event") as mock_log:
            comm_impl._log_ready_log2phy_perf_events()

        mock_log.assert_called_once_with(ready_event)
        self.assertEqual(comm_impl._log2phy_perf_events, [pending_event])
