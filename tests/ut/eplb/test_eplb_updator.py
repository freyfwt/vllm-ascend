import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ascend_config import EplbPerfLogMode
import vllm_ascend.eplb.eplb_updator as eplb_updator
from vllm_ascend.eplb.eplb_updator import EplbUpdator


class TestEplbUpdatorComputeAndSetMoeLoad(unittest.TestCase):
    def setUp(self):
        # ====================== 1. Mock environment ======================
        self.rank = 0
        self.world_size = 4
        self.device = torch.device("cpu")

        # mock dist
        p1 = patch("torch.distributed.get_rank", return_value=self.rank)
        p2 = patch("torch.distributed.get_world_size", return_value=self.world_size)
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        p1.start()
        p2.start()

        # ====================== 2. Mock comm group ======================
        self.mock_comm_group = MagicMock()

        def mock_all_gather(tensor, dim):
            gathered = torch.cat([tensor for _ in range(self.world_size)], dim=dim)
            return gathered

        self.mock_comm_group.all_gather = mock_all_gather

        p3 = patch("vllm_ascend.eplb.eplb_updator.get_dynamic_eplb_group", return_value=self.mock_comm_group)
        self.addCleanup(p3.stop)
        p3.start()

        # mock _PP in vllm.distributed.parallel_state (PP+EPLB support)
        # Patching the variable directly so that even the real get_pp_group()
        # (already imported into eplb_updator's namespace) reads a non-None _PP.
        self.mock_pp = MagicMock()
        self.mock_pp.rank_in_group = 0
        p4 = patch("vllm.distributed.parallel_state._PP", self.mock_pp)
        self.addCleanup(p4.stop)
        p4.start()

        # ====================== 3. Mock EplbUpdator ======================
        self.eplb_config = MagicMock()
        self.eplb_config.eplb_perf_log_mode = EplbPerfLogMode.DISABLED.value
        self.loader = MagicMock()
        self.eplb_process = MagicMock()
        self.process = MagicMock()
        self.eplb_process.shared_dict = {}

        self.updator = EplbUpdator(
            eplb_config=self.eplb_config, loader=self.loader, eplb_process=self.eplb_process, process=self.process
        )

        # ====================== 4. Mock adaptor ======================
        self.adaptor = MagicMock()
        self.adaptor.num_moe_layers = 4
        self.adaptor.num_dense_layers = 2
        self.mock_local_load = torch.randn(58, 100, 8, device=self.device)
        self.adaptor.get_rank_expert_workload.return_value = self.mock_local_load

        self.updator.set_adaptor(self.adaptor)

    def test_compute_and_set_moe_load_normal(self):
        self.updator.multi_stage = False

        moe_load = self.updator.compute_and_set_moe_load()

        self.assertEqual(moe_load.shape, (58, self.world_size, 100, 8))
        self.assertTrue("moe_load" in self.updator.shared_dict)
        self.assertEqual(moe_load.device.type, "cpu")
        self.assertEqual(moe_load.shape[1], self.world_size)

    def test_compute_and_set_moe_load_multi_stage(self):
        self.updator.multi_stage = True

        moe_load = self.updator.compute_and_set_moe_load()

        self.assertEqual(moe_load.shape, (100, 58, self.world_size, 8))
        self.assertTrue("moe_load" in self.updator.shared_dict)
        self.assertEqual(moe_load.device.type, "cpu")

    def test_forward_before_flushes_pending_perf_events(self):
        self.updator.get_update_info_flag = MagicMock(return_value=False)
        self.updator.update_expert_weight_flag = MagicMock(return_value=False)

        self.updator.forward_before()

        self.loader.log_ready_perf_events.assert_called_once()

    def test_forward_end_records_perf_event_for_eplb_work(self):
        self.updator.wakeup_eplb_worker_flag = MagicMock(return_value=False)
        self.updator.update_expert_weight_flag = MagicMock(return_value=True)
        self.updator.expert_map_record_path = None
        self.updator.update_iteration = MagicMock()
        self.updator.reqs = ["req"]
        self.loader.pending_perf_events = []
        start_event = MagicMock()
        end_event = MagicMock()
        fake_npu = SimpleNamespace(Event=MagicMock(side_effect=[start_event, end_event]))

        with (
            patch.object(eplb_updator.eplb_perf_logger, "enabled", True),
            patch.object(eplb_updator.torch, "npu", fake_npu, create=True),
        ):
            self.updator.forward_end()

        start_event.record.assert_called_once()
        end_event.record.assert_called_once()
        self.loader.update_expert_map_and_weight.assert_called_once_with(["req"])
        self.assertEqual(
            self.loader.pending_perf_events,
            [("forward_end_execute", self.updator.eplb_cycle_round, start_event, end_event)],
        )

    def test_forward_end_skips_perf_event_without_eplb_work(self):
        self.updator.wakeup_eplb_worker_flag = MagicMock(return_value=False)
        self.updator.update_expert_weight_flag = MagicMock(return_value=False)
        self.updator.expert_map_record_path = None
        self.updator.update_iteration = MagicMock()
        self.loader.pending_perf_events = []
        event_factory = MagicMock()

        with (
            patch.object(eplb_updator.eplb_perf_logger, "enabled", True),
            patch.object(eplb_updator.torch, "npu", SimpleNamespace(Event=event_factory), create=True),
        ):
            self.updator.forward_end()

        event_factory.assert_not_called()
        self.assertEqual(self.loader.pending_perf_events, [])


if __name__ == "__main__":
    unittest.main()
