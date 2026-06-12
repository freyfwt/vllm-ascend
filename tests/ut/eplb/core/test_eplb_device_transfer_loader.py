from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.eplb.core.eplb_device_transfer_loader as loader


class TestD2DExpertWeightLoader(unittest.TestCase):

    def _new_loader(self):
        with patch("vllm_ascend.eplb.core.eplb_device_transfer_loader.get_dynamic_eplb_group", return_value=None):
            return loader.D2DExpertWeightLoader()

    def _new_adaptor(self):
        adaptor = MagicMock()
        adaptor.expert_map_per_layer_cpu = {0: {10: torch.tensor(1), 20: torch.tensor(0)}}
        adaptor.expert_param_per_layer = {0: {0: [[torch.tensor([1.0])]], 1: [[torch.tensor([2.0])]]}}
        adaptor.expert_weight_key_per_layer = {0: "weight_key"}
        adaptor.buffer_tensor_list = {
            "weight_key": [[torch.tensor([3.0]), torch.tensor([4.0])], [torch.tensor([5.0]), torch.tensor([6.0])]]
        }
        return adaptor

    def _ready_loader(self):
        loader_obj = self._new_loader()
        loader_obj.comm_op_list = ["fake_op"]
        loader_obj.state = loader.ExpertWeightUpdateState.READY
        return loader_obj

    def test_generate_task_and_state_flow(self):
        loader_obj = self._new_loader()
        loader_obj.set_adator(self._new_adaptor())
        with (
            patch("torch.distributed.P2POp") as mock_p2p,
            patch("torch.distributed.isend", return_value="isend_op"),
            patch("torch.distributed.irecv", return_value="irecv_op"),
        ):
            mock_p2p.side_effect = lambda op, tensor, rank, group=None: (op, tensor, rank, group)
            loader_obj.state = loader.ExpertWeightUpdateState.READY
            loader_obj.generate_expert_d2d_transfer_task([(1, 10)], [(2, 20)], {20: torch.tensor(0)}, 0)
            self.assertIsNone(loader_obj.comm_op_list)
            loader_obj.state = loader.ExpertWeightUpdateState.WAITING
            loader_obj.generate_expert_d2d_transfer_task([], [], {}, 0)
            self.assertFalse(loader_obj.comm_op_list)
            self.assertEqual(loader_obj.state, loader.ExpertWeightUpdateState.READY)

    def test_async_transfer_and_update(self):
        adaptor = self._new_adaptor()
        loader_obj = self._ready_loader()
        loader_obj.set_adator(adaptor)
        reqs = []
        with patch("torch.distributed.batch_isend_irecv", return_value=[MagicMock(), MagicMock()]):
            loader_obj.async_expert_weight_transfer(reqs)
        self.assertEqual(loader_obj.state, loader.ExpertWeightUpdateState.TRANSFERRING)
        self.assertGreater(len(reqs), 0)

        mock_req = MagicMock()
        mock_req.wait.return_value = None
        loader_obj.recv_expert_list = [(0, 0)]
        loader_obj.updated_expert_map = {20: torch.tensor(0)}
        loader_obj.updated_log2phy_map = {"dummy": 1}
        loader_obj.layer_id = 0
        loader_obj.comm_op_list = ["op"]
        loader_obj.update_expert_map_and_weight([mock_req])

        adaptor.do_update_expert_map.assert_called_once()
        adaptor.do_update_log2phy_map.assert_called_once()
        adaptor.do_update_expert_weight.assert_called_once()
        self.assertEqual(loader_obj.state, loader.ExpertWeightUpdateState.WAITING)
        self.assertEqual(loader_obj.recv_expert_list, [])

    def test_async_transfer_records_d2d_events(self):
        loader_obj = self._ready_loader()
        start_event = MagicMock()
        end_event = MagicMock()
        fake_npu = SimpleNamespace(Event=MagicMock(side_effect=[start_event, end_event]))
        with (
            patch.object(loader.eplb_perf_logger, "enabled", True),
            patch.object(loader.torch, "npu", fake_npu, create=True),
            patch("torch.distributed.batch_isend_irecv", return_value=[]),
        ):
            loader_obj.async_expert_weight_transfer([])
        start_event.record.assert_called_once()
        end_event.record.assert_called_once()
        self.assertIs(loader_obj.d2d_start_event, start_event)
        self.assertIs(loader_obj.d2d_end_event, end_event)
        self.assertEqual(loader_obj.d2d_perf_event, ("d2d_transfer_execute", 0, start_event, end_event))

    def test_log_ready_perf_events_keeps_unready_events(self):
        loader_obj = self._new_loader()
        ready_end_event = MagicMock(query=MagicMock(return_value=True))
        pending_end_event = MagicMock(query=MagicMock(return_value=False))
        ready_event = ("ready", 1, MagicMock(), ready_end_event)
        pending_event = ("pending", 2, MagicMock(), pending_end_event)
        loader_obj.pending_perf_events = [ready_event, pending_event]
        with patch.object(loader.eplb_perf_logger, "log_npu_event") as mock_log_npu_event:
            loader_obj.log_ready_perf_events()
        mock_log_npu_event.assert_called_once_with(ready_event)
        self.assertEqual(loader_obj.pending_perf_events, [pending_event])

    def test_set_log2phy_map_and_invalid_state(self):
        adaptor = self._new_adaptor()
        loader_obj = self._new_loader()
        loader_obj.set_adator(adaptor)
        loader_obj.set_log2phy_map({"a": 1})
        self.assertEqual(loader_obj.updated_log2phy_map, {"a": 1})

        reqs = []
        loader_obj.async_expert_weight_transfer(reqs)
        self.assertEqual(reqs, [])
        loader_obj.state = loader.ExpertWeightUpdateState.READY
        loader_obj.update_expert_map_and_weight([])
        self.assertFalse(adaptor.do_update_expert_map.called)
