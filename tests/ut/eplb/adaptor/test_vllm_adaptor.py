import unittest
from unittest.mock import MagicMock, patch

import torch
from transformers import DeepseekV2Config

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.quantization.quant_type import QuantType


class TestVllmAdaptor(unittest.TestCase):
    def setUp(self):
        VllmEplbAdaptor._registered_moe_layers = []

        n_routed_experts = 256
        self.mock_layer = MagicMock()
        self.mock_layer.local_num_experts = n_routed_experts
        self.mock_layer.ep_rank = 0
        self.mock_layer.quant_type = QuantType.W8A8
        self.mock_layer.w13_weight_list = [torch.randn(256, 128) for _ in range(n_routed_experts)]
        self.mock_layer.w2_weight_list = [torch.randn(128, 256) for _ in range(n_routed_experts)]
        self.mock_layer.w13_weight_scale_fp32_list = [torch.tensor([1.0]) for _ in range(n_routed_experts)]
        self.mock_layer.w2_weight_scale_list = [torch.tensor([1.0]) for _ in range(n_routed_experts)]
        self.mock_layer.w13_weight = torch.randn(n_routed_experts, 256, 128)
        self.mock_layer.w2_weight = torch.randn(n_routed_experts, 128, 256)
        self.mock_layer.moe_load = torch.randn(n_routed_experts)
        self.mock_layer.global_expert_map = torch.arange(n_routed_experts * 4).reshape(n_routed_experts, 4)
        self.mock_layer.get_log2phy_map.return_value = torch.arange(4)
        self.mock_layer.clear_moe_load = MagicMock()
        VllmEplbAdaptor.register_layer(self.mock_layer)

        mock_model = MagicMock()
        mock_model.model.named_parameters.return_value = dict()
        config = DeepseekV2Config(n_routed_experts=n_routed_experts)
        mock_model.config = config
        del mock_model.language_model
        self.model = mock_model
        num_dense_layers = getattr(config, "first_k_dense_replace", 0)
        self.model.model.layers[num_dense_layers].mlp.experts.quant_type = QuantType.W8A8

        self.mock_rank = patch("vllm_ascend.eplb.adaptor.vllm_adaptor.dist.get_rank", return_value=0).start()
        self.mock_size = patch("vllm_ascend.eplb.adaptor.vllm_adaptor.dist.get_world_size", return_value=4).start()

    def _build_small_fp16_adaptor(self):
        VllmEplbAdaptor._registered_moe_layers = []
        layer = MagicMock()
        layer.local_num_experts = 2
        layer.ep_rank = 0
        layer.w13_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        layer.w2_weight = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        layer.moe_load = torch.zeros(2)
        layer.global_expert_map = torch.tensor([[0, 1, -1, -1], [-1, -1, 0, 1]], dtype=torch.int32)
        layer.get_log2phy_map.return_value = torch.arange(4)
        VllmEplbAdaptor.register_layer(layer)

        model = MagicMock()
        model.quant_config = None
        model.config.first_k_dense_replace = 0
        del model.language_model
        return VllmEplbAdaptor(model)

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    def test_init_fp16(self, mock_func):
        self.model.quant_config = None
        VllmEplbAdaptor(self.model)

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    @patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config")
    def test_init_w8a8(self, mock_get_config, mock_func):
        mock_config = MagicMock()
        mock_config.enable_fused_mc2 = 0
        mock_get_config.return_value = mock_config
        VllmEplbAdaptor(self.model)

    @patch("torch.empty_like", return_value=torch.zeros(16, 32))
    @patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config")
    def test_language_model_w8a8(self, mock_get_config, mock_func):
        mock_config = MagicMock()
        mock_config.enable_fused_mc2 = 0
        mock_get_config.return_value = mock_config
        model = MagicMock()
        model.language_model = self.model
        model.config.text_config = self.model.config
        VllmEplbAdaptor(model)

    def test_pp_eplb_adaptor_init_with_registered_layer(self):
        """PP+EPLB: adaptor picks up MoE layers registered via register_layer."""
        VllmEplbAdaptor._registered_moe_layers = []
        layer = MagicMock()
        layer.local_num_experts = 4
        layer.ep_rank = 0
        layer.quant_type = QuantType.W8A8
        layer.w13_weight_list = [torch.randn(256, 128) for _ in range(4)]
        layer.w2_weight_list = [torch.randn(128, 256) for _ in range(4)]
        layer.w13_weight_scale_fp32_list = [torch.tensor([1.0]) for _ in range(4)]
        layer.w2_weight_scale_list = [torch.tensor([1.0]) for _ in range(4)]
        layer.moe_load = torch.randn(4)
        layer.global_expert_map = torch.arange(16).reshape(4, 4)
        layer.get_log2phy_map.return_value = torch.arange(4)
        VllmEplbAdaptor.register_layer(layer)

        with patch("vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.enable_fused_mc2 = 0
            mock_get_config.return_value = mock_config
            model = MagicMock()
            model.quant_config = MagicMock()
            model.config.first_k_dense_replace = 0
            del model.language_model
            adaptor = VllmEplbAdaptor(model)

        self.assertEqual(adaptor.num_moe_layers, 1)
        self.assertEqual(adaptor.num_local_experts, 4)
        self.assertEqual(adaptor.ep_rank, 0)

    def test_init_expert_weight_stats_gathers_all_ranks(self):
        adaptor = self._build_small_fp16_adaptor()
        adaptor.get_global_expert_map()
        comm_group = MagicMock()
        comm_group.world_size = 2
        comm_group.device_group = object()
        remote_stats = {(0, 2, "w13_weight"): {"max": 10.0, "min": 9.0, "mean": 9.5, "rank": 1, "local_expert_id": 0}}

        def fake_all_gather_object(gathered_stats, local_stats, group=None):
            gathered_stats[0] = local_stats
            gathered_stats[1] = remote_stats

        with patch(
            "vllm_ascend.eplb.adaptor.vllm_adaptor.dist.all_gather_object",
            side_effect=fake_all_gather_object,
        ) as mock_all_gather:
            adaptor.init_expert_weight_stats(comm_group)

        mock_all_gather.assert_called_once()
        self.assertEqual(adaptor.initial_expert_weight_stats[(0, 0, "w13_weight")]["max"], 2.0)
        self.assertEqual(adaptor.initial_expert_weight_stats[(0, 2, "w13_weight")]["mean"], 9.5)

    def test_do_update_expert_weight_warns_on_stats_mismatch(self):
        adaptor = self._build_small_fp16_adaptor()
        adaptor.get_global_expert_map()
        adaptor.init_expert_weight_stats(None)
        adaptor.buffer_tensor_list[0][0].copy_(torch.tensor([10.0, 20.0]))
        adaptor.buffer_tensor_list[0][1].copy_(torch.tensor([30.0, 40.0]))

        with patch("vllm_ascend.eplb.adaptor.vllm_adaptor.logger.warning") as mock_warning:
            adaptor.do_update_expert_weight(0, 0, 0, 0)

        self.assertTrue(mock_warning.called)

    def test_do_update_expert_weight_logs_error_on_logical_expert_mismatch(self):
        adaptor = self._build_small_fp16_adaptor()
        adaptor.get_global_expert_map()
        adaptor.init_expert_weight_stats(None)

        with patch("vllm_ascend.eplb.adaptor.vllm_adaptor.logger.error") as mock_error:
            adaptor.do_update_expert_weight(0, 0, 0, 1)

        mock_error.assert_called_once()

    def tearDown(self):
        self.mock_rank.stop()
        self.mock_size.stop()
        VllmEplbAdaptor._registered_moe_layers = []


if __name__ == "__main__":
    unittest.main()
