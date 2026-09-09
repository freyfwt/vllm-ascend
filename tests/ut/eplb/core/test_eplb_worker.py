# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend.eplb.core import eplb_worker
from vllm_ascend.eplb.core.eplb_worker import EplbWorker


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(eplb_worker, "logger", logging.getLogger(__name__))
    monkeypatch.setattr(eplb_worker, "get_ep_group", lambda: SimpleNamespace(rank_in_group=0))
    policy = Mock()
    monkeypatch.setattr(eplb_worker.PolicyFactory, "generate_policy", lambda _: policy)
    placement = torch.tensor([[[0, 1], [2, 0]]])
    # Expert 0 has two replicas but all its traffic reaches rank 0.
    loads = torch.tensor([[[100, 10], [10, 0]]])
    result = EplbWorker({"moe_load": loads}, policy_type=2)
    result.old_expert_maps = result.local2global(placement)
    result.num_local_experts = 2
    result.policy.rebalance_experts.return_value = (False, None, placement.clone())
    return result


def test_observed_load_keeps_replica_skew(worker, caplog):
    old_maps = worker.old_expert_maps.clone()
    loads = worker.shared_dict["moe_load"].clone()
    with caplog.at_level("INFO", logger=eplb_worker.logger.name):
        worker.do_update()

    assert "Observed rank load imbalance (collection window): layer_mean=1.833 layer_max=1.833" in caplog.text
    assert "worst_local_layer=0 worst_rank=0 active_layers=1/1" in caplog.text
    assert "Expert hotness imbalance, current: mean=1.000 max=1.000, updated: mean=1.000 max=1.000" in caplog.text
    assert worker.latest_expert_hotness == {
        "current_mean": 1.0,
        "current_max": 1.0,
        "update_mean": 1.0,
        "update_max": 1.0,
        "current_imbalance_list": [1.0],
        "update_imbalance_list": [1.0],
    }
    torch.testing.assert_close(worker.old_expert_maps, old_maps)
    torch.testing.assert_close(worker.policy.rebalance_experts.call_args.args[1], loads)


def test_multistage_log_reports_window_and_sample_imbalance(worker, caplog):
    worker.multi_stage = True
    loads = torch.tensor([[[[100, 0], [0, 0]]], [[[0, 0], [0, 100]]]])
    worker.shared_dict["moe_load"] = loads.clone()

    with caplog.at_level("INFO", logger=eplb_worker.logger.name):
        worker.do_update()

    # Each sample is skewed, but their accumulated window is balanced.
    assert "Observed rank load imbalance (collection window): layer_mean=1.000 layer_max=1.000" in caplog.text
    assert "Observed rank load imbalance (per collection sample): sample_mean=2.000 sample_max=2.000" in caplog.text
    assert "worst_sample=0 worst_local_layer=0 worst_rank=0" in caplog.text
    torch.testing.assert_close(worker.policy.rebalance_experts.call_args.args[1], loads)


def test_empty_layers_do_not_dilute_observed_imbalance():
    mean, maximum, layers = EplbWorker._compute_rank_imbalance(np.array([[0, 0], [20, 0]]), return_list=True)

    assert mean == maximum == 2.0
    assert np.isnan(layers[0])
    assert layers[1] == 2.0


@pytest.mark.parametrize("multi_stage", [False, True])
def test_no_samples_are_not_reported_as_balanced(worker, caplog, multi_stage):
    worker.shared_dict["moe_load"].zero_()
    worker.multi_stage = multi_stage
    if multi_stage:
        worker.shared_dict["moe_load"] = worker.shared_dict["moe_load"].unsqueeze(0)
    with caplog.at_level("INFO", logger=eplb_worker.logger.name), np.errstate(invalid="ignore"):
        worker.do_update()

    assert "Observed rank load imbalance (collection window): layer_mean=nan layer_max=nan" in caplog.text
    assert "worst_local_layer=-1 worst_rank=-1 active_layers=0/1" in caplog.text
    if multi_stage:
        assert "sample_mean=nan sample_max=nan" in caplog.text
        assert "worst_sample=-1 worst_local_layer=-1 worst_rank=-1" in caplog.text


def test_nonzero_rank_does_not_log_observed_load(worker, caplog):
    worker.rank_id = 1
    with caplog.at_level("INFO", logger=eplb_worker.logger.name):
        worker.do_update()

    assert "Observed rank load imbalance" not in caplog.text
    worker.policy.rebalance_experts.assert_called_once()
