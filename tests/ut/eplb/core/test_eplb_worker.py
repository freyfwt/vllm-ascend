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
    with caplog.at_level("INFO", logger=eplb_worker.logger.name):
        worker.do_update()

    stats = worker.latest_expert_hotness
    assert stats["observed_mean"] == pytest.approx(110 / 60)
    assert stats["observed_max"] == pytest.approx(110 / 60)
    # Existing metric keys retain their estimated-load meaning.
    assert stats["current_mean"] == 1.0
    assert stats["update_mean"] == 1.0
    assert "Observed rank load imbalance (collection window)" in caplog.text
    assert "worst_local_layer=0 worst_rank=0" in caplog.text
    assert "Estimated rank load imbalance (equal replica split, collection window)" in caplog.text
    assert "candidate:" in caplog.text


def test_candidate_metrics_follow_placement_validation(worker):
    # Missing expert 2: validation must fall back before scoring the candidate.
    worker.policy.rebalance_experts.return_value = (True, None, [[[0, 1], [0, 0]]])
    old_maps = worker.old_expert_maps.clone()

    worker.do_update()

    torch.testing.assert_close(worker.old_expert_maps, old_maps)
    assert worker.latest_expert_hotness["update_mean"] == worker.latest_expert_hotness["current_mean"]


def test_multistage_log_reports_window_and_sample_imbalance(worker, caplog):
    worker.multi_stage = True
    worker.shared_dict["moe_load"] = torch.tensor([[[[100, 0], [0, 0]]], [[[0, 0], [0, 100]]]])

    with caplog.at_level("INFO", logger=eplb_worker.logger.name):
        worker.do_update()

    # Each sample is skewed, but their accumulated window is balanced.
    assert worker.latest_expert_hotness["observed_mean"] == 1.0
    assert worker.latest_expert_hotness["observed_sample_mean"] == 2.0
    assert worker.latest_expert_hotness["observed_sample_max"] == 2.0
    assert "Observed rank load imbalance (per collection sample)" in caplog.text
    assert "worst_sample=0 worst_local_layer=0 worst_rank=0" in caplog.text
    assert worker.policy.rebalance_experts.call_args.args[1].shape == (2, 1, 2, 2)


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
    with caplog.at_level("INFO", logger=eplb_worker.logger.name):
        worker.do_update()

    stats = worker.latest_expert_hotness
    assert np.isnan(stats["observed_mean"])
    assert np.isnan(stats["current_mean"])
    assert np.isnan(stats["update_mean"])
    assert "worst_local_layer=-1 worst_rank=-1 active_layers=0/1" in caplog.text
    if multi_stage:
        assert np.isnan(stats["observed_sample_mean"])
        assert np.isnan(stats["observed_sample_max"])
        assert "worst_sample=-1 worst_local_layer=-1 worst_rank=-1" in caplog.text
