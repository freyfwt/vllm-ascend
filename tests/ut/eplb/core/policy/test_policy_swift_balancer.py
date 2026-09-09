# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend.eplb.core.policy.policy_swift_balancer import SwiftBalanceEplb


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(torch.npu, "device_count", lambda: 2)
    return SwiftBalanceEplb()


@pytest.mark.parametrize(
    "loads,should_search",
    [
        ([[100, 10], [10, 0]], True),  # Estimated PAR 1.0, observed PAR 1.833.
        ([[90, 10], [90, 10]], False),  # Estimated PAR 1.4, observed PAR 1.0.
        ([[0, 0], [0, 0]], False),
    ],
)
def test_rebalance_decision_uses_observed_replica_load(policy, monkeypatch, loads, should_search):
    placement = torch.tensor([[[0, 1], [2, 0]]])
    search = Mock(wraps=policy.redundant_expert_deployment)
    monkeypatch.setattr(policy, "redundant_expert_deployment", search)

    policy.rebalance_experts(placement, torch.tensor([loads]))

    assert search.call_count == int(should_search)


def test_candidate_uses_each_layers_observed_baseline(policy, monkeypatch):
    placement = torch.tensor([[[0, 1], [2, 0]], [[0, 1], [2, 0]]])
    loads = torch.tensor([[[100, 90], [10, 0]], [[1000, 900], [100, 0]]])
    candidate = np.array([[0, 2], [1, 0]])
    monkeypatch.setattr(policy, "redundant_expert_deployment", Mock(return_value=(None,) * 5))
    exchange = Mock(side_effect=[(candidate.copy(), 140), (candidate.copy(), 1400)])
    monkeypatch.setattr(policy, "exchange_experts", exchange)

    changed, _, updated = policy.rebalance_experts(placement, loads)

    # Both layers have observed PAR 1.9 and estimated candidate PAR 1.4.
    # Layer 1 must use its own mean (1000), not layer 0's mean (100).
    assert changed == 1
    np.testing.assert_array_equal(updated, np.stack([candidate, candidate]))
    assert policy.swap_threshold == pytest.approx(1000 * policy.increment)


@pytest.mark.parametrize("candidate_kind", ["unchanged", "local_permutation", "worse"])
def test_rejected_or_unchanged_candidate_does_not_claim_improvement(policy, monkeypatch, candidate_kind):
    placement = torch.tensor([[[0, 1], [2, 0]]])
    loads = torch.tensor([[[100, 90], [10, 0]]])
    candidate = placement[0].numpy().copy()
    estimate = 140
    if candidate_kind == "local_permutation":
        candidate = candidate[:, ::-1].copy()
    elif candidate_kind == "worse":
        candidate = np.array([[0, 2], [1, 0]])
        estimate = 200
    monkeypatch.setattr(policy, "redundant_expert_deployment", Mock(return_value=(None,) * 5))
    monkeypatch.setattr(policy, "exchange_experts", Mock(return_value=(candidate, estimate)))

    changed, _, updated = policy.rebalance_experts(placement, loads)

    assert changed == 0
    np.testing.assert_array_equal(updated, placement.numpy())
