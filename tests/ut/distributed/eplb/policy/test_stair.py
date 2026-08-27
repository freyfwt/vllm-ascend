# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np
import pytest
import torch

from vllm_ascend.distributed.eplb.policy import stair_constants as constants
from vllm_ascend.distributed.eplb.policy.stair import (
    StairEplbPolicy,
    _build_incremental_candidate,
    compute_balance_score,
)
from vllm_ascend.distributed.eplb.policy.stair_types import (
    StairBalanceScore,
    StairRejectReason,
)


def _rebalance(
    policy: StairEplbPolicy,
    load_window: torch.Tensor,
    placement: torch.Tensor,
    num_ranks: int = 2,
) -> torch.Tensor:
    return policy.rebalance_experts(
        load_window,
        num_replicas=placement.shape[1],
        num_groups=1,
        num_nodes=1,
        num_ranks=num_ranks,
        old_global_expert_indices=placement,
    )


def test_stair_builds_incremental_candidate_without_legacy_policy():
    logical_load = np.array([[1, 1, 100, 1]], dtype=np.float64)
    current = np.array([[0, 1, 2, 3, 0, 1]], dtype=np.int64)

    candidate = _build_incremental_candidate(logical_load, current, num_ranks=2)

    assert candidate.shape == current.shape
    assert not np.array_equal(candidate, current)
    assert compute_balance_score(logical_load, candidate.reshape(2, 3)) < compute_balance_score(
        logical_load,
        current.reshape(2, 3),
    )
    for rank, old_rank in zip(candidate.reshape(2, 3), current.reshape(2, 3)):
        assert np.unique(rank).size == rank.size
        for slot_idx, expert_id in enumerate(rank):
            if expert_id in old_rank:
                assert expert_id == old_rank[slot_idx]


def test_stair_candidate_never_worsens_aggregate_balance():
    random = np.random.default_rng(7)
    num_ranks = 4
    slots_per_rank = 3
    current = np.tile(np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]), (64, 1))
    logical_load = random.integers(0, 1000, size=(64, 8)).astype(np.float64)

    candidate = _build_incremental_candidate(logical_load, current, num_ranks)

    for layer_idx in range(logical_load.shape[0]):
        current_score = compute_balance_score(
            logical_load[layer_idx : layer_idx + 1],
            current[layer_idx].reshape(num_ranks, slots_per_rank),
        )
        candidate_score = compute_balance_score(
            logical_load[layer_idx : layer_idx + 1],
            candidate[layer_idx].reshape(num_ranks, slots_per_rank),
        )
        assert candidate_score <= current_score


def test_stair_is_instance_owned_and_improves_balance():
    load_window = torch.tensor(
        [[[1, 1, 100, 1]], [[1, 1, 120, 1]], [[1, 1, 80, 1]], [[1, 1, 110, 1]]],
        dtype=torch.int32,
    )
    placement = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long)
    first = StairEplbPolicy()
    second = StairEplbPolicy()

    result = _rebalance(first, load_window, placement)

    assert first.average_to_peak_history is not second.average_to_peak_history
    old_score = compute_balance_score(load_window[:, 0].numpy(), placement.reshape(2, 3).numpy())
    new_score = compute_balance_score(load_window[:, 0].numpy(), result.reshape(2, 3).numpy())
    assert new_score < old_score


def test_stair_uses_full_time_series_for_temporal_acceptance(monkeypatch):
    load_window = torch.tensor(
        [[[100, 90, 1, 1], [100, 1, 90, 1]], [[90, 100, 1, 1], [90, 1, 100, 1]]],
        dtype=torch.int32,
    )
    placement = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    candidate = np.array([[0, 2, 1, 3], [0, 2, 1, 3]], dtype=np.int64)
    observed_load: list[np.ndarray] = []
    candidates = iter(candidate.reshape(2, 2, 2))

    def fake_candidate(logical_load, current_placement, topology):
        del current_placement, topology
        observed_load.append(logical_load.copy())
        return next(candidates).copy()

    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.policy.stair.build_greedy_layer_candidate",
        fake_candidate,
    )
    result = _rebalance(StairEplbPolicy(), load_window, placement)

    expected_risk_load = load_window.to(dtype=torch.float64).mean(dim=0).numpy()
    expected_risk_load += load_window.to(dtype=torch.float64).std(dim=0, unbiased=False).numpy()
    np.testing.assert_array_equal(np.stack(observed_load), expected_risk_load)
    torch.testing.assert_close(result[0], torch.from_numpy(candidate[0]))
    torch.testing.assert_close(result[1], placement[1])


def test_temporal_gate_runs_before_candidate_construction(monkeypatch):
    policy = StairEplbPolicy()
    topology = policy._resolve_topology(num_nodes=1, num_ranks=2)
    policy._shape_key = (1, 4, 4, 2, int(topology.topology_hash, 16), constants.STAIR_TUNING_VERSION)
    policy.average_to_peak_history[0] = 1.0

    def fail_candidate(*_args, **_kwargs):
        raise AssertionError("candidate construction must not run behind the temporal gate")

    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.policy.stair.build_greedy_layer_candidate",
        fail_candidate,
    )
    load = torch.tensor([[[100, 96, 1, 1]], [[96, 100, 1, 1]]], dtype=torch.float64)
    placement = torch.tensor([[0, 2, 1, 3]], dtype=torch.long)

    plan = policy.plan_rebalance(load, 4, 1, 1, 2, placement)

    assert plan.selected_layers == ()
    assert any(item.reject_reason is StairRejectReason.TEMPORAL_GATE for item in plan.rejected_layers)


def test_mini_flash_tree_disabled_skips_search(monkeypatch):
    policy = StairEplbPolicy()
    search_called = False
    monkeypatch.setattr(constants, "ENABLE_MINI_FLASH_TREE", False)

    def fail_search(*_args, **_kwargs):
        nonlocal search_called
        search_called = True
        raise AssertionError("search must stay disabled")

    monkeypatch.setattr(
        policy._mini_search,
        "search",
        fail_search,
    )

    _rebalance(
        policy,
        torch.tensor([[[1, 1, 100, 1]], [[1, 1, 120, 1]]], dtype=torch.float64),
        torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long),
    )

    assert not search_called


def test_production_shape_candidate_work_is_bounded(monkeypatch):
    random = np.random.default_rng(17)
    load = torch.from_numpy(random.lognormal(mean=2.0, sigma=1.5, size=(2, 48, 128)))
    placement = torch.from_numpy(np.tile(np.arange(132, dtype=np.int64) % 128, (48, 1)))
    build_count = 0
    from vllm_ascend.distributed.eplb.policy import stair as stair_module

    real_build = stair_module.build_greedy_layer_candidate

    def counted_build(*args, **kwargs):
        nonlocal build_count
        build_count += 1
        return real_build(*args, **kwargs)

    monkeypatch.setattr(stair_module, "build_greedy_layer_candidate", counted_build)
    plan = StairEplbPolicy(runtime_history=False).plan_rebalance(load, 132, 1, 1, 4, placement)

    assert build_count == placement.shape[0]
    assert all(
        layer.transfer_cost.max_rank_pair_transfers <= constants.MAX_TRANSFERS_PER_RANK_PAIR
        for layer in plan.selected_layers
    )
    assert not any(layer.reject_reason is StairRejectReason.INVALID_TOPOLOGY for layer in plan.rejected_layers)


def test_stair_keeps_zero_load_and_balanced_placement():
    placement = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    zero_result = _rebalance(
        StairEplbPolicy(),
        torch.zeros((4, 1, 4), dtype=torch.float64),
        placement,
    )
    balanced_result = _rebalance(
        StairEplbPolicy(),
        torch.ones((4, 1, 4), dtype=torch.float64),
        placement,
    )

    torch.testing.assert_close(zero_result, placement)
    torch.testing.assert_close(balanced_result, placement)


def test_plan_rebalance_orders_and_selects_all_eligible_layers():
    load_window = torch.tensor(
        [
            [[1, 1, 100, 1], [1, 1, 80, 1]],
            [[1, 1, 120, 1], [1, 1, 90, 1]],
        ],
        dtype=torch.int32,
    )
    placement = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1],
            [0, 1, 2, 3, 0, 1],
        ],
        dtype=torch.long,
    )
    plan = StairEplbPolicy().plan_rebalance(
        load_window,
        num_replicas=6,
        num_groups=1,
        num_nodes=1,
        num_ranks=2,
        old_global_expert_indices=placement,
    )

    assert len(plan.selected_layers) == 2
    assert plan.budget_usage.selected_layers == 2
    assert tuple(plan.selected_layers) == tuple(
        sorted(plan.selected_layers, key=lambda item: (-item.utility, item.layer_idx))
    )
    assert all(
        layer.transfer_cost.max_rank_pair_transfers <= constants.MAX_TRANSFERS_PER_RANK_PAIR
        for layer in plan.selected_layers
    )


def test_all_eligible_layers_converge_in_one_cycle_under_pair_limit():
    num_layers = 8
    load_window = torch.tensor([[[1, 1, 100, 1]] * num_layers] * 2, dtype=torch.int32)
    placement = torch.tensor([[0, 1, 2, 3, 0, 1]] * num_layers, dtype=torch.long)
    policy = StairEplbPolicy(runtime_history=False)

    first = policy.plan_rebalance(load_window, 6, 1, 1, 2, placement)
    policy.finish_cycle(first.plan_id, ())
    second = policy.plan_rebalance(load_window, 6, 1, 1, 2, first.new_mapping)

    assert len(first.selected_layers) == num_layers
    assert all(layer.transfer_cost.max_rank_pair_transfers == 1 for layer in first.selected_layers)
    assert second.selected_layers == ()


def test_plan_digest_changes_with_tuning_version(monkeypatch):
    load_window = torch.tensor([[[1, 1, 100, 1]]], dtype=torch.int32)
    placement = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long)
    first = StairEplbPolicy().plan_rebalance(load_window, 6, 1, 1, 2, placement)
    monkeypatch.setattr(constants, "STAIR_TUNING_VERSION", constants.STAIR_TUNING_VERSION + 1)
    second = StairEplbPolicy().plan_rebalance(load_window, 6, 1, 1, 2, placement)

    assert first.plan_id != second.plan_id


def test_stair_uses_temporal_hysteresis_and_absolute_thresholds():
    policy = StairEplbPolicy()
    policy.average_to_peak_history[0] = 1.0

    assert not policy._needs_temporal_update(0, current_score=1 / 0.96, num_ranks=2)
    assert policy._needs_temporal_update(0, current_score=1 / 0.94, num_ranks=2)

    policy.average_to_peak_history[0] = 0.92
    assert policy._needs_temporal_update(0, current_score=1 / 0.89, num_ranks=2)


def test_stair_rejects_mean_gain_that_regresses_p95(monkeypatch):
    policy = StairEplbPolicy()
    current_score = StairBalanceScore(1.5, 1.6, 1.7)
    candidate_score = StairBalanceScore(1.1, 1.7, 1.8)
    scores = iter((current_score, candidate_score))
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.policy.stair.compute_balance_metrics",
        lambda *_args: next(scores),
    )

    layer_plan = policy._build_layer_plan(
        layer_idx=0,
        layer_load=np.ones((2, 4)),
        current=np.array([[0, 1], [2, 3]]),
        candidate=np.array([[0, 2], [1, 3]]),
        num_ranks=2,
    )

    assert not layer_plan.accepted
    assert layer_plan.reject_reason is StairRejectReason.TAIL_REGRESSION


def test_stair_commits_history_only_after_real_layer_commit():
    policy = StairEplbPolicy()
    load_window = torch.tensor([[[100, 100, 1, 1]], [[1, 1, 100, 100]]], dtype=torch.int32)
    placement = torch.tensor([0, 2, 1, 3], dtype=torch.long)

    policy.commit_layer(load_window, 0, placement, num_ranks=2)

    assert set(policy.average_to_peak_history) == {0}
    expected_score = compute_balance_score(load_window[:, 0].numpy(), placement.reshape(2, 2).numpy())
    assert policy.average_to_peak_history[0] == pytest.approx(1 / expected_score)


def test_stair_clears_history_when_committed_placement_is_not_observed():
    policy = StairEplbPolicy()
    load_window = torch.ones((2, 1, 4), dtype=torch.float64)
    current = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    committed = torch.tensor([0, 2, 1, 3], dtype=torch.long)
    policy.commit_layer(load_window, 0, committed, num_ranks=2)

    _rebalance(policy, load_window, current)

    assert policy.average_to_peak_history == {}


@pytest.mark.parametrize(
    ("load", "placement", "error"),
    [
        (
            torch.ones((2, 1, 4)),
            None,
            "requires the current",
        ),
        (
            torch.ones((2, 1, 4)),
            torch.tensor([[0, 1, 2, 4]]),
            "invalid logical expert",
        ),
        (
            torch.ones((2, 1, 4)),
            torch.tensor([[0, 0, 1, 2]]),
            "at least one physical replica",
        ),
        (
            torch.tensor([[[1.0, -1.0, 1.0, 1.0]]]),
            torch.tensor([[0, 1, 2, 3]]),
            "finite, non-negative",
        ),
    ],
)
def test_stair_rejects_invalid_inputs(
    load: torch.Tensor,
    placement: torch.Tensor | None,
    error: str,
):
    with pytest.raises(ValueError, match=error):
        StairEplbPolicy().rebalance_experts(
            load,
            num_replicas=4,
            num_groups=1,
            num_nodes=1,
            num_ranks=2,
            old_global_expert_indices=placement,
        )
