# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np

from vllm_ascend.distributed.eplb.policy import stair_constants as constants
from vllm_ascend.distributed.eplb.policy.stair_search import (
    MiniFlashTreeSearch,
    SearchCandidate,
    _match_equivalent_ranks,
    _solve_linear_assignment,
    build_greedy_layer_candidate,
    validate_layer_placement,
)
from vllm_ascend.distributed.eplb.policy.stair_types import StairTopology


def test_topology_normalizes_noncontiguous_node_ids():
    topology = StairTopology.from_rank_to_node((9, 4, 9, 4))

    assert topology.rank_to_node == (0, 1, 0, 1)
    assert topology.equivalent_rank_groups == ((0, 2), (1, 3))
    assert topology.same_node(0, 2)
    assert not topology.same_node(0, 1)


def test_greedy_candidate_preserves_all_placement_invariants():
    current = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int64)
    risk_load = np.array([1, 1, 100, 1], dtype=np.float64)

    candidate = build_greedy_layer_candidate(
        risk_load,
        current,
        StairTopology.contiguous(num_ranks=2, num_nodes=1),
    )

    validate_layer_placement(candidate, num_experts=4, num_ranks=2)
    assert not np.array_equal(candidate, current)


def test_hierarchical_placement_prefers_same_node_capacity():
    current = np.array([[0, 4], [1, 2], [2, 3], [3, 5]], dtype=np.int64)
    risk_load = np.array([100, 90, 1, 1, 1, 1], dtype=np.float64)

    contiguous = build_greedy_layer_candidate(
        risk_load,
        current,
        StairTopology.from_rank_to_node((0, 0, 1, 1)),
    )
    noncontiguous = build_greedy_layer_candidate(
        risk_load,
        current,
        StairTopology.from_rank_to_node((0, 1, 0, 1)),
    )

    contiguous_ranks = set(np.flatnonzero(np.any(contiguous == 0, axis=1)))
    noncontiguous_ranks = set(np.flatnonzero(np.any(noncontiguous == 0, axis=1)))
    assert 1 in contiguous_ranks and 2 not in contiguous_ranks
    assert 2 in noncontiguous_ranks and 1 not in noncontiguous_ranks


def test_rank_matching_never_crosses_equivalent_group():
    topology = StairTopology.from_rank_to_node((0, 1, 0, 1))
    current = np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int64)
    desired = np.array([[4, 5], [6, 7], [0, 1], [2, 3]], dtype=np.int64)

    matched = _match_equivalent_ranks(current, desired, topology)

    np.testing.assert_array_equal(matched, current)


def test_linear_assignment_solves_deterministically():
    np.testing.assert_array_equal(_solve_linear_assignment(np.zeros((4, 4))), np.arange(4))


def test_mini_flash_tree_is_bounded_and_deterministic(monkeypatch):
    monkeypatch.setattr(constants, "SEARCH_DEPTH", 2)
    monkeypatch.setattr(constants, "SEARCH_WIDTH", 2)
    monkeypatch.setattr(constants, "MAX_CANDIDATES_PER_LAYER", 4)
    counts = np.array([2, 1, 1], dtype=np.int64)
    root = SearchCandidate(
        placement=np.zeros((2, 2), dtype=np.int64),
        replica_counts=counts,
        objective=0.0,
        path_key=(2, 1, 1),
    )

    def evaluate(candidate_counts):
        return SearchCandidate(
            placement=candidate_counts[None, :],
            replica_counts=candidate_counts.copy(),
            objective=float(candidate_counts[1] * 10 + candidate_counts[2]),
            path_key=tuple(int(value) for value in candidate_counts),
        )

    first = MiniFlashTreeSearch().search(
        root,
        np.array([100, 50, 10], dtype=np.float64),
        num_ranks=2,
        evaluator=evaluate,
    )
    second = MiniFlashTreeSearch().search(
        root,
        np.array([100, 50, 10], dtype=np.float64),
        num_ranks=2,
        evaluator=evaluate,
    )

    assert first.candidate.path_key == second.candidate.path_key == (1, 2, 1)
    assert first.evaluated_candidates <= constants.MAX_CANDIDATES_PER_LAYER


def test_mini_flash_tree_returns_root_when_no_candidate_is_valid():
    root = SearchCandidate(
        placement=np.zeros((2, 2), dtype=np.int64),
        replica_counts=np.array([2, 1, 1], dtype=np.int64),
        objective=1.0,
        path_key=(2, 1, 1),
    )

    result = MiniFlashTreeSearch().search(
        root,
        np.ones(3),
        num_ranks=2,
        evaluator=lambda *_args: None,
    )

    assert result.candidate is root
    assert result.evaluated_candidates <= constants.MAX_CANDIDATES_PER_LAYER
