# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np

from vllm_ascend.distributed.eplb.policy.stair_search import (
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
    current = np.array([[0, 1], [2, 3], [4, 5], [6, 1]], dtype=np.int64)
    risk_load = np.array([100, 1, 1, 1, 1, 1, 1], dtype=np.float64)

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

    np.testing.assert_array_equal(np.flatnonzero(np.any(contiguous == 0, axis=1)), [0, 1])
    np.testing.assert_array_equal(np.flatnonzero(np.any(noncontiguous == 0, axis=1)), [0, 2])


def test_rank_matching_never_crosses_equivalent_group():
    topology = StairTopology.from_rank_to_node((0, 1, 0, 1))
    current = np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int64)
    desired = np.array([[4, 5], [6, 7], [0, 1], [2, 3]], dtype=np.int64)

    matched = _match_equivalent_ranks(current, desired, topology)

    np.testing.assert_array_equal(matched, current)


def test_linear_assignment_deadline_returns_none():
    assert _solve_linear_assignment(np.zeros((4, 4)), deadline=0.0) is None
