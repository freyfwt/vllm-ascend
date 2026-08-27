# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest

from vllm_ascend.distributed.eplb.policy import stair_constants as constants


def test_validate_stair_constants_accepts_repository_defaults():
    constants.validate_stair_constants()


def test_validate_stair_constants_rejects_nonpositive_hard_budget(monkeypatch):
    monkeypatch.setattr(constants, "MAX_TRANSFERS_PER_RANK_PAIR", 0)

    with pytest.raises(ValueError, match="hard limits"):
        constants.validate_stair_constants()


def test_validate_stair_constants_rejects_score_weight_sum(monkeypatch):
    monkeypatch.setattr(constants, "MEAN_SCORE_WEIGHT", 0.8)
    monkeypatch.setattr(constants, "P95_SCORE_WEIGHT", 0.8)

    with pytest.raises(ValueError, match="sum to 1"):
        constants.validate_stair_constants()


def test_validate_stair_constants_rejects_unbounded_search(monkeypatch):
    monkeypatch.setattr(constants, "SEARCH_DEPTH", 100)

    with pytest.raises(ValueError, match="safe internal bound"):
        constants.validate_stair_constants()


def test_validate_stair_constants_requires_one_transfer_per_directed_pair(monkeypatch):
    monkeypatch.setattr(constants, "MAX_TRANSFERS_PER_RANK_PAIR", 2)

    with pytest.raises(ValueError, match="exactly one transfer"):
        constants.validate_stair_constants()
