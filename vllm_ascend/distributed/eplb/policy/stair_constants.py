# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Internal tuning constants for STAIR EPLB.

These values are intentionally not part of the user configuration surface.
Developers should update them together with focused tests and benchmarks.
"""

STAIR_TUNING_VERSION = 3

# Cycle budgets.
MAX_LAYERS_PER_CYCLE = 8
MAX_EXPERT_TRANSFERS_PER_CYCLE = 128
MAX_TRANSFER_BYTES_PER_CYCLE = 64 * 1024**3
MAX_CROSS_NODE_BYTES_PER_CYCLE = 32 * 1024**3

# Admission and scoring.
BALANCE_EPSILON = 1e-6
IMBALANCE_THRESHOLD = 1.01
MIN_BALANCE_GAIN = BALANCE_EPSILON
MIN_GAIN_PER_GIB = 0.0
TAIL_REGRESSION_TOLERANCE = 0.0
MEAN_SCORE_WEIGHT = 0.5
P95_SCORE_WEIGHT = 0.5
SWAP_IMPROVEMENT_RATIO = 0.01

# Temporal risk statistics.
EWMA_ALPHA = 0.25
RISK_Z = 1.0
ENABLE_SPARSE_COVARIANCE = False
COVARIANCE_TOP_K = 8
TEMPORAL_UPDATE_THRESHOLD_RATIO = 0.9
TEMPORAL_UPDATE_THRESHOLD_VALUE = 0.85
SMALL_WORLD_SIZE = 32
SMALL_WORLD_UPDATE_THRESHOLD_RATIO = 0.95
SMALL_WORLD_UPDATE_THRESHOLD_VALUE = 0.9

# Topology and transfer costs.
INTRA_NODE_COST_MULTIPLIER = 1.0
CROSS_NODE_COST_MULTIPLIER = 4.0
MAX_TRANSFERS_PER_RANK_PAIR = 16
MAX_SWAP_ATTEMPTS = 100

# Bounded MiniFlashTree search. Keep disabled until benchmark evidence is
# available for the repository defaults.
ENABLE_MINI_FLASH_TREE = False
MAX_SEARCH_LAYERS = 4
SEARCH_DEPTH = 2
SEARCH_WIDTH = 4
MAX_CANDIDATES_PER_LAYER = 16
MAX_SEARCH_MS_PER_LAYER = 10.0
MAX_PLANNER_MS = 100.0
SEARCH_STAGNATION_CYCLES = 2

_MAX_SAFE_SEARCH_DEPTH = 4
_MAX_SAFE_SEARCH_WIDTH = 16
_MAX_SAFE_CANDIDATES_PER_LAYER = 64
_MAX_SAFE_PLANNER_MS = 1000.0


def validate_stair_constants() -> None:
    """Reject contradictory or unbounded repository tuning constants."""
    positive_limits = {
        "MAX_LAYERS_PER_CYCLE": MAX_LAYERS_PER_CYCLE,
        "MAX_EXPERT_TRANSFERS_PER_CYCLE": MAX_EXPERT_TRANSFERS_PER_CYCLE,
        "MAX_TRANSFER_BYTES_PER_CYCLE": MAX_TRANSFER_BYTES_PER_CYCLE,
        "MAX_CROSS_NODE_BYTES_PER_CYCLE": MAX_CROSS_NODE_BYTES_PER_CYCLE,
        "MAX_TRANSFERS_PER_RANK_PAIR": MAX_TRANSFERS_PER_RANK_PAIR,
        "MAX_SWAP_ATTEMPTS": MAX_SWAP_ATTEMPTS,
        "MAX_SEARCH_LAYERS": MAX_SEARCH_LAYERS,
        "SEARCH_DEPTH": SEARCH_DEPTH,
        "SEARCH_WIDTH": SEARCH_WIDTH,
        "MAX_CANDIDATES_PER_LAYER": MAX_CANDIDATES_PER_LAYER,
        "MAX_SEARCH_MS_PER_LAYER": MAX_SEARCH_MS_PER_LAYER,
        "MAX_PLANNER_MS": MAX_PLANNER_MS,
        "SEARCH_STAGNATION_CYCLES": SEARCH_STAGNATION_CYCLES,
    }
    invalid = [name for name, value in positive_limits.items() if value <= 0]
    if invalid:
        raise ValueError(f"STAIR hard limits must be positive: {', '.join(invalid)}")
    if abs(MEAN_SCORE_WEIGHT + P95_SCORE_WEIGHT - 1.0) > BALANCE_EPSILON:
        raise ValueError("STAIR mean and p95 score weights must sum to 1.")
    if MEAN_SCORE_WEIGHT < 0 or P95_SCORE_WEIGHT < 0:
        raise ValueError("STAIR score weights must be non-negative.")
    if not 0 < EWMA_ALPHA <= 1:
        raise ValueError("STAIR EWMA_ALPHA must be in (0, 1].")
    if RISK_Z < 0:
        raise ValueError("STAIR RISK_Z must be non-negative.")
    if CROSS_NODE_COST_MULTIPLIER < INTRA_NODE_COST_MULTIPLIER or INTRA_NODE_COST_MULTIPLIER < 1:
        raise ValueError("STAIR topology cost multipliers must satisfy cross-node >= intra-node >= 1.")
    if MAX_TRANSFERS_PER_RANK_PAIR > MAX_EXPERT_TRANSFERS_PER_CYCLE:
        raise ValueError("STAIR rank-pair transfer limit cannot exceed the cycle transfer budget.")
    if SEARCH_DEPTH > _MAX_SAFE_SEARCH_DEPTH:
        raise ValueError("STAIR SEARCH_DEPTH exceeds the safe internal bound.")
    if SEARCH_WIDTH > _MAX_SAFE_SEARCH_WIDTH:
        raise ValueError("STAIR SEARCH_WIDTH exceeds the safe internal bound.")
    if MAX_CANDIDATES_PER_LAYER < SEARCH_WIDTH or MAX_CANDIDATES_PER_LAYER > _MAX_SAFE_CANDIDATES_PER_LAYER:
        raise ValueError("STAIR candidate count must cover search width and remain bounded.")
    if MAX_PLANNER_MS > _MAX_SAFE_PLANNER_MS or MAX_SEARCH_MS_PER_LAYER > MAX_PLANNER_MS:
        raise ValueError("STAIR planner deadlines are inconsistent or unbounded.")
    if ENABLE_SPARSE_COVARIANCE and COVARIANCE_TOP_K <= 0:
        raise ValueError("STAIR COVARIANCE_TOP_K must be positive when covariance is enabled.")
