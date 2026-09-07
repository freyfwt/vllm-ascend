from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.policy.stair_candidate import config_digest, passes_hysteresis


def test_hysteresis_only_opens_after_committed_anchor_degrades():
    config = StairConfig(hysteresis_relative=0.9, hysteresis_absolute=0.8)
    assert passes_hysteresis(1.1, None, config)
    assert not passes_hysteresis(1.1, 1.05, config)
    assert passes_hysteresis(1.3, 1.05, config)


def test_config_digest_is_stable_and_sensitive_to_tuning():
    assert config_digest(StairConfig()) == config_digest(StairConfig())
    assert config_digest(StairConfig()) != config_digest(StairConfig(sample_size=8))
