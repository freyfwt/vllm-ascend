# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.patch.platform import patch_eplb


def test_non_matching_scope_discards_pass_without_advancing_load_window(monkeypatch):
    model_state = SimpleNamespace(expert_load_pass=torch.ones(2, dtype=torch.int64))
    eplb_state = patch_eplb._eplb_state.EplbState.__new__(patch_eplb._eplb_state.EplbState)
    eplb_state.model_states = {"model": model_state}
    eplb_state.parallel_config = SimpleNamespace(
        eplb_config=SimpleNamespace(
            log_balancedness_interval=1,
        )
    )
    eplb_state.expert_rearrangement_step = 0
    eplb_state.expert_rearrangement_step_interval = 10
    eplb_state.expert_load_window_step = 0
    eplb_state.expert_load_window_size = 2
    eplb_state.should_record_tensor = None
    eplb_state.is_async = False
    eplb_state._ascend_scope_matched = False
    ep_group = SimpleNamespace(device_group=MagicMock())
    monkeypatch.setattr(patch_eplb._eplb_state, "get_ep_group", lambda: ep_group)

    eplb_state.step()

    torch.testing.assert_close(model_state.expert_load_pass, torch.zeros(2, dtype=torch.int64))
    assert eplb_state.expert_load_window_step == 0
    assert eplb_state.expert_rearrangement_step == 1
