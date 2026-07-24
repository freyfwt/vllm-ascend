# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Apply Ascend's load-scope policy at the upstream EPLB step boundary."""

from functools import wraps
from inspect import signature

from vllm.distributed.eplb import eplb_state as _eplb_state

_PATCH_MARKER = "_vllm_ascend_eplb_patch"


def _patch_eplb_state() -> None:
    original_step = _eplb_state.EplbState.step
    if getattr(original_step, _PATCH_MARKER, False):
        return
    required_parameters = {"self", "is_dummy", "is_profile", "log_stats"}
    if not required_parameters.issubset(signature(original_step).parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: EplbState.step signature changed.")

    @wraps(original_step)
    def _step(self, is_dummy=False, is_profile=False, log_stats=False):
        if not is_dummy and not is_profile and not getattr(self, "_ascend_scope_matched", True):
            is_dummy = True
            log_stats = False
        return original_step(self, is_dummy=is_dummy, is_profile=is_profile, log_stats=log_stats)

    setattr(_step, _PATCH_MARKER, True)
    _eplb_state.EplbState.step = _step


_patch_eplb_state()
