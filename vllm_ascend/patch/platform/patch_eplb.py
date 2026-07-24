# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Apply Ascend's load-scope policy at the upstream EPLB step boundary."""

from functools import wraps
from inspect import signature

from vllm.distributed.eplb import eplb_state as _eplb_state

_PATCH_MARKER = "_vllm_ascend_eplb_patch"


def _wrap_eplb_state_step(original_step):
    step_signature = signature(original_step)
    required_parameters = {"self", "is_dummy", "is_profile", "log_stats"}
    if not required_parameters.issubset(step_signature.parameters):
        raise RuntimeError("Unsupported vLLM EPLB contract: EplbState.step signature changed.")

    @wraps(original_step)
    def _step(self, *args, **kwargs):
        bound = step_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if (
            not getattr(self, "_ascend_scope_matched", True)
            and not bound.arguments["is_dummy"]
            and not bound.arguments["is_profile"]
        ):
            bound.arguments["is_dummy"] = True
            bound.arguments["log_stats"] = False
        return original_step(*bound.args, **bound.kwargs)

    setattr(_step, _PATCH_MARKER, True)
    return _step


def _patch_eplb_state() -> None:
    original_step = _eplb_state.EplbState.step
    if not getattr(original_step, _PATCH_MARKER, False):
        _eplb_state.EplbState.step = _wrap_eplb_state_step(original_step)


_patch_eplb_state()
