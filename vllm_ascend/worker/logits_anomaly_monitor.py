# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import logger
from vllm.utils.torch_utils import PIN_MEMORY

_ANOMALY_NAMES = ("NaN", "+Inf", "-Inf")
_BUFFER_SIZE = 2
_REPORT_DELAY = 2


@dataclass
class _CheckSlot:
    host_flags: torch.Tensor
    copy_done: Any
    device_flags: torch.Tensor | None = None
    iteration: int = -1
    sampling_stage: str = ""

    @property
    def pending(self) -> bool:
        return self.iteration >= 0

    def reset(self) -> None:
        self.device_flags = None
        self.iteration = -1
        self.sampling_stage = ""


class LogitsAnomalyMonitor:
    """Report non-finite model logits without synchronizing the model thread."""

    def __init__(self) -> None:
        self._iteration = 0
        self._copy_stream: Any | None = None
        self._device: torch.device | None = None
        self._slots: list[_CheckSlot] = []

    def check(self, logits: torch.Tensor, sampling_stage: str) -> None:
        """Enqueue a device-side check and report results at least two steps old."""
        self._ensure_initialized(logits.device)
        self._report_completed(self._iteration - _REPORT_DELAY)

        if logits.numel() > 0:
            slot = self._get_free_slot()
            flags = torch.stack(
                (
                    logits.isnan().any(),
                    logits.eq(float("inf")).any(),
                    logits.eq(float("-inf")).any(),
                )
            )
            slot.device_flags = flags
            slot.iteration = self._iteration
            slot.sampling_stage = sampling_stage

            main_stream = torch.npu.current_stream()
            with torch.npu.stream(self._copy_stream):
                self._copy_stream.wait_stream(main_stream)
                slot.host_flags.copy_(flags, non_blocking=True)
                slot.copy_done.record(self._copy_stream)

        self._iteration += 1

    def flush(self) -> None:
        """Report pending results during shutdown, when blocking is harmless."""
        for slot in sorted(
            (slot for slot in self._slots if slot.pending),
            key=lambda slot: slot.iteration,
        ):
            slot.copy_done.synchronize()
            self._report_slot(slot)

    def _ensure_initialized(self, device: torch.device) -> None:
        if self._copy_stream is not None:
            if device != self._device:
                raise RuntimeError(f"Logits anomaly monitor cannot move from {self._device} to {device}.")
            return

        self._device = device
        self._copy_stream = torch.npu.Stream()
        self._slots = [self._new_slot() for _ in range(_BUFFER_SIZE)]

    @staticmethod
    def _new_slot() -> _CheckSlot:
        return _CheckSlot(
            host_flags=torch.empty(
                len(_ANOMALY_NAMES),
                dtype=torch.bool,
                device="cpu",
                pin_memory=PIN_MEMORY,
            ),
            copy_done=torch.npu.Event(),
        )

    def _get_free_slot(self) -> _CheckSlot:
        for slot in self._slots:
            if not slot.pending:
                return slot

        # Two slots are sufficient in steady state. If the NPU is more than two
        # iterations behind, retain the pending copies rather than synchronizing
        # the model thread or overwriting a host buffer still used by DMA.
        slot = self._new_slot()
        self._slots.append(slot)
        return slot

    def _report_completed(self, max_iteration: int) -> None:
        for slot in sorted(
            (slot for slot in self._slots if slot.pending and slot.iteration <= max_iteration),
            key=lambda slot: slot.iteration,
        ):
            if slot.copy_done.query():
                self._report_slot(slot)

    def _report_slot(self, slot: _CheckSlot) -> None:
        anomalies = [name for name, present in zip(_ANOMALY_NAMES, slot.host_flags.tolist()) if present]
        if anomalies:
            logger.warning(
                "Detected %s in model logits before %s (reported %d iterations later).",
                ", ".join(anomalies),
                slot.sampling_stage,
                self._iteration - slot.iteration,
            )
        slot.reset()
