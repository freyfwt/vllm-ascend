# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Per-rank transfer worker for explicit STAIR layer plans."""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
from vllm.distributed.eplb.eplb_utils import CpuGpuEvent
from vllm.distributed.eplb.rebalance_execute import AsyncEplbLayerResult
from vllm.logger import logger

from vllm_ascend.distributed.eplb.policy.stair_types import LayerPlan, RankTopology
from vllm_ascend.distributed.eplb.transfer_adapter import stage_layer


@dataclass(frozen=True)
class TransferWork:
    model_state: Any
    layer_plan: LayerPlan
    topology: RankTopology


def transfer_one_layer(
    work: TransferWork,
    *,
    ep_rank: int,
    stream: torch.cuda.Stream | None,
) -> AsyncEplbLayerResult:
    model_state, layer = work.model_state, work.layer_plan
    model_state.communicator.set_stream(stream)
    metadata = stage_layer(
        layer,
        model_state.model.expert_weights[layer.layer_idx],
        model_state.expert_buffer,
        model_state.communicator,
        work.topology,
        ep_rank=ep_rank,
        cuda_stream=stream,
    )
    if stream is not None:
        stream.synchronize()
    return AsyncEplbLayerResult(
        layer_idx=layer.layer_idx,
        new_physical_to_logical_map=torch.from_numpy(layer.new_placement.reshape(-1).copy()),
        transfer_metadata=metadata,
        consumed_event=CpuGpuEvent(),
    )


class StairTransferWorker:
    """Serialize layer staging and wait for main-thread consumption."""

    def __init__(self, device_index: int, ep_rank: int) -> None:
        self._device_index = device_index
        self._ep_rank = ep_rank
        self._queue: queue.Queue[TransferWork | None] = queue.Queue()
        self.failure: BaseException | None = None
        self._current: TransferWork | None = None
        self._thread = threading.Thread(target=self._run, name="stair-transfer", daemon=True)
        self._thread.start()

    def submit(self, work: TransferWork) -> None:
        if self.failure is not None:
            raise RuntimeError("STAIR transfer worker failed") from self.failure
        self._queue.put_nowait(work)

    def _run(self) -> None:
        try:
            torch.accelerator.set_device_index(self._device_index)
            stream = torch.cuda.Stream(device=self._device_index)
            while (work := self._queue.get()) is not None:
                self._current = work
                result = transfer_one_layer(work, ep_rank=self._ep_rank, stream=stream)
                work.model_state.pending_result = result
                result.consumed_event.wait(stream=stream)
                if work.model_state.pending_result is not None:
                    raise RuntimeError("STAIR result was acknowledged without being consumed")
                self._current = None
        except BaseException as error:  # pragma: no cover - hardware/runtime failure
            self.failure = error
            logger.exception("STAIR transfer worker failed: %s", error)

    def close(self) -> None:
        self._queue.put(None)
        while self._thread.is_alive():
            work = self._current
            result = None if work is None else work.model_state.pending_result
            if result is not None:
                work.model_state.pending_result = None
                result.consumed_event.record()
            self._thread.join(timeout=0.01)
            time.sleep(0.001)
