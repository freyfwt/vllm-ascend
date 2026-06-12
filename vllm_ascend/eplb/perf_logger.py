#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

from vllm.logger import logger

from vllm_ascend.ascend_config import EplbPerfLogMode


class EplbPerfLogger:
    def __init__(self):
        self.enabled = False
        self.rank = -1
        self.cycle_round = 0

    def configure(self, log_mode: int | EplbPerfLogMode, rank: int):
        self.rank = rank
        self.cycle_round = 0
        log_mode = EplbPerfLogMode(log_mode)
        self.enabled = log_mode == EplbPerfLogMode.ALL_RANKS or (
            log_mode == EplbPerfLogMode.RANK0 and rank == 0
        )

    def start(self):
        if not self.enabled:
            return 0
        return time.perf_counter_ns()

    def log(self, event: str, start_ns: int):
        if not self.enabled:
            return
        duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        logger.info(
            "[EPLB_PERF] event=%s rank=%s cycle_round=%s duration_ms=%.3f",
            event,
            self.rank,
            self.cycle_round,
            duration_ms,
        )

    def perf_event(self, event: str, start_event, end_event):
        return event, self.cycle_round, start_event, end_event

    def log_npu_event(self, perf_event):
        event, cycle_round, start_event, end_event = perf_event
        if not self.enabled or not end_event.query():
            return
        duration_ms = start_event.elapsed_time(end_event)
        logger.info(
            "[EPLB_PERF] event=%s rank=%s cycle_round=%s duration_ms=%.3f",
            event,
            self.rank,
            cycle_round,
            duration_ms,
        )


eplb_perf_logger = EplbPerfLogger()
