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

    def configure(self, log_mode: int | EplbPerfLogMode, rank: int):
        self.rank = rank
        log_mode = EplbPerfLogMode(log_mode)
        self.enabled = log_mode == EplbPerfLogMode.ALL_RANKS or (
            log_mode == EplbPerfLogMode.RANK0 and rank == 0
        )

    def log(self, event: str, round_id: int, start_ns: int):
        if not self.enabled:
            return
        duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        logger.info(
            "[EPLB_PERF] event=%s rank=%s round=%s duration_ms=%.3f", event, self.rank, round_id, duration_ms
        )


eplb_perf_logger = EplbPerfLogger()
