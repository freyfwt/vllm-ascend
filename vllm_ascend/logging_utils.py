#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import logging
from datetime import datetime

MICROSECOND_LOG_FORMAT = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
MICROSECOND_DATE_FORMAT = "%m-%d %H:%M:%S.%f"


class MicrosecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        record_datetime = datetime.fromtimestamp(record.created)
        if datefmt:
            formatted_time = record_datetime.strftime(datefmt)
            if "%f" in datefmt:
                microsecond = f"{record_datetime.microsecond:06d}"
                formatted_time = formatted_time.replace(microsecond, f"{microsecond[:3]}.{microsecond[3:]}", 1)
            return formatted_time
        return record_datetime.isoformat(timespec="microseconds")


def configure_vllm_microsecond_logging():
    configured_handlers = []
    target_logger = logging.getLogger("vllm.logger")
    while target_logger:
        configured_handlers.extend(target_logger.handlers)
        if not target_logger.propagate:
            break
        target_logger = target_logger.parent

    if not configured_handlers:
        configured_handlers.extend(logging.getLogger("vllm").handlers)

    for handler in set(configured_handlers):
        if not isinstance(handler.formatter, MicrosecondFormatter):
            handler.setFormatter(
                MicrosecondFormatter(
                    fmt=MICROSECOND_LOG_FORMAT,
                    datefmt=MICROSECOND_DATE_FORMAT,
                )
            )
