# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Process creation details kept outside the STAIR planner client."""

import os
import socket
import subprocess
import sys
from multiprocessing.connection import Connection

from vllm.logger import logger

from vllm_ascend.ascend_config import StairConfig


def spawn_planner(shared_fds: tuple[int, ...]) -> tuple[subprocess.Popen, Connection]:
    parent_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    inherited = (child_socket.fileno(), *shared_fds)
    environment = os.environ.copy()
    for variable in (
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(variable, None)
    environment.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": "",
            "VLLM_PLUGINS": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm_ascend.distributed.eplb.planner_process",
                "--control-fd",
                str(child_socket.fileno()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=inherited,
            close_fds=True,
            start_new_session=True,
            env=environment,
        )
    except BaseException:
        parent_socket.close()
        child_socket.close()
        raise
    child_socket.close()
    return process, Connection(parent_socket.detach())


def apply_affinity(process: subprocess.Popen, config: StairConfig) -> None:
    if config.planner_cpu_set == "auto":
        return
    try:
        os.sched_setaffinity(process.pid, set(config.planner_cpu_set))  # type: ignore[attr-defined,arg-type]
    except (AttributeError, OSError) as error:
        if config.planner_affinity_strict:
            raise RuntimeError(f"STAIR planner CPU affinity failed: {error}") from error
        logger.warning("STAIR planner CPU affinity was not applied: %s", error)
