#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
from enum import Enum

import torch
import torch.distributed as dist
from vllm.logger import logger
from vllm.v1.utils import record_function_or_nullcontext

from vllm_ascend.distributed.parallel_state import get_dynamic_eplb_group
from vllm_ascend.eplb.perf_logger import eplb_perf_logger


class ExpertWeightUpdateState(Enum):
    WAITING = 0  # waiting for updated expert_map by EplbWorker
    READY = 1  # ready for d2d expert weights updating
    TRANSFERRING = 2  # d2d finished and waiting for updating expert_map into model


class D2DExpertWeightLoader:
    def __init__(self):
        self.comm_op_list = None
        self.updated_expert_map = None
        self.updated_log2phy_map = None
        self.layer_id = -1  # layer id to be updated
        self.state = ExpertWeightUpdateState.WAITING
        self.recv_expert_list = []
        self.num_layers = 0
        self.comm_group = get_dynamic_eplb_group()
        self.eplb_cycle_round = 0
        self.d2d_start_event = None
        self.d2d_end_event = None
        self.pending_update_events = []

    def set_adator(self, eplb_adaptor):
        self.eplb_adaptor = eplb_adaptor

    def generate_expert_d2d_transfer_task(self, expert_send_info, expert_recv_info, updated_expert_map, layer_id):
        # When current send/recv and weight.expert_map update tasks are not finished, cannot accept new d2d task
        if self.state != ExpertWeightUpdateState.WAITING:
            logger.warning_once(
                "[eplb/d2d_loader] Current D2D weight update is on-going, cannot accept new update task"
            )
            return

        self.updated_expert_map = updated_expert_map

        self.layer_id = layer_id
        self.comm_op_list = []
        for send_info in expert_send_info:
            dst_rank, global_expert_id_to_send = send_info
            local_expert_id = self.eplb_adaptor.expert_map_per_layer_cpu[layer_id][global_expert_id_to_send].item()
            for src_tensor in self.eplb_adaptor.expert_param_per_layer[layer_id][local_expert_id]:
                self.comm_op_list.append(
                    dist.P2POp(
                        dist.isend, src_tensor, self.comm_group.ranks[dst_rank], group=self.comm_group.device_group
                    )
                )

        for buffer_tensor_id, recv_info in enumerate(expert_recv_info):
            recv_rank, global_expert_id_to_recv = recv_info
            expert_weight_key = self.eplb_adaptor.expert_weight_key_per_layer[layer_id]
            for buffer_tensor in self.eplb_adaptor.buffer_tensor_list[expert_weight_key][buffer_tensor_id]:
                self.comm_op_list.append(
                    dist.P2POp(
                        dist.irecv, buffer_tensor, self.comm_group.ranks[recv_rank], group=self.comm_group.device_group
                    )
                )
            local_expert_to_replace = self.updated_expert_map[global_expert_id_to_recv].item()
            self.recv_expert_list.append((local_expert_to_replace, buffer_tensor_id))

        self.state = ExpertWeightUpdateState.READY

    def set_log2phy_map(self, log2phy_map):
        self.updated_log2phy_map = log2phy_map

    def asyn_expert_weight_transfer(self, reqs):
        # Only when send/recv tasks are parsed into self.comm_op_list, d2d send/recv tasks can be launched
        if self.state != ExpertWeightUpdateState.READY:
            return

        # set asynchronous stream for d2d expert weight transfer
        if self.comm_op_list:
            if eplb_perf_logger.enabled:
                self.d2d_start_event = torch.npu.Event(enable_timing=True)
                self.d2d_end_event = torch.npu.Event(enable_timing=True)
                self.d2d_start_event.record()
            ret_list = dist.batch_isend_irecv(self.comm_op_list)
            reqs.extend(ret_list)
            if eplb_perf_logger.enabled:
                self.d2d_end_event.record()

        self.state = ExpertWeightUpdateState.TRANSFERRING

    def update_expert_map_and_weight(self, reqs):
        # Only after send/recv tasks have been launched, expert_map and weight can be updated
        if self.state != ExpertWeightUpdateState.TRANSFERRING:
            return

        # Waiting for send/recv tasks finish
        if reqs:
            with record_function_or_nullcontext("EPLB weight D2D wait"):
                for req in reqs:
                    req.wait()
                if self.d2d_start_event is not None and self.d2d_end_event is not None:
                    eplb_perf_logger.log_npu_event(
                        "d2d_transfer_execute", self.eplb_cycle_round, self.d2d_start_event, self.d2d_end_event
                    )
                    self.d2d_start_event = None
                    self.d2d_end_event = None

        if self.pending_update_events:
            pending_update_events = []
            for event, cycle_round, start_event, end_event in self.pending_update_events:
                if end_event.query():
                    eplb_perf_logger.log_npu_event(event, cycle_round, start_event, end_event)
                else:
                    pending_update_events.append((event, cycle_round, start_event, end_event))
            self.pending_update_events = pending_update_events

        if self.comm_op_list is not None:
            self.comm_op_list = None

        # update expert_map
        start_ns = eplb_perf_logger.start()
        self.eplb_adaptor.do_update_expert_map(self.layer_id, self.updated_expert_map)
        eplb_perf_logger.log("expert_map_update", self.eplb_cycle_round, start_ns)

        # update log2phy_map
        if eplb_perf_logger.enabled:
            log2phy_start_event = torch.npu.Event(enable_timing=True)
            log2phy_end_event = torch.npu.Event(enable_timing=True)
            log2phy_start_event.record()
        self.eplb_adaptor.do_update_log2phy_map(self.layer_id, self.updated_log2phy_map)
        if eplb_perf_logger.enabled:
            log2phy_end_event.record()
            self.pending_update_events.append(
                ("log2phy_update_execute", self.eplb_cycle_round, log2phy_start_event, log2phy_end_event)
            )

        # update expert weight
        if eplb_perf_logger.enabled:
            weight_start_event = torch.npu.Event(enable_timing=True)
            weight_end_event = torch.npu.Event(enable_timing=True)
            weight_start_event.record()
        buffer_tensor_id = 0
        for recv_expert_info in self.recv_expert_list:
            local_expert_to_replace, buffer_tensor_id = recv_expert_info
            self.eplb_adaptor.do_update_expert_weight(self.layer_id, local_expert_to_replace, buffer_tensor_id)
        if eplb_perf_logger.enabled:
            weight_end_event.record()
            self.pending_update_events.append(
                ("expert_weight_update_execute", self.eplb_cycle_round, weight_start_event, weight_end_event)
            )

        logger.debug(
            "[eplb/d2d_loader] Layer %s D2D transfer completed, updated_experts=%s",
            self.layer_id,
            len(self.recv_expert_list),
        )

        if self.layer_id == self.eplb_adaptor.num_moe_layers - 1:
            logger.info(
                "[eplb/d2d_loader] Full expert weight update cycle completed, total_layers=%s",
                self.eplb_adaptor.num_moe_layers,
            )

        self.recv_expert_list = []
        self.updated_expert_map = None
        self.layer_id = -1
        self.state = ExpertWeightUpdateState.WAITING
