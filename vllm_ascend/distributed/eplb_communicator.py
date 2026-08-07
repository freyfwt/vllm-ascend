# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from vllm.distributed.eplb.eplb_communicator import (
    TorchDistGlooStagedEplbCommunicator,
    TorchDistNcclEplbCommunicator,
)


class HcclEplbCommunicator(TorchDistNcclEplbCommunicator):
    """Torch-distributed EPLB transfers over the HCCL device group."""

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        # Ascend keeps each expert in an independent persistent tensor. The
        # upstream profile collective expects every weight entry to be one
        # stacked tensor, so reserve HCCL buffers during actual P2P transfers.
        return False


class AscendGlooEplbCommunicator(TorchDistGlooStagedEplbCommunicator):
    """Gloo CPU-staging EPLB communicator for async mode on Ascend.

    Gloo uses CPU-side P2P and does not require the NCCL/HCCL buffer
    reservation collective that the upstream profile path runs. Disabling
    it also avoids passing Ascend's EplbExpertTensorList to all_gather,
    which does not implement the __torch_function__ protocol for
    distributed collectives.
    """

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        return False
