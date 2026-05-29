/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "gumbel_sample.h"

using vllm_ascend::sampling::GUMBEL_SAMPLE_FULL_VOCAB;
using vllm_ascend::sampling::GumbelSampleKernel;
using vllm_ascend::sampling::GumbelSampleTilingData;

extern "C" __global__ __aicore__ void gumbel_sample(
    GM_ADDR logits, GM_ADDR idx_mapping, GM_ADDR temperature, GM_ADDR seeds,
    GM_ADDR positions, GM_ADDR sampled_token_ids, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AscendC::AIC) {
        return;
    }

    REGISTER_TILING_DEFAULT(GumbelSampleTilingData);
    GET_TILING_DATA_WITH_STRUCT(GumbelSampleTilingData, tilingData, tiling);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr || tilingData.sampleMode != GUMBEL_SAMPLE_FULL_VOCAB) {
        return;
    }

    AscendC::TPipe pipe;
    GumbelSampleKernel<DTYPE_LOGITS, false> op;
    op.Init(logits, nullptr, nullptr, idx_mapping, temperature, seeds, positions,
            sampled_token_ids, userWorkspace, &tilingData, &pipe);
    op.Process();
}
