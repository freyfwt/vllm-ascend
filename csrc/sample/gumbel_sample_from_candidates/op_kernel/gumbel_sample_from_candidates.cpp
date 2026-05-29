/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "gumbel_sample_from_candidates.h"

using vllm_ascend::sampling::GUMBEL_SAMPLE_COMPACT_CANDIDATES;
using vllm_ascend::sampling::GumbelSampleKernel;
using vllm_ascend::sampling::GumbelSampleTilingData;

extern "C" __global__ __aicore__ void gumbel_sample_from_candidates(
    GM_ADDR logits, GM_ADDR candidate_ids, GM_ADDR candidate_lens,
    GM_ADDR idx_mapping, GM_ADDR seeds, GM_ADDR positions,
    GM_ADDR sampled_token_ids, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AscendC::AIC) {
        return;
    }

    REGISTER_TILING_DEFAULT(GumbelSampleTilingData);
    GET_TILING_DATA_WITH_STRUCT(GumbelSampleTilingData, tilingData, tiling);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr ||
        tilingData.sampleMode != GUMBEL_SAMPLE_COMPACT_CANDIDATES) {
        return;
    }

    AscendC::TPipe pipe;
    GumbelSampleKernel<DTYPE_LOGITS, true> op;
    op.Init(logits, candidate_ids, candidate_lens, idx_mapping, nullptr, seeds,
            positions, sampled_token_ids, userWorkspace, &tilingData, &pipe);
    op.Process();
}
