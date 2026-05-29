/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "build_top_k_top_p_candidates.h"

using vllm_ascend::sampling::BuildTopKTopPCandidatesKernel;
using vllm_ascend::sampling::BuildTopKTopPCandidatesTilingData;

extern "C" __global__ __aicore__ void build_top_k_top_p_candidates(
    GM_ADDR sorted_value, GM_ADDR sorted_indices, GM_ADDR idx_mapping,
    GM_ADDR temperature, GM_ADDR p, GM_ADDR k, GM_ADDR candidate_logits,
    GM_ADDR candidate_ids, GM_ADDR candidate_lens, GM_ADDR candidate_status,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AscendC::AIC) {
        return;
    }

    REGISTER_TILING_DEFAULT(BuildTopKTopPCandidatesTilingData);
    GET_TILING_DATA_WITH_STRUCT(BuildTopKTopPCandidatesTilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    BuildTopKTopPCandidatesKernel<DTYPE_SORTED_VALUE> op;
    op.Init(sorted_value, sorted_indices, idx_mapping, temperature, p, k,
            candidate_logits, candidate_ids, candidate_lens, candidate_status,
            &tilingData, &pipe);
    op.Process();
}
