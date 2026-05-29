/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef VLLM_ASCEND_ACLNN_BUILD_TOP_K_TOP_P_CANDIDATES_H
#define VLLM_ASCEND_ACLNN_BUILD_TOP_K_TOP_P_CANDIDATES_H

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnBuildTopKTopPCandidatesGetWorkspaceSize(
    const aclTensor* logits,
    const aclTensor* idxMapping,
    const aclTensor* temperature,
    const aclTensor* p,
    const aclTensor* k,
    int64_t candidateCapacity,
    bool applyTemperature,
    aclTensor* candidateLogits,
    aclTensor* candidateIds,
    aclTensor* candidateLens,
    aclTensor* candidateStatus,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnBuildTopKTopPCandidates(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // VLLM_ASCEND_ACLNN_BUILD_TOP_K_TOP_P_CANDIDATES_H
