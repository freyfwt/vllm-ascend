/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef VLLM_ASCEND_ACLNN_GUMBEL_SAMPLE_H
#define VLLM_ASCEND_ACLNN_GUMBEL_SAMPLE_H

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnGumbelSampleGetWorkspaceSize(
    const aclTensor* logits,
    const aclTensor* idxMapping,
    const aclTensor* temperature,
    const aclTensor* seeds,
    const aclTensor* positions,
    bool applyTemperature,
    aclTensor* sampledTokenIds,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnGumbelSample(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // VLLM_ASCEND_ACLNN_GUMBEL_SAMPLE_H
