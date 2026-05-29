/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef VLLM_ASCEND_L0_GUMBEL_SAMPLE_H
#define VLLM_ASCEND_L0_GUMBEL_SAMPLE_H

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* GumbelSample(const aclTensor* logits,
                             const aclTensor* idxMapping,
                             const aclTensor* temperature,
                             const aclTensor* seeds,
                             const aclTensor* positions,
                             bool applyTemperature,
                             aclOpExecutor* executor);
}

#endif  // VLLM_ASCEND_L0_GUMBEL_SAMPLE_H
