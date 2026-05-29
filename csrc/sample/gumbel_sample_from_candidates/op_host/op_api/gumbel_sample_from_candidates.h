/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef VLLM_ASCEND_L0_GUMBEL_SAMPLE_FROM_CANDIDATES_H
#define VLLM_ASCEND_L0_GUMBEL_SAMPLE_FROM_CANDIDATES_H

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* GumbelSampleFromCandidates(const aclTensor* logits,
                                            const aclTensor* candidateIds,
                                            const aclTensor* candidateLens,
                                            const aclTensor* idxMapping,
                                            const aclTensor* seeds,
                                            const aclTensor* positions,
                                            aclOpExecutor* executor);
}

#endif  // VLLM_ASCEND_L0_GUMBEL_SAMPLE_FROM_CANDIDATES_H
