/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef VLLM_ASCEND_L0_BUILD_TOP_K_TOP_P_CANDIDATES_H
#define VLLM_ASCEND_L0_BUILD_TOP_K_TOP_P_CANDIDATES_H

#include <tuple>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

namespace l0op {
std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*>
BuildTopKTopPCandidates(const aclTensor* sortedValue,
                        const aclTensor* sortedIndices,
                        const aclTensor* idxMapping,
                        const aclTensor* temperature,
                        const aclTensor* p,
                        const aclTensor* k,
                        int64_t candidateCapacity,
                        bool applyTemperature,
                        aclOpExecutor* executor);
}

#endif  // VLLM_ASCEND_L0_BUILD_TOP_K_TOP_P_CANDIDATES_H
