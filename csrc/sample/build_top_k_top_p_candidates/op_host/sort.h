/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef VLLM_ASCEND_SAMPLE_L0_SORT_H
#define VLLM_ASCEND_SAMPLE_L0_SORT_H

#include "opdev/fast_vector.h"
#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<aclTensor*, aclTensor*> Sort(const aclTensor* self,
                                              int64_t dim,
                                              bool descending,
                                              bool stable,
                                              op::DataType indicesType,
                                              aclOpExecutor* executor);
}

#endif  // VLLM_ASCEND_SAMPLE_L0_SORT_H
