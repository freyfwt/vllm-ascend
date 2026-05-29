/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "build_top_k_top_p_candidates.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(BuildTopKTopPCandidates);

std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*>
BuildTopKTopPCandidates(const aclTensor* sortedValue,
                        const aclTensor* sortedIndices,
                        const aclTensor* idxMapping,
                        const aclTensor* temperature,
                        const aclTensor* p,
                        const aclTensor* k,
                        int64_t candidateCapacity,
                        bool applyTemperature,
                        aclOpExecutor* executor)
{
    L0_DFX(BuildTopKTopPCandidates, sortedValue, sortedIndices, idxMapping,
           temperature, p, k, candidateCapacity, applyTemperature);
    gert::Shape candidateShape({sortedValue->GetViewShape().GetDim(0), candidateCapacity});
    gert::Shape rowShape({sortedValue->GetViewShape().GetDim(0)});
    auto candidateLogits = executor->AllocTensor(candidateShape, sortedValue->GetDataType(), Format::FORMAT_ND);
    auto candidateIds = executor->AllocTensor(candidateShape, DataType::DT_INT32, Format::FORMAT_ND);
    auto candidateLens = executor->AllocTensor(rowShape, DataType::DT_INT32, Format::FORMAT_ND);
    auto candidateStatus = executor->AllocTensor(rowShape, DataType::DT_INT32, Format::FORMAT_ND);
    if (candidateLogits == nullptr || candidateIds == nullptr ||
        candidateLens == nullptr || candidateStatus == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc BuildTopKTopPCandidates outputs failed.");
        return std::make_tuple(nullptr, nullptr, nullptr, nullptr);
    }
    if (p == nullptr) {
        p = executor->AllocTensor(sortedValue->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (k == nullptr) {
        k = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BuildTopKTopPCandidates,
        OP_INPUT(sortedValue, sortedIndices, idxMapping, temperature, p, k),
        OP_OUTPUT(candidateLogits, candidateIds, candidateLens, candidateStatus),
        OP_ATTR(candidateCapacity, applyTemperature));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE for BuildTopKTopPCandidates failed.");
        return std::make_tuple(nullptr, nullptr, nullptr, nullptr);
    }
    return std::make_tuple(candidateLogits, candidateIds, candidateLens, candidateStatus);
}
}  // namespace l0op
