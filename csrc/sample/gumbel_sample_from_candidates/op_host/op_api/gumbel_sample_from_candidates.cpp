/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "gumbel_sample_from_candidates.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GumbelSampleFromCandidates);

const aclTensor* GumbelSampleFromCandidates(const aclTensor* logits,
                                            const aclTensor* candidateIds,
                                            const aclTensor* candidateLens,
                                            const aclTensor* idxMapping,
                                            const aclTensor* seeds,
                                            const aclTensor* positions,
                                            aclOpExecutor* executor)
{
    L0_DFX(GumbelSampleFromCandidates, logits, candidateIds, candidateLens,
           idxMapping, seeds, positions);
    gert::Shape outputShape({logits->GetViewShape().GetDim(0)});
    auto output = executor->AllocTensor(outputShape, DataType::DT_INT32, Format::FORMAT_ND);
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc GumbelSampleFromCandidates output failed.");
        return nullptr;
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        GumbelSampleFromCandidates,
        OP_INPUT(logits, candidateIds, candidateLens, idxMapping, seeds, positions),
        OP_OUTPUT(output));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE for GumbelSampleFromCandidates failed.");
        return nullptr;
    }
    return output;
}
}  // namespace l0op
