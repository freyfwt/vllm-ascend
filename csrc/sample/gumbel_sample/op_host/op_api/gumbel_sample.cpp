/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "gumbel_sample.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GumbelSample);

const aclTensor* GumbelSample(const aclTensor* logits,
                             const aclTensor* idxMapping,
                             const aclTensor* temperature,
                             const aclTensor* seeds,
                             const aclTensor* positions,
                             bool applyTemperature,
                             aclOpExecutor* executor)
{
    L0_DFX(GumbelSample, logits, idxMapping, temperature, seeds, positions, applyTemperature);
    gert::Shape outputShape({logits->GetViewShape().GetDim(0)});
    auto output = executor->AllocTensor(outputShape, DataType::DT_INT32, Format::FORMAT_ND);
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc GumbelSample output failed.");
        return nullptr;
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        GumbelSample,
        OP_INPUT(logits, idxMapping, temperature, seeds, positions),
        OP_OUTPUT(output),
        OP_ATTR(applyTemperature));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE for GumbelSample failed.");
        return nullptr;
    }
    return output;
}
}  // namespace l0op
