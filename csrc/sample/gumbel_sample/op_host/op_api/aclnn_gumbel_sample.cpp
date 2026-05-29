/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "aclnn_gumbel_sample.h"
#include "gumbel_sample.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;

static const std::initializer_list<op::DataType> LOGITS_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> INT32_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> INT64_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> FLOAT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT};

static bool CheckNotNull(const aclTensor* logits, const aclTensor* idxMapping,
                         const aclTensor* temperature, const aclTensor* seeds,
                         const aclTensor* positions, const aclTensor* out)
{
    OP_CHECK_NULL(logits, return false);
    OP_CHECK_NULL(idxMapping, return false);
    OP_CHECK_NULL(temperature, return false);
    OP_CHECK_NULL(seeds, return false);
    OP_CHECK_NULL(positions, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* logits, const aclTensor* idxMapping,
                            const aclTensor* temperature, const aclTensor* seeds,
                            const aclTensor* positions, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(logits, LOGITS_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(idxMapping, INT32_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(temperature, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(seeds, INT64_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(positions, INT64_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, INT32_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckShapeValid(const aclTensor* logits, const aclTensor* idxMapping,
                            const aclTensor* temperature, const aclTensor* seeds,
                            const aclTensor* positions, const aclTensor* out)
{
    OP_CHECK_WRONG_DIMENSION(logits, DIM_TWO, return false);
    OP_CHECK_WRONG_DIMENSION(idxMapping, DIM_ONE, return false);
    OP_CHECK_WRONG_DIMENSION(temperature, DIM_ONE, return false);
    OP_CHECK_WRONG_DIMENSION(seeds, DIM_ONE, return false);
    OP_CHECK_WRONG_DIMENSION(positions, DIM_ONE, return false);
    OP_CHECK_WRONG_DIMENSION(out, DIM_ONE, return false);
    int64_t batch = logits->GetViewShape().GetDim(0);
    if (batch <= 0 || logits->GetViewShape().GetDim(1) <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "logits shape must be non-empty 2D.");
        return false;
    }
    if (idxMapping->GetViewShape().GetDim(0) != batch ||
        positions->GetViewShape().GetDim(0) != batch ||
        out->GetViewShape().GetDim(0) != batch) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "idxMapping, positions, and output must match logits batch.");
        return false;
    }
    if (temperature->GetViewShape().GetDim(0) != seeds->GetViewShape().GetDim(0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "temperature and seeds must have the same length.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* logits, const aclTensor* idxMapping,
                               const aclTensor* temperature, const aclTensor* seeds,
                               const aclTensor* positions, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(logits, idxMapping, temperature, seeds, positions, out),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(logits, idxMapping, temperature, seeds, positions, out),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(logits, idxMapping, temperature, seeds, positions, out),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
}  // namespace

extern "C" {
aclnnStatus aclnnGumbelSampleGetWorkspaceSize(
    const aclTensor* logits, const aclTensor* idxMapping,
    const aclTensor* temperature, const aclTensor* seeds,
    const aclTensor* positions, bool applyTemperature,
    aclTensor* sampledTokenIds, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnGumbelSample,
                   DFX_IN(logits, idxMapping, temperature, seeds, positions),
                   DFX_OUT(sampledTokenIds));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(logits, idxMapping, temperature, seeds, positions, sampledTokenIds);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto logitsContiguous = l0op::Contiguous(logits, uniqueExecutor.get());
    auto idxMappingContiguous = l0op::Contiguous(idxMapping, uniqueExecutor.get());
    auto temperatureContiguous = l0op::Contiguous(temperature, uniqueExecutor.get());
    auto seedsContiguous = l0op::Contiguous(seeds, uniqueExecutor.get());
    auto positionsContiguous = l0op::Contiguous(positions, uniqueExecutor.get());
    CHECK_RET(logitsContiguous != nullptr && idxMappingContiguous != nullptr &&
                  temperatureContiguous != nullptr && seedsContiguous != nullptr &&
                  positionsContiguous != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* result = l0op::GumbelSample(
        logitsContiguous, idxMappingContiguous, temperatureContiguous,
        seedsContiguous, positionsContiguous, applyTemperature,
        uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(result, sampledTokenIds, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGumbelSample(void* workspace, uint64_t workspaceSize,
                              aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGumbelSample);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}
