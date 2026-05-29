/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "aclnn_build_top_k_top_p_candidates.h"
#include "build_top_k_top_p_candidates.h"
#include "../sort.h"
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
static const std::initializer_list<op::DataType> FLOAT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT};

static aclnnStatus CheckParams(const aclTensor* logits, const aclTensor* idxMapping,
                               const aclTensor* temperature, const aclTensor* p,
                               const aclTensor* k, int64_t candidateCapacity,
                               const aclTensor* candidateLogits,
                               const aclTensor* candidateIds,
                               const aclTensor* candidateLens,
                               const aclTensor* candidateStatus)
{
    OP_CHECK_NULL(logits, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(idxMapping, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(temperature, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(candidateLogits, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(candidateIds, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(candidateLens, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(candidateStatus, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_DTYPE_NOT_SUPPORT(logits, LOGITS_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(idxMapping, INT32_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(temperature, FLOAT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    if (p != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(p, logits->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    }
    if (k != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(k, INT32_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    }
    OP_CHECK_DTYPE_NOT_MATCH(candidateLogits, logits->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(candidateIds, INT32_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(candidateLens, INT32_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(candidateStatus, INT32_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(logits, DIM_TWO, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(idxMapping, DIM_ONE, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(temperature, DIM_ONE, return ACLNN_ERR_PARAM_INVALID);
    if (p != nullptr) {
        OP_CHECK_WRONG_DIMENSION(p, DIM_ONE, return ACLNN_ERR_PARAM_INVALID);
    }
    if (k != nullptr) {
        OP_CHECK_WRONG_DIMENSION(k, DIM_ONE, return ACLNN_ERR_PARAM_INVALID);
    }
    OP_CHECK_WRONG_DIMENSION(candidateLogits, DIM_TWO, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(candidateIds, DIM_TWO, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(candidateLens, DIM_ONE, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(candidateStatus, DIM_ONE, return ACLNN_ERR_PARAM_INVALID);

    int64_t batch = logits->GetViewShape().GetDim(0);
    int64_t vocab = logits->GetViewShape().GetDim(1);
    if (batch <= 0 || vocab <= 0 || candidateCapacity <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "logits and candidate capacity must be non-empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (idxMapping->GetViewShape().GetDim(0) != batch ||
        candidateLogits->GetViewShape().GetDim(0) != batch ||
        candidateLogits->GetViewShape().GetDim(1) != candidateCapacity ||
        candidateIds->GetViewShape().GetDim(0) != batch ||
        candidateIds->GetViewShape().GetDim(1) != candidateCapacity ||
        candidateLens->GetViewShape().GetDim(0) != batch ||
        candidateStatus->GetViewShape().GetDim(0) != batch) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "candidate output shapes must match batch and capacity.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (p != nullptr && p->GetViewShape().GetDim(0) != batch) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "p length must match batch.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (k != nullptr && k->GetViewShape().GetDim(0) != batch) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "k length must match batch.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
}  // namespace

extern "C" {
aclnnStatus aclnnBuildTopKTopPCandidatesGetWorkspaceSize(
    const aclTensor* logits, const aclTensor* idxMapping,
    const aclTensor* temperature, const aclTensor* p, const aclTensor* k,
    int64_t candidateCapacity, bool applyTemperature,
    aclTensor* candidateLogits, aclTensor* candidateIds,
    aclTensor* candidateLens, aclTensor* candidateStatus,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnBuildTopKTopPCandidates,
                   DFX_IN(logits, idxMapping, temperature, p, k),
                   DFX_OUT(candidateLogits, candidateIds, candidateLens, candidateStatus));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(logits, idxMapping, temperature, p, k, candidateCapacity,
                           candidateLogits, candidateIds, candidateLens, candidateStatus);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto logitsContiguous = l0op::Contiguous(logits, uniqueExecutor.get());
    auto idxMappingContiguous = l0op::Contiguous(idxMapping, uniqueExecutor.get());
    auto temperatureContiguous = l0op::Contiguous(temperature, uniqueExecutor.get());
    const aclTensor* pContiguous = nullptr;
    const aclTensor* kContiguous = nullptr;
    if (p != nullptr) {
        pContiguous = l0op::Contiguous(p, uniqueExecutor.get());
        CHECK_RET(pContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (k != nullptr) {
        kContiguous = l0op::Contiguous(k, uniqueExecutor.get());
        CHECK_RET(kContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(logitsContiguous != nullptr && idxMappingContiguous != nullptr &&
                  temperatureContiguous != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    auto sortResult = l0op::Sort(logitsContiguous, -1, true, true,
                                 op::DataType::DT_INT32, uniqueExecutor.get());
    const aclTensor* sortedValue = std::get<0>(sortResult);
    const aclTensor* sortedIndices = std::get<1>(sortResult);
    CHECK_RET(sortedValue != nullptr && sortedIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::BuildTopKTopPCandidates(
        sortedValue, sortedIndices, idxMappingContiguous, temperatureContiguous,
        pContiguous, kContiguous, candidateCapacity, applyTemperature,
        uniqueExecutor.get());
    const aclTensor* candidateLogitsResult = std::get<0>(result);
    const aclTensor* candidateIdsResult = std::get<1>(result);
    const aclTensor* candidateLensResult = std::get<2>(result);
    const aclTensor* candidateStatusResult = std::get<3>(result);
    CHECK_RET(candidateLogitsResult != nullptr && candidateIdsResult != nullptr &&
                  candidateLensResult != nullptr && candidateStatusResult != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(l0op::ViewCopy(candidateLogitsResult, candidateLogits, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(candidateIdsResult, candidateIds, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(candidateLensResult, candidateLens, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(candidateStatusResult, candidateStatus, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBuildTopKTopPCandidates(void* workspace, uint64_t workspaceSize,
                                         aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBuildTopKTopPCandidates);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}
