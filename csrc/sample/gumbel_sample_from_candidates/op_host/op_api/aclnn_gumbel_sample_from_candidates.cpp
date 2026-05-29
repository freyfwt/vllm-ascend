/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "aclnn_gumbel_sample_from_candidates.h"
#include "gumbel_sample_from_candidates.h"
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

static aclnnStatus CheckParams(const aclTensor* logits, const aclTensor* candidateIds,
                               const aclTensor* candidateLens, const aclTensor* idxMapping,
                               const aclTensor* seeds, const aclTensor* positions,
                               const aclTensor* out)
{
    OP_CHECK_NULL(logits, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(candidateIds, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(candidateLens, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(idxMapping, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(seeds, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(positions, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_DTYPE_NOT_SUPPORT(logits, LOGITS_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(candidateIds, INT32_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(candidateLens, INT32_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(idxMapping, INT32_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(seeds, INT64_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(positions, INT64_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, INT32_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(logits, DIM_TWO, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(candidateIds, DIM_TWO, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(candidateLens, DIM_ONE, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(idxMapping, DIM_ONE, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(seeds, DIM_ONE, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(positions, DIM_ONE, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(out, DIM_ONE, return ACLNN_ERR_PARAM_INVALID);
    int64_t batch = logits->GetViewShape().GetDim(0);
    int64_t capacity = logits->GetViewShape().GetDim(1);
    if (batch <= 0 || capacity <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "candidate logits shape must be non-empty 2D.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (candidateIds->GetViewShape().GetDim(0) != batch ||
        candidateIds->GetViewShape().GetDim(1) != capacity ||
        candidateLens->GetViewShape().GetDim(0) != batch ||
        idxMapping->GetViewShape().GetDim(0) != batch ||
        positions->GetViewShape().GetDim(0) != batch ||
        out->GetViewShape().GetDim(0) != batch) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "candidate tensors must have matching batch/capacity.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
}  // namespace

extern "C" {
aclnnStatus aclnnGumbelSampleFromCandidatesGetWorkspaceSize(
    const aclTensor* candidateLogits, const aclTensor* candidateIds,
    const aclTensor* candidateLens, const aclTensor* idxMapping,
    const aclTensor* seeds, const aclTensor* positions,
    aclTensor* sampledTokenIds, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnGumbelSampleFromCandidates,
                   DFX_IN(candidateLogits, candidateIds, candidateLens, idxMapping,
                          seeds, positions),
                   DFX_OUT(sampledTokenIds));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(candidateLogits, candidateIds, candidateLens, idxMapping,
                           seeds, positions, sampledTokenIds);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto logitsContiguous = l0op::Contiguous(candidateLogits, uniqueExecutor.get());
    auto idsContiguous = l0op::Contiguous(candidateIds, uniqueExecutor.get());
    auto lensContiguous = l0op::Contiguous(candidateLens, uniqueExecutor.get());
    auto idxMappingContiguous = l0op::Contiguous(idxMapping, uniqueExecutor.get());
    auto seedsContiguous = l0op::Contiguous(seeds, uniqueExecutor.get());
    auto positionsContiguous = l0op::Contiguous(positions, uniqueExecutor.get());
    CHECK_RET(logitsContiguous != nullptr && idsContiguous != nullptr &&
                  lensContiguous != nullptr && idxMappingContiguous != nullptr &&
                  seedsContiguous != nullptr && positionsContiguous != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* result = l0op::GumbelSampleFromCandidates(
        logitsContiguous, idsContiguous, lensContiguous, idxMappingContiguous,
        seedsContiguous, positionsContiguous, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(result, sampledTokenIds, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGumbelSampleFromCandidates(void* workspace, uint64_t workspaceSize,
                                            aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGumbelSampleFromCandidates);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}
