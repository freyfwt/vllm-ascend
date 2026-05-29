/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef VLLM_ASCEND_BUILD_TOP_K_TOP_P_CANDIDATES_TORCH_ADPT_H
#define VLLM_ASCEND_BUILD_TOP_K_TOP_P_CANDIDATES_TORCH_ADPT_H

#include <tuple>
#include <torch/extension.h>
#include "../../aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_build_top_k_top_p_candidates(
    const at::Tensor& logits,
    const at::Tensor& idx_mapping,
    const at::Tensor& temperature,
    const c10::optional<at::Tensor>& p,
    const c10::optional<at::Tensor>& k,
    int64_t candidate_capacity,
    bool apply_temperature)
{
    TORCH_CHECK(logits.dim() == 2,
                "npu_build_top_k_top_p_candidates: logits must be 2D");
    TORCH_CHECK(candidate_capacity > 0,
                "npu_build_top_k_top_p_candidates: candidate_capacity must be positive");
    TORCH_CHECK(idx_mapping.dim() == 1 && idx_mapping.size(0) == logits.size(0),
                "npu_build_top_k_top_p_candidates: idx_mapping batch mismatch");
    TORCH_CHECK(temperature.dim() == 1,
                "npu_build_top_k_top_p_candidates: temperature must be 1D");
    TORCH_CHECK(idx_mapping.scalar_type() == at::kInt,
                "npu_build_top_k_top_p_candidates: idx_mapping must be int32");
    TORCH_CHECK(temperature.scalar_type() == at::kFloat,
                "npu_build_top_k_top_p_candidates: temperature must be float32");
    TORCH_CHECK(logits.scalar_type() == at::kFloat ||
                    logits.scalar_type() == at::kHalf ||
                    logits.scalar_type() == at::kBFloat16,
                "npu_build_top_k_top_p_candidates: logits must be float16, bfloat16, or float32");
    if (p.has_value()) {
        TORCH_CHECK(p.value().dim() == 1 && p.value().size(0) == logits.size(0),
                    "npu_build_top_k_top_p_candidates: p batch mismatch");
    }
    if (k.has_value()) {
        TORCH_CHECK(k.value().dim() == 1 && k.value().size(0) == logits.size(0),
                    "npu_build_top_k_top_p_candidates: k batch mismatch");
        TORCH_CHECK(k.value().scalar_type() == at::kInt,
                    "npu_build_top_k_top_p_candidates: k must be int32");
    }

    c10::optional<at::Tensor> p_for_op = c10::nullopt;
    at::Tensor p_cast;
    if (p.has_value()) {
        p_cast = p.value().scalar_type() == logits.scalar_type()
                     ? p.value()
                     : p.value().to(logits.scalar_type());
        p_for_op = p_cast;
    }

    auto candidate_logits = at::empty(
        {logits.size(0), candidate_capacity}, logits.options());
    auto int_options = logits.options().dtype(at::kInt);
    auto candidate_ids = at::empty({logits.size(0), candidate_capacity}, int_options);
    auto candidate_lens = at::empty({logits.size(0)}, int_options);
    auto candidate_status = at::empty({logits.size(0)}, int_options);

    EXEC_NPU_CMD(aclnnBuildTopKTopPCandidates, logits, idx_mapping,
                 temperature, p_for_op, k, candidate_capacity,
                 apply_temperature, candidate_logits, candidate_ids,
                 candidate_lens, candidate_status);
    return std::make_tuple(candidate_logits, candidate_ids,
                           candidate_lens, candidate_status);
}

}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_BUILD_TOP_K_TOP_P_CANDIDATES_TORCH_ADPT_H
