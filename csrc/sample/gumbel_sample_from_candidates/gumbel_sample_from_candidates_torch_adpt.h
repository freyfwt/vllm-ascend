/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef VLLM_ASCEND_GUMBEL_SAMPLE_FROM_CANDIDATES_TORCH_ADPT_H
#define VLLM_ASCEND_GUMBEL_SAMPLE_FROM_CANDIDATES_TORCH_ADPT_H

#include <torch/extension.h>
#include "../../aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {

inline at::Tensor npu_gumbel_sample_from_candidates(
    const at::Tensor& candidate_logits,
    const at::Tensor& candidate_ids,
    const at::Tensor& candidate_lens,
    const at::Tensor& idx_mapping,
    const at::Tensor& seeds,
    const at::Tensor& positions)
{
    TORCH_CHECK(candidate_logits.dim() == 2,
                "npu_gumbel_sample_from_candidates: candidate_logits must be 2D");
    TORCH_CHECK(candidate_ids.sizes() == candidate_logits.sizes(),
                "npu_gumbel_sample_from_candidates: candidate_ids shape mismatch");
    TORCH_CHECK(candidate_lens.dim() == 1 && candidate_lens.size(0) == candidate_logits.size(0),
                "npu_gumbel_sample_from_candidates: candidate_lens batch mismatch");
    TORCH_CHECK(idx_mapping.dim() == 1 && idx_mapping.size(0) == candidate_logits.size(0),
                "npu_gumbel_sample_from_candidates: idx_mapping batch mismatch");
    TORCH_CHECK(positions.dim() == 1 && positions.size(0) == candidate_logits.size(0),
                "npu_gumbel_sample_from_candidates: positions batch mismatch");
    TORCH_CHECK(candidate_ids.scalar_type() == at::kInt,
                "npu_gumbel_sample_from_candidates: candidate_ids must be int32");
    TORCH_CHECK(candidate_lens.scalar_type() == at::kInt,
                "npu_gumbel_sample_from_candidates: candidate_lens must be int32");
    TORCH_CHECK(idx_mapping.scalar_type() == at::kInt,
                "npu_gumbel_sample_from_candidates: idx_mapping must be int32");
    TORCH_CHECK(seeds.scalar_type() == at::kLong,
                "npu_gumbel_sample_from_candidates: seeds must be int64");
    TORCH_CHECK(positions.scalar_type() == at::kLong,
                "npu_gumbel_sample_from_candidates: positions must be int64");
    TORCH_CHECK(candidate_logits.scalar_type() == at::kFloat ||
                    candidate_logits.scalar_type() == at::kHalf ||
                    candidate_logits.scalar_type() == at::kBFloat16,
                "npu_gumbel_sample_from_candidates: logits must be float16, bfloat16, or float32");

    auto output = at::empty({candidate_logits.size(0)}, candidate_logits.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnGumbelSampleFromCandidates, candidate_logits, candidate_ids,
                 candidate_lens, idx_mapping, seeds, positions, output);
    return output;
}

}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_GUMBEL_SAMPLE_FROM_CANDIDATES_TORCH_ADPT_H
