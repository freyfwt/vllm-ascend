/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef VLLM_ASCEND_GUMBEL_SAMPLE_TORCH_ADPT_H
#define VLLM_ASCEND_GUMBEL_SAMPLE_TORCH_ADPT_H

#include <torch/extension.h>
#include "../../aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {

inline at::Tensor npu_gumbel_sample(
    const at::Tensor& logits,
    const at::Tensor& idx_mapping,
    const at::Tensor& temperature,
    const at::Tensor& seeds,
    const at::Tensor& positions,
    bool apply_temperature)
{
    TORCH_CHECK(logits.dim() == 2, "npu_gumbel_sample: logits must be 2D");
    TORCH_CHECK(idx_mapping.dim() == 1, "npu_gumbel_sample: idx_mapping must be 1D");
    TORCH_CHECK(temperature.dim() == 1, "npu_gumbel_sample: temperature must be 1D");
    TORCH_CHECK(seeds.dim() == 1, "npu_gumbel_sample: seeds must be 1D");
    TORCH_CHECK(positions.dim() == 1, "npu_gumbel_sample: positions must be 1D");
    TORCH_CHECK(idx_mapping.scalar_type() == at::kInt,
                "npu_gumbel_sample: idx_mapping must be int32");
    TORCH_CHECK(temperature.scalar_type() == at::kFloat,
                "npu_gumbel_sample: temperature must be float32");
    TORCH_CHECK(seeds.scalar_type() == at::kLong,
                "npu_gumbel_sample: seeds must be int64");
    TORCH_CHECK(positions.scalar_type() == at::kLong,
                "npu_gumbel_sample: positions must be int64");
    TORCH_CHECK(logits.scalar_type() == at::kFloat ||
                    logits.scalar_type() == at::kHalf ||
                    logits.scalar_type() == at::kBFloat16,
                "npu_gumbel_sample: logits must be float16, bfloat16, or float32");
    TORCH_CHECK(idx_mapping.size(0) == logits.size(0),
                "npu_gumbel_sample: idx_mapping batch mismatch");
    TORCH_CHECK(positions.size(0) == logits.size(0),
                "npu_gumbel_sample: positions batch mismatch");
    TORCH_CHECK(temperature.size(0) == seeds.size(0),
                "npu_gumbel_sample: temperature and seeds length mismatch");

    auto output = at::empty({logits.size(0)}, logits.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnGumbelSample, logits, idx_mapping, temperature, seeds,
                 positions, apply_temperature, output);
    return output;
}

}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_GUMBEL_SAMPLE_TORCH_ADPT_H
