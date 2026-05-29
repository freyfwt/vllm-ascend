/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef VLLM_ASCEND_SAMPLE_GUMBEL_SAMPLING_TILING_DATA_H
#define VLLM_ASCEND_SAMPLE_GUMBEL_SAMPLING_TILING_DATA_H

#include <cstdint>

namespace vllm_ascend {
namespace sampling {

constexpr uint32_t GUMBEL_SAMPLE_FULL_VOCAB = 0;
constexpr uint32_t GUMBEL_SAMPLE_COMPACT_CANDIDATES = 1;
constexpr uint32_t GUMBEL_SAMPLE_TILE_SIZE = 1024;
constexpr uint32_t GUMBEL_SAMPLE_SYS_WORKSPACE_BYTES = 16U * 1024U * 1024U;

struct GumbelSampleTilingData {
    uint32_t batchSize;
    uint32_t vocabSize;
    uint32_t candidateCapacity;
    uint32_t requestCount;
    uint32_t tileSize;
    uint32_t tileSizeAligned;
    uint32_t numTiles;
    uint32_t usedCoreNum;
    uint32_t logitsDtypeBytes;
    uint32_t applyTemperature;
    uint32_t sampleMode;
    uint32_t workspaceTileMaxOffset;
    uint32_t workspaceRowMaxOffset;
    uint32_t workspaceBestScoreOffset;
    uint32_t workspaceBestIdOffset;
    uint32_t workspaceBytes;
};

struct BuildTopKTopPCandidatesTilingData {
    uint32_t batchSize;
    uint32_t vocabSize;
    uint32_t candidateCapacity;
    uint32_t requestCount;
    uint32_t tileSize;
    uint32_t tileSizeAligned;
    uint32_t usedCoreNum;
    uint32_t logitsDtypeBytes;
    uint32_t applyTemperature;
    uint32_t hasTopP;
    uint32_t hasTopK;
};

}  // namespace sampling
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_SAMPLE_GUMBEL_SAMPLING_TILING_DATA_H
