/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef VLLM_ASCEND_SAMPLE_GUMBEL_SAMPLE_COMMON_H
#define VLLM_ASCEND_SAMPLE_GUMBEL_SAMPLE_COMMON_H

#include "counter_rng.h"
#include "gumbel_sampling_tiling_data.h"
#include "kernel_operator.h"

namespace vllm_ascend {
namespace sampling {

constexpr float GUMBEL_NEG_INF = -3.4028234663852886e38F;
constexpr int32_t GUMBEL_INVALID_TOKEN_ID = 2147483647;

template <typename T>
__aicore__ inline float ToFloatValue(T value)
{
    return static_cast<float>(value);
}

template <>
__aicore__ inline float ToFloatValue<bfloat16_t>(bfloat16_t value)
{
    return AscendC::ToFloat(value);
}

template <typename T>
__aicore__ inline T FromFloatValue(float value)
{
    return static_cast<T>(value);
}

template <>
__aicore__ inline bfloat16_t FromFloatValue<bfloat16_t>(float value)
{
    return AscendC::ToBfloat16(value);
}

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint64_t Align32(uint64_t value)
{
    return (value + 31UL) / 32UL * 32UL;
}

template <typename LogitT, bool COMPACT_MODE>
class GumbelSampleKernel {
public:
    __aicore__ inline GumbelSampleKernel() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR candidateIds,
                                GM_ADDR candidateLens, GM_ADDR idxMapping,
                                GM_ADDR temperature, GM_ADDR seeds,
                                GM_ADDR positions, GM_ADDR sampledTokenIds,
                                GM_ADDR userWorkspace,
                                const GumbelSampleTilingData* tilingData,
                                AscendC::TPipe* pipe)
    {
        tilingData_ = tilingData;
        pipe_ = pipe;
        batchSize_ = tilingData_->batchSize;
        effectiveSize_ = COMPACT_MODE ? tilingData_->candidateCapacity : tilingData_->vocabSize;
        tileSize_ = tilingData_->tileSize;
        tileSizeAligned_ = tilingData_->tileSizeAligned;
        numTiles_ = tilingData_->numTiles;
        usedCoreNum_ = tilingData_->usedCoreNum;
        applyTemperature_ = tilingData_->applyTemperature != 0;

        logitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ LogitT*>(logits),
                                  static_cast<uint64_t>(batchSize_) * effectiveSize_);
        if constexpr (COMPACT_MODE) {
            candidateIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(candidateIds),
                                            static_cast<uint64_t>(batchSize_) * effectiveSize_);
            candidateLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(candidateLens), batchSize_);
        }
        idxMappingGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxMapping), batchSize_);
        if constexpr (!COMPACT_MODE) {
            temperatureGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(temperature),
                                           tilingData_->requestCount);
        }
        seedsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(seeds),
                                 tilingData_->requestCount);
        positionsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(positions), batchSize_);
        sampledTokenIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sampledTokenIds), batchSize_);

        tileMaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                                       userWorkspace + tilingData_->workspaceTileMaxOffset),
                                   static_cast<uint64_t>(batchSize_) * numTiles_);
        rowMaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                                      userWorkspace + tilingData_->workspaceRowMaxOffset),
                                  batchSize_);
        bestScoreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                                         userWorkspace + tilingData_->workspaceBestScoreOffset),
                                     static_cast<uint64_t>(batchSize_) * numTiles_);
        bestIdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                                      userWorkspace + tilingData_->workspaceBestIdOffset),
                                  static_cast<uint64_t>(batchSize_) * numTiles_);

        pipe_->InitBuffer(scoreBuf_, tileSizeAligned_ * sizeof(float));
        pipe_->InitBuffer(noiseBuf_, tileSizeAligned_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        PhaseTileMax();
        AscendC::SyncAll();
        PhaseRowMax();
        AscendC::SyncAll();
        PhaseTileBest();
        AscendC::SyncAll();
        PhaseFinalBest();
    }

private:
    __aicore__ inline uint32_t ValidCount(uint32_t row, uint32_t tile)
    {
        uint32_t rowSize = effectiveSize_;
        if constexpr (COMPACT_MODE) {
            int32_t lens = candidateLensGm_.GetValue(row);
            rowSize = lens > 0 ? static_cast<uint32_t>(lens) : 0U;
            if (rowSize > effectiveSize_) {
                rowSize = effectiveSize_;
            }
        }
        uint32_t start = tile * tileSize_;
        if (start >= rowSize) {
            return 0;
        }
        return MinU32(tileSize_, rowSize - start);
    }

    __aicore__ inline float RowTemperature(uint32_t row, int32_t req)
    {
        if constexpr (COMPACT_MODE) {
            (void)row;
            (void)req;
            return 1.0F;
        } else {
            return temperatureGm_.GetValue(static_cast<uint32_t>(req));
        }
    }

    __aicore__ inline float LoadScaledLogit(uint32_t row, uint32_t col,
                                            float temperature)
    {
        float value = ToFloatValue(logitsGm_.GetValue(static_cast<uint64_t>(row) * effectiveSize_ + col));
        if constexpr (!COMPACT_MODE) {
            if (applyTemperature_ && temperature != 0.0F && temperature != 1.0F) {
                value = value / temperature;
            }
        }
        return value;
    }

    __aicore__ inline int32_t TokenId(uint32_t row, uint32_t col)
    {
        if constexpr (COMPACT_MODE) {
            return candidateIdsGm_.GetValue(static_cast<uint64_t>(row) * effectiveSize_ + col);
        } else {
            return static_cast<int32_t>(col);
        }
    }

    __aicore__ inline void PhaseTileMax()
    {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t totalTiles = batchSize_ * numTiles_;
        for (uint32_t task = coreIdx; task < totalTiles; task += usedCoreNum_) {
            uint32_t row = task / numTiles_;
            uint32_t tile = task % numTiles_;
            uint32_t valid = ValidCount(row, tile);
            int32_t req = idxMappingGm_.GetValue(row);
            float temperature = RowTemperature(row, req);
            float best = GUMBEL_NEG_INF;
            uint32_t start = tile * tileSize_;
            for (uint32_t i = 0; i < valid; ++i) {
                float value = LoadScaledLogit(row, start + i, temperature);
                if (value > best) {
                    best = value;
                }
            }
            tileMaxGm_.SetValue(static_cast<uint64_t>(row) * numTiles_ + tile, best);
        }
    }

    __aicore__ inline void PhaseRowMax()
    {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        for (uint32_t row = coreIdx; row < batchSize_; row += usedCoreNum_) {
            float best = GUMBEL_NEG_INF;
            for (uint32_t tile = 0; tile < numTiles_; ++tile) {
                float value = tileMaxGm_.GetValue(static_cast<uint64_t>(row) * numTiles_ + tile);
                if (value > best) {
                    best = value;
                }
            }
            rowMaxGm_.SetValue(row, best);
        }
    }

    __aicore__ inline void FillTileScores(uint32_t row, uint32_t tile,
                                          uint32_t valid, bool greedy,
                                          float temperature, int64_t seed,
                                          int64_t position)
    {
        auto score = scoreBuf_.Get<float>();
        auto noise = noiseBuf_.Get<float>();
        uint32_t start = tile * tileSize_;
        float rowMax = rowMaxGm_.GetValue(row);
        for (uint32_t i = 0; i < tileSizeAligned_; ++i) {
            if (i < valid) {
                uint32_t col = start + i;
                float value = LoadScaledLogit(row, col, temperature);
                if (greedy) {
                    score.SetValue(i, value);
                    noise.SetValue(i, 1.0F);
                } else {
                    int32_t tokenId = TokenId(row, col);
                    score.SetValue(i, value - rowMax);
                    noise.SetValue(i, UniformFromCounter(seed, position, tokenId));
                }
            } else {
                score.SetValue(i, GUMBEL_NEG_INF);
                noise.SetValue(i, 1.0F);
            }
        }
        if (!greedy) {
            AscendC::Log(noise, noise, tileSizeAligned_);
            AscendC::Muls(noise, noise, -1.0F, tileSizeAligned_);
            AscendC::Exp(score, score, tileSizeAligned_);
            AscendC::Div(score, score, noise, tileSizeAligned_);
        }
    }

    __aicore__ inline void PhaseTileBest()
    {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t totalTiles = batchSize_ * numTiles_;
        auto score = scoreBuf_.Get<float>();
        for (uint32_t task = coreIdx; task < totalTiles; task += usedCoreNum_) {
            uint32_t row = task / numTiles_;
            uint32_t tile = task % numTiles_;
            uint32_t valid = ValidCount(row, tile);
            if (valid == 0) {
                bestScoreGm_.SetValue(static_cast<uint64_t>(row) * numTiles_ + tile, GUMBEL_NEG_INF);
                bestIdGm_.SetValue(static_cast<uint64_t>(row) * numTiles_ + tile,
                                   GUMBEL_INVALID_TOKEN_ID);
                continue;
            }

            int32_t req = idxMappingGm_.GetValue(row);
            float temperature = RowTemperature(row, req);
            bool greedy = false;
            if constexpr (COMPACT_MODE) {
                greedy = (candidateLensGm_.GetValue(row) == 1);
            } else {
                greedy = (temperature == 0.0F);
            }
            int64_t seed = greedy ? 0 : seedsGm_.GetValue(static_cast<uint32_t>(req));
            int64_t position = greedy ? 0 : positionsGm_.GetValue(row);
            FillTileScores(row, tile, valid, greedy, temperature, seed, position);

            float bestScore = GUMBEL_NEG_INF;
            int32_t bestToken = 0;
            bool hasLocalCandidate = false;
            uint32_t start = tile * tileSize_;
            for (uint32_t i = 0; i < valid; ++i) {
                int32_t tokenId = TokenId(row, start + i);
                float candidateScore = score.GetValue(i);
                if (!hasLocalCandidate ||
                    candidateScore > bestScore ||
                    (candidateScore == bestScore && tokenId < bestToken)) {
                    bestScore = candidateScore;
                    bestToken = tokenId;
                    hasLocalCandidate = true;
                }
            }
            bestScoreGm_.SetValue(static_cast<uint64_t>(row) * numTiles_ + tile, bestScore);
            bestIdGm_.SetValue(static_cast<uint64_t>(row) * numTiles_ + tile, bestToken);
        }
    }

    __aicore__ inline void PhaseFinalBest()
    {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        for (uint32_t row = coreIdx; row < batchSize_; row += usedCoreNum_) {
            float bestScore = GUMBEL_NEG_INF;
            int32_t bestToken = 0;
            bool hasCandidate = false;
            for (uint32_t tile = 0; tile < numTiles_; ++tile) {
                float candidateScore = bestScoreGm_.GetValue(static_cast<uint64_t>(row) * numTiles_ + tile);
                int32_t candidateToken = bestIdGm_.GetValue(static_cast<uint64_t>(row) * numTiles_ + tile);
                if (candidateToken == GUMBEL_INVALID_TOKEN_ID &&
                    candidateScore == GUMBEL_NEG_INF) {
                    continue;
                }
                if (!hasCandidate ||
                    candidateScore > bestScore ||
                    (candidateScore == bestScore && candidateToken < bestToken)) {
                    bestScore = candidateScore;
                    bestToken = candidateToken;
                    hasCandidate = true;
                }
            }
            sampledTokenIdsGm_.SetValue(row, bestToken);
        }
    }

private:
    const GumbelSampleTilingData* tilingData_ = nullptr;
    AscendC::TPipe* pipe_ = nullptr;
    uint32_t batchSize_ = 0;
    uint32_t effectiveSize_ = 0;
    uint32_t tileSize_ = 0;
    uint32_t tileSizeAligned_ = 0;
    uint32_t numTiles_ = 0;
    uint32_t usedCoreNum_ = 1;
    bool applyTemperature_ = true;

    AscendC::GlobalTensor<LogitT> logitsGm_;
    AscendC::GlobalTensor<int32_t> candidateIdsGm_;
    AscendC::GlobalTensor<int32_t> candidateLensGm_;
    AscendC::GlobalTensor<int32_t> idxMappingGm_;
    AscendC::GlobalTensor<float> temperatureGm_;
    AscendC::GlobalTensor<int64_t> seedsGm_;
    AscendC::GlobalTensor<int64_t> positionsGm_;
    AscendC::GlobalTensor<int32_t> sampledTokenIdsGm_;
    AscendC::GlobalTensor<float> tileMaxGm_;
    AscendC::GlobalTensor<float> rowMaxGm_;
    AscendC::GlobalTensor<float> bestScoreGm_;
    AscendC::GlobalTensor<int32_t> bestIdGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scoreBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> noiseBuf_;
};

}  // namespace sampling
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_SAMPLE_GUMBEL_SAMPLE_COMMON_H
