/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef VLLM_ASCEND_BUILD_TOP_K_TOP_P_CANDIDATES_KERNEL_H
#define VLLM_ASCEND_BUILD_TOP_K_TOP_P_CANDIDATES_KERNEL_H

#include "../../sampling_common/gumbel_sample_common.h"

namespace vllm_ascend {
namespace sampling {

constexpr int32_t CANDIDATE_STATUS_COMPACT = 0;
constexpr int32_t CANDIDATE_STATUS_NO_FILTER = 1;
constexpr int32_t CANDIDATE_STATUS_OVERFLOW = 2;

template <typename LogitT>
class BuildTopKTopPCandidatesKernel {
public:
    __aicore__ inline BuildTopKTopPCandidatesKernel() {}

    __aicore__ inline void Init(GM_ADDR sortedValue, GM_ADDR sortedIndices,
                                GM_ADDR idxMapping, GM_ADDR temperature,
                                GM_ADDR p, GM_ADDR k,
                                GM_ADDR candidateLogits,
                                GM_ADDR candidateIds,
                                GM_ADDR candidateLens,
                                GM_ADDR candidateStatus,
                                const BuildTopKTopPCandidatesTilingData* tilingData,
                                AscendC::TPipe* pipe)
    {
        tilingData_ = tilingData;
        pipe_ = pipe;
        batchSize_ = tilingData_->batchSize;
        vocabSize_ = tilingData_->vocabSize;
        candidateCapacity_ = tilingData_->candidateCapacity;
        tileSize_ = tilingData_->tileSize;
        tileSizeAligned_ = tilingData_->tileSizeAligned;
        usedCoreNum_ = tilingData_->usedCoreNum;
        applyTemperature_ = tilingData_->applyTemperature != 0;
        hasTopP_ = tilingData_->hasTopP != 0;
        hasTopK_ = tilingData_->hasTopK != 0;

        sortedValueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ LogitT*>(sortedValue),
                                       static_cast<uint64_t>(batchSize_) * vocabSize_);
        sortedIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sortedIndices),
                                         static_cast<uint64_t>(batchSize_) * vocabSize_);
        idxMappingGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxMapping), batchSize_);
        temperatureGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(temperature),
                                       tilingData_->requestCount);
        pGm_.SetGlobalBuffer(reinterpret_cast<__gm__ LogitT*>(p), batchSize_);
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(k), batchSize_);
        candidateLogitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ LogitT*>(candidateLogits),
                                           static_cast<uint64_t>(batchSize_) * candidateCapacity_);
        candidateIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(candidateIds),
                                        static_cast<uint64_t>(batchSize_) * candidateCapacity_);
        candidateLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(candidateLens), batchSize_);
        candidateStatusGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(candidateStatus), batchSize_);
        pipe_->InitBuffer(expBuf_, tileSizeAligned_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreIdx = usedCoreNum_ == 1 ? 0U : AscendC::GetBlockIdx();
        for (uint32_t row = coreIdx; row < batchSize_; row += usedCoreNum_) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline float RowTemperature(uint32_t row)
    {
        int32_t req = idxMappingGm_.GetValue(row);
        return temperatureGm_.GetValue(static_cast<uint32_t>(req));
    }

    __aicore__ inline float SortedValue(uint32_t row, uint32_t rank,
                                        float temperature)
    {
        float value = ToFloatValue(
            sortedValueGm_.GetValue(static_cast<uint64_t>(row) * vocabSize_ + rank));
        if (applyTemperature_ && temperature != 0.0F && temperature != 1.0F) {
            value = value / temperature;
        }
        return value;
    }

    __aicore__ inline uint32_t TopKLimit(uint32_t row)
    {
        if (!hasTopK_) {
            return vocabSize_;
        }
        int32_t kValue = kGm_.GetValue(row);
        if (kValue <= 0) {
            return 1;
        }
        uint32_t limit = static_cast<uint32_t>(kValue);
        return limit < vocabSize_ ? limit : vocabSize_;
    }

    __aicore__ inline float TopPValue(uint32_t row)
    {
        if (!hasTopP_) {
            return 1.0F;
        }
        return ToFloatValue(pGm_.GetValue(row));
    }

    __aicore__ inline void ClearCandidates(uint32_t row)
    {
        uint64_t base = static_cast<uint64_t>(row) * candidateCapacity_;
        for (uint32_t i = 0; i < candidateCapacity_; ++i) {
            candidateLogitsGm_.SetValue(base + i, FromFloatValue<LogitT>(GUMBEL_NEG_INF));
            candidateIdsGm_.SetValue(base + i, 0);
        }
    }

    __aicore__ inline float ExpTileSum(uint32_t row, uint32_t start,
                                       uint32_t valid, float rowMax,
                                       float temperature)
    {
        auto expLocal = expBuf_.Get<float>();
        for (uint32_t i = 0; i < tileSizeAligned_; ++i) {
            if (i < valid) {
                expLocal.SetValue(i, SortedValue(row, start + i, temperature) - rowMax);
            } else {
                expLocal.SetValue(i, GUMBEL_NEG_INF);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(expLocal, expLocal, tileSizeAligned_);
        AscendC::PipeBarrier<PIPE_V>();
        float sum = 0.0F;
        for (uint32_t i = 0; i < valid; ++i) {
            sum += expLocal.GetValue(i);
        }
        return sum;
    }

    __aicore__ inline uint32_t TopPLimit(uint32_t row, float pValue,
                                         float temperature)
    {
        if (!hasTopP_ || pValue >= 1.0F) {
            return vocabSize_;
        }
        if (pValue <= 0.0F) {
            return 1;
        }
        float rowMax = SortedValue(row, 0, temperature);
        float sum = 0.0F;
        for (uint32_t start = 0; start < vocabSize_; start += tileSize_) {
            uint32_t valid = MinU32(tileSize_, vocabSize_ - start);
            sum += ExpTileSum(row, start, valid, rowMax, temperature);
        }
        if (sum <= 0.0F) {
            return 1;
        }
        float threshold = sum * pValue;
        float cumulative = 0.0F;
        auto expLocal = expBuf_.Get<float>();
        for (uint32_t start = 0; start < vocabSize_; start += tileSize_) {
            uint32_t valid = MinU32(tileSize_, vocabSize_ - start);
            for (uint32_t i = 0; i < tileSizeAligned_; ++i) {
                if (i < valid) {
                    expLocal.SetValue(i, SortedValue(row, start + i, temperature) - rowMax);
                } else {
                    expLocal.SetValue(i, GUMBEL_NEG_INF);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(expLocal, expLocal, tileSizeAligned_);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < valid; ++i) {
                cumulative += expLocal.GetValue(i);
                if (cumulative >= threshold) {
                    return start + i + 1;
                }
            }
        }
        return vocabSize_;
    }

    __aicore__ inline void WriteCompactCandidates(uint32_t row, uint32_t count,
                                                  float temperature)
    {
        uint64_t srcBase = static_cast<uint64_t>(row) * vocabSize_;
        uint64_t dstBase = static_cast<uint64_t>(row) * candidateCapacity_;
        for (uint32_t i = 0; i < candidateCapacity_; ++i) {
            if (i < count) {
                float value = SortedValue(row, i, temperature);
                candidateLogitsGm_.SetValue(dstBase + i, FromFloatValue<LogitT>(value));
                candidateIdsGm_.SetValue(dstBase + i, sortedIndicesGm_.GetValue(srcBase + i));
            } else {
                candidateLogitsGm_.SetValue(dstBase + i, FromFloatValue<LogitT>(GUMBEL_NEG_INF));
                candidateIdsGm_.SetValue(dstBase + i, 0);
            }
        }
        candidateLensGm_.SetValue(row, static_cast<int32_t>(count));
        candidateStatusGm_.SetValue(row, CANDIDATE_STATUS_COMPACT);
    }

    __aicore__ inline void ProcessRow(uint32_t row)
    {
        ClearCandidates(row);
        float temperature = RowTemperature(row);
        if (temperature == 0.0F) {
            uint64_t srcBase = static_cast<uint64_t>(row) * vocabSize_;
            candidateLogitsGm_.SetValue(static_cast<uint64_t>(row) * candidateCapacity_,
                                        sortedValueGm_.GetValue(srcBase));
            candidateIdsGm_.SetValue(static_cast<uint64_t>(row) * candidateCapacity_,
                                     sortedIndicesGm_.GetValue(srcBase));
            candidateLensGm_.SetValue(row, 1);
            candidateStatusGm_.SetValue(row, CANDIDATE_STATUS_COMPACT);
            return;
        }

        uint32_t topKLimit = TopKLimit(row);
        float topPValue = TopPValue(row);
        bool activeTopK = hasTopK_ && topKLimit < vocabSize_;
        bool activeTopP = hasTopP_ && topPValue < 1.0F;
        if (!activeTopK && !activeTopP) {
            candidateLensGm_.SetValue(row, 0);
            candidateStatusGm_.SetValue(row, CANDIDATE_STATUS_NO_FILTER);
            return;
        }

        uint32_t topPLimit = TopPLimit(row, topPValue, temperature);
        uint32_t candidateCount = activeTopK ? topKLimit : vocabSize_;
        if (activeTopP && topPLimit < candidateCount) {
            candidateCount = topPLimit;
        }
        if (candidateCount == 0) {
            candidateCount = 1;
        }
        if (candidateCount > candidateCapacity_) {
            candidateLensGm_.SetValue(row, 0);
            candidateStatusGm_.SetValue(row, CANDIDATE_STATUS_OVERFLOW);
            return;
        }
        WriteCompactCandidates(row, candidateCount, temperature);
    }

private:
    const BuildTopKTopPCandidatesTilingData* tilingData_ = nullptr;
    AscendC::TPipe* pipe_ = nullptr;
    uint32_t batchSize_ = 0;
    uint32_t vocabSize_ = 0;
    uint32_t candidateCapacity_ = 0;
    uint32_t tileSize_ = 0;
    uint32_t tileSizeAligned_ = 0;
    uint32_t usedCoreNum_ = 1;
    bool applyTemperature_ = true;
    bool hasTopP_ = false;
    bool hasTopK_ = false;

    AscendC::GlobalTensor<LogitT> sortedValueGm_;
    AscendC::GlobalTensor<int32_t> sortedIndicesGm_;
    AscendC::GlobalTensor<int32_t> idxMappingGm_;
    AscendC::GlobalTensor<float> temperatureGm_;
    AscendC::GlobalTensor<LogitT> pGm_;
    AscendC::GlobalTensor<int32_t> kGm_;
    AscendC::GlobalTensor<LogitT> candidateLogitsGm_;
    AscendC::GlobalTensor<int32_t> candidateIdsGm_;
    AscendC::GlobalTensor<int32_t> candidateLensGm_;
    AscendC::GlobalTensor<int32_t> candidateStatusGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expBuf_;
};

}  // namespace sampling
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_BUILD_TOP_K_TOP_P_CANDIDATES_KERNEL_H
