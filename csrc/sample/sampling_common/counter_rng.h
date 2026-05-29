/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef VLLM_ASCEND_SAMPLE_COUNTER_RNG_H
#define VLLM_ASCEND_SAMPLE_COUNTER_RNG_H

#include "kernel_operator.h"

namespace vllm_ascend {
namespace sampling {

constexpr float RNG_FLOAT_SCALE = 5.9604644775390625e-8F;  // 2^-24

__aicore__ inline uint32_t WangHash(uint32_t x, uint32_t seedLow,
                                    uint32_t seedHigh, uint32_t posLow)
{
    x ^= seedLow;
    x += posLow * 0x9e3779b9U;
    x ^= seedHigh * 0x85ebca6bU;
    x = (x ^ 61U) ^ (x >> 16);
    x *= 9U;
    x ^= (x >> 4);
    x *= 0x27d4eb2dU;
    x ^= (x >> 15);
    return x;
}

__aicore__ inline float UniformFromCounter(int64_t seed, int64_t position,
                                           int32_t tokenId)
{
    uint64_t seedBits = static_cast<uint64_t>(seed);
    uint32_t seedLow = static_cast<uint32_t>(seedBits & 0xffffffffULL);
    uint32_t seedHigh = static_cast<uint32_t>((seedBits >> 32) & 0xffffffffULL);
    uint32_t posLow = static_cast<uint32_t>(
        static_cast<uint64_t>(position) & 0xffffffffULL);
    uint32_t token = static_cast<uint32_t>(tokenId);
    uint32_t hash = WangHash(token, seedLow, seedHigh, posLow);
    uint32_t mantissa = (hash >> 8) & 0x00ffffffU;
    int32_t mantissaSigned = static_cast<int32_t>(mantissa);
    float u = (static_cast<float>(mantissaSigned) + 0.5F) * RNG_FLOAT_SCALE;
    if (u < RNG_FLOAT_SCALE) {
        u = RNG_FLOAT_SCALE;
    }
    const float upper = 1.0F - RNG_FLOAT_SCALE;
    if (u > upper) {
        u = upper;
    }
    return u;
}

}  // namespace sampling
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_SAMPLE_COUNTER_RNG_H
