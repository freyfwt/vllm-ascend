/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include <algorithm>
#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../sampling_common/gumbel_sampling_tiling_data.h"

using namespace ge;
using vllm_ascend::sampling::BuildTopKTopPCandidatesTilingData;
using vllm_ascend::sampling::GUMBEL_SAMPLE_TILE_SIZE;

namespace {
constexpr uint32_t SORTED_VALUE_INDEX = 0;
constexpr uint32_t TEMPERATURE_INDEX = 3;
constexpr uint32_t P_INDEX = 4;
constexpr uint32_t K_INDEX = 5;
constexpr uint32_t ATTR_CANDIDATE_CAPACITY_INDEX = 0;
constexpr uint32_t ATTR_APPLY_TEMPERATURE_INDEX = 1;

uint32_t DtypeBytes(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT ? 4U : 2U;
}

bool OptionalInputPresent(gert::TilingContext* context, uint32_t index)
{
    auto shape = context->GetOptionalInputShape(index);
    if (shape == nullptr) {
        return false;
    }
    return shape->GetStorageShape().GetDimNum() != 0;
}
}  // namespace

namespace optiling {

static ge::graphStatus TilingForBuildTopKTopPCandidates(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto* tilingData = context->GetTilingData<BuildTopKTopPCandidatesTilingData>();
    OPS_CHECK(tilingData == nullptr,
              OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    const gert::StorageShape* sortedShape = context->GetInputShape(SORTED_VALUE_INDEX);
    const gert::StorageShape* temperatureShape = context->GetInputShape(TEMPERATURE_INDEX);
    OPS_CHECK(sortedShape == nullptr,
              OPS_LOG_E(nodeName, "sorted_value shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_CHECK(temperatureShape == nullptr,
              OPS_LOG_E(nodeName, "temperature shape is nullptr."), return ge::GRAPH_FAILED);

    auto sortedStorageShape = sortedShape->GetStorageShape();
    OPS_CHECK(sortedStorageShape.GetDimNum() != 2,
              OPS_LOG_E(nodeName, "sorted_value must be 2D."), return ge::GRAPH_FAILED);
    uint32_t batchSize = static_cast<uint32_t>(sortedStorageShape.GetDim(0));
    uint32_t vocabSize = static_cast<uint32_t>(sortedStorageShape.GetDim(1));
    OPS_CHECK(batchSize == 0 || vocabSize == 0,
              OPS_LOG_E(nodeName, "batch and vocab must be greater than 0."), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr,
              OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    const int64_t* candidateCapacityPtr =
        attrs->GetAttrPointer<int64_t>(ATTR_CANDIDATE_CAPACITY_INDEX);
    OPS_CHECK(candidateCapacityPtr == nullptr || *candidateCapacityPtr <= 0,
              OPS_LOG_E(nodeName, "candidate_capacity must be greater than 0."),
              return ge::GRAPH_FAILED);
    uint32_t candidateCapacity = static_cast<uint32_t>(*candidateCapacityPtr);
    const bool* applyTemperaturePtr =
        attrs->GetAttrPointer<bool>(ATTR_APPLY_TEMPERATURE_INDEX);
    bool applyTemperature = applyTemperaturePtr == nullptr ? true : *applyTemperaturePtr;

    auto tempStorageShape = temperatureShape->GetStorageShape();
    uint32_t requestCount = tempStorageShape.GetDimNum() == 0
                                ? 1U
                                : static_cast<uint32_t>(tempStorageShape.GetDim(0));

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }
    uint32_t usedCoreNum = std::min(coreNum, batchSize);
    usedCoreNum = std::max(usedCoreNum, 1U);

    uint32_t dtypeBytes = DtypeBytes(context->GetInputDesc(SORTED_VALUE_INDEX)->GetDataType());
    uint32_t dataPerBlock = 32U / dtypeBytes;
    uint32_t tileSizeAligned = (GUMBEL_SAMPLE_TILE_SIZE + dataPerBlock - 1U) / dataPerBlock * dataPerBlock;

    tilingData->batchSize = batchSize;
    tilingData->vocabSize = vocabSize;
    tilingData->candidateCapacity = candidateCapacity;
    tilingData->requestCount = requestCount;
    tilingData->tileSize = GUMBEL_SAMPLE_TILE_SIZE;
    tilingData->tileSizeAligned = tileSizeAligned;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->logitsDtypeBytes = dtypeBytes;
    tilingData->applyTemperature = applyTemperature ? 1U : 0U;
    tilingData->hasTopP = OptionalInputPresent(context, P_INDEX) ? 1U : 0U;
    tilingData->hasTopK = OptionalInputPresent(context, K_INDEX) ? 1U : 0U;

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OPS_CHECK(workspaceSizes == nullptr,
              OPS_LOG_E(nodeName, "workspaceSizes is nullptr."), return ge::GRAPH_FAILED);
    workspaceSizes[0] = 0;
    context->SetBlockDim(usedCoreNum);
    OPS_LOG_D(nodeName, "BuildTopKTopPCandidates B=%u V=%u C=%u cores=%u topP=%u topK=%u",
              batchSize, vocabSize, candidateCapacity, usedCoreNum,
              tilingData->hasTopP, tilingData->hasTopK);
    return ge::GRAPH_SUCCESS;
}

struct BuildTopKTopPCandidatesCompileInfo {};

static ge::graphStatus TilingParseForBuildTopKTopPCandidates(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BuildTopKTopPCandidates)
    .Tiling(TilingForBuildTopKTopPCandidates)
    .TilingParse<BuildTopKTopPCandidatesCompileInfo>(TilingParseForBuildTopKTopPCandidates);

}  // namespace optiling
