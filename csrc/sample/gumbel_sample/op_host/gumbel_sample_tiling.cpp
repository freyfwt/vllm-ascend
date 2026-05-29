/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include <algorithm>
#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../sampling_common/gumbel_sampling_tiling_data.h"

using namespace ge;
using vllm_ascend::sampling::GUMBEL_SAMPLE_FULL_VOCAB;
using vllm_ascend::sampling::GUMBEL_SAMPLE_SYS_WORKSPACE_BYTES;
using vllm_ascend::sampling::GUMBEL_SAMPLE_TILE_SIZE;
using vllm_ascend::sampling::GumbelSampleTilingData;

namespace {
constexpr uint32_t LOGITS_INDEX = 0;
constexpr uint32_t TEMPERATURE_INDEX = 2;
constexpr uint32_t FLOAT_BYTES = 4;

uint64_t Align32(uint64_t value)
{
    return (value + 31UL) / 32UL * 32UL;
}

uint32_t DtypeBytes(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT ? 4U : 2U;
}
}  // namespace

namespace optiling {

static ge::graphStatus TilingForGumbelSample(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto* tilingData = context->GetTilingData<GumbelSampleTilingData>();
    OPS_CHECK(tilingData == nullptr,
              OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    const gert::StorageShape* logitsShape = context->GetInputShape(LOGITS_INDEX);
    const gert::StorageShape* temperatureShape = context->GetInputShape(TEMPERATURE_INDEX);
    OPS_CHECK(logitsShape == nullptr,
              OPS_LOG_E(nodeName, "logits shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_CHECK(temperatureShape == nullptr,
              OPS_LOG_E(nodeName, "temperature shape is nullptr."), return ge::GRAPH_FAILED);

    auto logitsStorageShape = logitsShape->GetStorageShape();
    OPS_CHECK(logitsStorageShape.GetDimNum() != 2,
              OPS_LOG_E(nodeName, "logits must be 2D."), return ge::GRAPH_FAILED);
    uint32_t batchSize = static_cast<uint32_t>(logitsStorageShape.GetDim(0));
    uint32_t vocabSize = static_cast<uint32_t>(logitsStorageShape.GetDim(1));
    OPS_CHECK(batchSize == 0 || vocabSize == 0,
              OPS_LOG_E(nodeName, "batch and vocab must be greater than 0."), return ge::GRAPH_FAILED);

    auto tempStorageShape = temperatureShape->GetStorageShape();
    uint32_t requestCount = tempStorageShape.GetDimNum() == 0
                                ? 1U
                                : static_cast<uint32_t>(tempStorageShape.GetDim(0));

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }

    uint32_t dtypeBytes = DtypeBytes(context->GetInputDesc(LOGITS_INDEX)->GetDataType());
    uint32_t dataPerBlock = 32U / dtypeBytes;
    uint32_t tileSizeAligned = (GUMBEL_SAMPLE_TILE_SIZE + dataPerBlock - 1U) / dataPerBlock * dataPerBlock;
    uint32_t numTiles = (vocabSize + GUMBEL_SAMPLE_TILE_SIZE - 1U) / GUMBEL_SAMPLE_TILE_SIZE;
    uint32_t totalTiles = batchSize * numTiles;
    uint32_t usedCoreNum = std::min(coreNum, totalTiles);
    usedCoreNum = std::max(usedCoreNum, 1U);

    uint64_t tileCount = static_cast<uint64_t>(batchSize) * numTiles;
    uint64_t offset = 0;
    uint32_t tileMaxOffset = static_cast<uint32_t>(offset);
    offset += Align32(tileCount * FLOAT_BYTES);
    uint32_t rowMaxOffset = static_cast<uint32_t>(offset);
    offset += Align32(static_cast<uint64_t>(batchSize) * FLOAT_BYTES);
    uint32_t bestScoreOffset = static_cast<uint32_t>(offset);
    offset += Align32(tileCount * FLOAT_BYTES);
    uint32_t bestIdOffset = static_cast<uint32_t>(offset);
    offset += Align32(tileCount * sizeof(int32_t));

    tilingData->batchSize = batchSize;
    tilingData->vocabSize = vocabSize;
    tilingData->candidateCapacity = vocabSize;
    tilingData->requestCount = requestCount;
    tilingData->tileSize = GUMBEL_SAMPLE_TILE_SIZE;
    tilingData->tileSizeAligned = tileSizeAligned;
    tilingData->numTiles = numTiles;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->logitsDtypeBytes = dtypeBytes;
    tilingData->applyTemperature = 1;
    tilingData->sampleMode = GUMBEL_SAMPLE_FULL_VOCAB;
    tilingData->workspaceTileMaxOffset = tileMaxOffset;
    tilingData->workspaceRowMaxOffset = rowMaxOffset;
    tilingData->workspaceBestScoreOffset = bestScoreOffset;
    tilingData->workspaceBestIdOffset = bestIdOffset;
    tilingData->workspaceBytes = static_cast<uint32_t>(offset);

    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool* applyTemperature = attrs->GetAttrPointer<bool>(0);
        if (applyTemperature != nullptr) {
            tilingData->applyTemperature = *applyTemperature ? 1U : 0U;
        }
    }

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OPS_CHECK(workspaceSizes == nullptr,
              OPS_LOG_E(nodeName, "workspaceSizes is nullptr."), return ge::GRAPH_FAILED);
    workspaceSizes[0] = GUMBEL_SAMPLE_SYS_WORKSPACE_BYTES + offset;
    context->SetBlockDim(usedCoreNum);
    OPS_LOG_D(nodeName, "GumbelSample B=%u V=%u tiles=%u cores=%u workspace=%lu",
              batchSize, vocabSize, numTiles, usedCoreNum, workspaceSizes[0]);
    return ge::GRAPH_SUCCESS;
}

struct GumbelSampleCompileInfo {};

static ge::graphStatus TilingParseForGumbelSample(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GumbelSample)
    .Tiling(TilingForGumbelSample)
    .TilingParse<GumbelSampleCompileInfo>(TilingParseForGumbelSample);

}  // namespace optiling
