/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include <algorithm>
#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../sampling_common/gumbel_sampling_tiling_data.h"

using namespace ge;
using vllm_ascend::sampling::GUMBEL_SAMPLE_COMPACT_CANDIDATES;
using vllm_ascend::sampling::GUMBEL_SAMPLE_SYNC_WORKSPACE_BYTES;
using vllm_ascend::sampling::GUMBEL_SAMPLE_SYS_WORKSPACE_BYTES;
using vllm_ascend::sampling::GUMBEL_SAMPLE_TILE_SIZE;
using vllm_ascend::sampling::GumbelSampleTilingData;

namespace {
constexpr uint32_t LOGITS_INDEX = 0;
constexpr uint32_t SEEDS_INDEX = 4;
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

static ge::graphStatus TilingForGumbelSampleFromCandidates(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    auto* tilingData = context->GetTilingData<GumbelSampleTilingData>();
    OPS_CHECK(tilingData == nullptr,
              OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    const gert::StorageShape* logitsShape = context->GetInputShape(LOGITS_INDEX);
    const gert::StorageShape* seedsShape = context->GetInputShape(SEEDS_INDEX);
    OPS_CHECK(logitsShape == nullptr,
              OPS_LOG_E(nodeName, "candidate logits shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_CHECK(seedsShape == nullptr,
              OPS_LOG_E(nodeName, "seeds shape is nullptr."), return ge::GRAPH_FAILED);

    auto logitsOriginShape = logitsShape->GetOriginShape();
    OPS_CHECK(logitsOriginShape.GetDimNum() != 2,
              OPS_LOG_E(nodeName, "candidate logits must be 2D."), return ge::GRAPH_FAILED);
    uint32_t batchSize = static_cast<uint32_t>(logitsOriginShape.GetDim(0));
    uint32_t candidateCapacity = static_cast<uint32_t>(logitsOriginShape.GetDim(1));
    OPS_CHECK(batchSize == 0 || candidateCapacity == 0,
              OPS_LOG_E(nodeName, "batch and candidate capacity must be greater than 0."),
              return ge::GRAPH_FAILED);

    auto seedsOriginShape = seedsShape->GetOriginShape();
    uint32_t requestCount = seedsOriginShape.GetDimNum() == 0
                                ? 1U
                                : static_cast<uint32_t>(seedsOriginShape.GetDim(0));

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }
    // 910B AIV block ids may be sparse in this launch mode. Keep one work core
    // until the kernel has a dense logical-rank mapping.
    (void)coreNum;

    uint32_t dtypeBytes = DtypeBytes(context->GetInputDesc(LOGITS_INDEX)->GetDataType());
    uint32_t dataPerBlock = 32U / dtypeBytes;
    uint32_t tileSizeAligned = (GUMBEL_SAMPLE_TILE_SIZE + dataPerBlock - 1U) / dataPerBlock * dataPerBlock;
    uint32_t numTiles = (candidateCapacity + GUMBEL_SAMPLE_TILE_SIZE - 1U) / GUMBEL_SAMPLE_TILE_SIZE;
    uint32_t usedCoreNum = 1U;

    uint64_t tileCount = static_cast<uint64_t>(batchSize) * numTiles;
    uint64_t offset = Align32(GUMBEL_SAMPLE_SYNC_WORKSPACE_BYTES);
    uint32_t tileMaxOffset = static_cast<uint32_t>(offset);
    offset += Align32(tileCount * FLOAT_BYTES);
    uint32_t rowMaxOffset = static_cast<uint32_t>(offset);
    offset += Align32(static_cast<uint64_t>(batchSize) * FLOAT_BYTES);
    uint32_t bestScoreOffset = static_cast<uint32_t>(offset);
    offset += Align32(tileCount * FLOAT_BYTES);
    uint32_t bestIdOffset = static_cast<uint32_t>(offset);
    offset += Align32(tileCount * sizeof(int32_t));

    tilingData->batchSize = batchSize;
    tilingData->vocabSize = candidateCapacity;
    tilingData->candidateCapacity = candidateCapacity;
    tilingData->requestCount = requestCount;
    tilingData->tileSize = GUMBEL_SAMPLE_TILE_SIZE;
    tilingData->tileSizeAligned = tileSizeAligned;
    tilingData->numTiles = numTiles;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->logitsDtypeBytes = dtypeBytes;
    tilingData->applyTemperature = 0;
    tilingData->sampleMode = GUMBEL_SAMPLE_COMPACT_CANDIDATES;
    tilingData->workspaceTileMaxOffset = tileMaxOffset;
    tilingData->workspaceRowMaxOffset = rowMaxOffset;
    tilingData->workspaceBestScoreOffset = bestScoreOffset;
    tilingData->workspaceBestIdOffset = bestIdOffset;
    tilingData->workspaceBytes = static_cast<uint32_t>(offset);

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OPS_CHECK(workspaceSizes == nullptr,
              OPS_LOG_E(nodeName, "workspaceSizes is nullptr."), return ge::GRAPH_FAILED);
    workspaceSizes[0] = GUMBEL_SAMPLE_SYS_WORKSPACE_BYTES + offset;
    uint32_t blockDim = platform.CalcTschBlockDim(usedCoreNum, 0, usedCoreNum);
    context->SetBlockDim(blockDim);
    OPS_LOG_D(nodeName, "GumbelSampleFromCandidates B=%u C=%u tiles=%u cores=%u blockDim=%u workspace=%lu",
              batchSize, candidateCapacity, numTiles, usedCoreNum, blockDim, workspaceSizes[0]);
    return ge::GRAPH_SUCCESS;
}

struct GumbelSampleFromCandidatesCompileInfo {};

static ge::graphStatus TilingParseForGumbelSampleFromCandidates(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GumbelSampleFromCandidates)
    .Tiling(TilingForGumbelSampleFromCandidates)
    .TilingParse<GumbelSampleFromCandidatesCompileInfo>(TilingParseForGumbelSampleFromCandidates);

}  // namespace optiling
