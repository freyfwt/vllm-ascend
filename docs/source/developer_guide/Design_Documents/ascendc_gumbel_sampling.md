# AscendC Fused Gumbel Sampling Design

## 1. Background

v1 sampling optimization currently routes normal decode rows through
`vllm_ascend.worker.v2.sample.gumbel.gumbel_sample`. The existing
implementation is a Triton-Ascend kernel plus a PyTorch-side second-stage
reduction:

1. one Triton program per `(batch_row, vocab_tile)`;
2. load a tile of logits;
3. optionally divide by temperature;
4. generate Gumbel noise with two logarithms;
5. write `local_argmax` and `local_max`;
6. call `local_max.argmax()` and `local_argmax.gather()` from Python.

The local benchmark is `tests/ut/sample/test_gumbel_sample_perf.py`. It
compares the current Triton Gumbel path with the existing
`softmax -> exponential_ -> div -> argmax` reference used as a performance
golden. The golden path is not the desired implementation route; it is only a
same-distribution performance reference.

Target-shape benchmark result with vocab size `152064`:

| Batch size | Golden ms | Current `gumbel_sample` ms | Speedup vs golden |
|---:|---:|---:|---:|
| 1 | 0.515 | 1.446 | 0.356x |
| 8 | 0.491 | 9.748 | 0.050x |
| 32 | 0.728 | 38.571 | 0.019x |
| 128 | 2.643 | 152.749 | 0.017x |

The measured single-op performance is far slower than the current
`softmax -> exponential_ -> div -> argmax` reference, especially as batch size
increases. Parameter tuning on the Triton kernel does not change the conclusion:
the bottleneck is the operator shape, not the block size. The failed tuning
attempts included disabling temperature, changing Triton `BLOCK_SIZE` between
`2048` and `4096`, disabling multibuffer, and trying `BLOCK_SIZE=8192`.
`BLOCK_SIZE=8192` exceeded UB capacity during compilation. UB means the Ascend
AI Core Unified Buffer, the on-core scratchpad used by AscendC kernels.

This document designs custom AscendC fused Gumbel sampling operators. The
goal is to make the sampling path NPU-friendly in its final shape: full-vocab
sampling and compact-candidate top-k/top-p sampling both run through fused
AscendC custom ops with device-side RNG, local reduction, and row-level final
selection inside the kernel boundary.

RNG means random number generation. The design uses stateless counter-based RNG
so the sampled token is reproducible from request seed, logical position, and
global token ID without updating CPU-side random state in the hot path.

## 2. Goals

The PR must deliver the final target architecture, not only a temporary
full-vocab fast path. It can be implemented in internal phases, but the merged
design must contain both execution modes:

1. **Full-vocab mode**: sample from `[num_rows, vocab_size]` logits when no
   compact candidate set is available.
2. **Compact-candidate mode**: sample from candidate logits/IDs produced by
   top-k/top-p filtering so sampling no longer scans the whole vocabulary for
   `top_k=50` and similar workloads.

The implementation must provide:

- input logits of shape `[num_rows, vocab_size]`;
- compact candidate inputs of shape `[num_rows, candidate_capacity]`;
- `float16`, `bfloat16`, and `float32` logits;
- per-request `temperature`, `seed`, and per-row `pos` tensors on device;
- full vocabulary sampling with no top-k/top-p compaction;
- compact top-k/top-p candidate Gumbel sampling with token-ID gather inside the
  sampling op;
- greedy rows where `temperature == 0`;
- output token IDs as `int32`;
- custom ACLNN ops exposed through `torch.ops._C_ascend`.

The performance target is to beat both existing baselines:

- faster than the existing exponential-race reference for full-vocab sampling;
- much faster than the current Triton `gumbel_sample` for batch sizes `1`, `8`,
  `32`, and `128` with vocab size `152064`;
- for rows whose exact filtered set fits in the chosen compact
  `candidate_capacity`, candidate Gumbel sampling time should scale with that
  candidate count, not `vocab_size`; `top_k=50` should not launch a full-vocab
  sampling scan.

The final design must include temperature support without a separate
temperature kernel, graph capture meta implementations, and deterministic RNG
without host-side seed updates.

### 2.1 Performance Model and Acceptance Targets

The exponential-race reference is a performance baseline, not a theoretical
lower bound. It uses highly optimized framework kernels, but it still pays for
full-vocab materialization and multiple full-vocab passes:

1. softmax reads logits, reduces row max, computes exponentials and row sums,
   and writes normalized probabilities;
2. `exponential_` generates a full `[num_rows, vocab_size]` random tensor;
3. division reads probabilities and random values and writes an intermediate
   score tensor;
4. argmax reads the score tensor and reduces to one token per row.

The fused AscendC full-vocab sampler has a lower memory-traffic target:

1. read logits for row max only when the probability-race formula is selected;
2. read logits for score computation and reduction;
3. keep tile-local scores and token IDs in UB;
4. write only `[num_rows, num_tiles]` partial `float32` scores and `int32`
   token IDs, plus the final `[num_rows]` result.

Therefore the expected full-vocab gain should come from less HBM traffic, fewer
kernel launches, no full-vocab random/probability/score tensors, no Python-side
second-stage `argmax`/`gather`, and `int32` token workspaces instead of `int64`.
The fused op adds RNG and transcendental work, so the target is not an order of
magnitude faster than the optimized exponential-race reference for full vocab.
It must still be faster than that reference; otherwise the custom op is not
worth enabling as the default implementation.

Use the benchmark numbers above as the acceptance table for vocab size
`152064` and `temperature=1`:

| Batch size | Golden ms | Required fused full-vocab ms | Required speedup |
|---:|---:|---:|---:|
| 1 | 0.515 | <= 0.500 | >= 1.03x |
| 8 | 0.491 | <= 0.440 | >= 1.12x |
| 32 | 0.728 | <= 0.580 | >= 1.25x |
| 128 | 2.643 | <= 2.110 | >= 1.25x |

The small-batch target is intentionally closer to golden because launch and
row-reduction synchronization dominate. The larger-batch target is stricter
because memory traffic and full-vocab intermediate tensors dominate, which is
where fusion should pay off.

Compact mode has a separate acceptance target:

- `npu_gumbel_sample_from_candidates` must not read full-vocab logits;
- for `top_k=50`, candidate sampling should be launch/reduction dominated and
  no more than `10%` of the fused full-vocab sampling time for the same batch;
- the end-to-end compact path, including
  `npu_build_top_k_top_p_candidates`, must beat the existing
  `npu_apply_top_k_top_p` plus full-vocab sampling path for compactable rows;
- top-k/top-p rows whose exact candidate set exceeds `candidate_capacity` must
  fall back for correctness and are excluded from the compact-speed target.

If the implementation only beats the current Triton `gumbel_sample` but is
slower than the exponential-race reference, it is useful profiling data but not
the default PR acceptance target.

## 3. Non-Goals

The following are intentionally out of scope:

- logprob output or processed logits output;
- speculative expanded rows beyond the current identity request mapping use
  case, unless the existing `idx_mapping` and `pos` tensors already describe
  them correctly;
- multi-token sampling per request;
- exact bitwise equivalence with vLLM GPU Philox/Triton random draws.

Mixed batches where some rows use top-k/top-p and others do not should be
handled by row partitioning before sampling: compactable rows use candidate
sampling, unfiltered rows use full-vocab sampling, and overflow top-p rows use
the full-vocab masked fallback. Do not truncate an unfiltered row to
`candidate_capacity`; that changes the sampling distribution. Also do not
treat `candidate_capacity` as a vLLM API limit. Large `top_k` values remain
valid; they simply bypass compact sampling when they do not fit the compact
buffer.

The required semantic contract is Gumbel-max distribution equivalence for each
row and deterministic replay for the same
`(request_seed, logical_position, global_token_id)` counters. Batch row index,
tile index, compact-candidate column, and core scheduling must not enter the
random counter.

## 4. Existing Engineering Pattern

Use the same custom ACLNN path as the existing custom ops:

```text
Python
  -> torch.ops._C_ascend.npu_gumbel_sample
  -> csrc/torch_binding.cpp torch adapter
  -> EXEC_NPU_CMD(aclnnGumbelSample, ...)
  -> custom op API / l0op
  -> op_host shape, dtype, tiling, workspace
  -> op_kernel AscendC implementation
  -> packaged into vllm_ascend/_cann_ops_custom
```

Do not implement the first version as a direct `vllm_ascend_kernels` launch.
The custom ACLNN route is already packaged by `csrc/build_aclnn.sh` and is
visible through `torch.ops._C_ascend`.

Place all new sampling operators under `csrc/sample`, not `csrc/moe`. The
existing `npu_apply_top_k_top_p` custom op lives under `csrc/moe` for
historical reasons, but this design must not add new sampling code there.

## 5. Mathematical Formulation

Gumbel sampling can be implemented with an equivalent exponential race:

```text
token = argmax_i(logit_i / temperature - log(E_i))
E_i ~ Exp(1)
```

This is distribution-equivalent to Gumbel-max because:

```text
G_i = -log(E_i)
token = argmax_i(logit_i / temperature + G_i)
```

For an implementable AscendC first version, use a stable probability-race form:

```text
scaled_i = logit_i / temperature
row_max = max_i(scaled_i)
score_i = exp(scaled_i - row_max) / E_i
E_i = -log(U_i), U_i in (0, 1)
token = argmax_i(score_i)
```

This is exactly equivalent because `exp(-row_max)` is a row-constant factor:

```text
argmax_i(exp(scaled_i - row_max) / E_i)
  = argmax_i(scaled_i - row_max - log(E_i))
  = argmax_i(scaled_i - log(E_i))
```

This form avoids computing `log(E_i)` in the kernel. With a counter-based
uniform RNG, each sampled token needs one `Log`, one `Exp`, one `Div`, and a
reduction. It also avoids materializing probabilities, noise, local argmax
tensors, or Python-side second-stage reductions.

For `temperature == 0`, the row is greedy:

```text
token = argmax_i(logit_i)
```

No RNG, `Exp`, `Log`, or `Div` should be executed for greedy rows.

## 6. Public Operator Interface

### 6.1 Torch Schemas

Add these schemas in `csrc/torch_binding.cpp`.

Full-vocab Gumbel sampler:

```cpp
ops.def(
    "npu_gumbel_sample(Tensor logits, "
    "                       Tensor idx_mapping, "
    "                       Tensor temperature, "
    "                       Tensor seeds, "
    "                       Tensor positions, "
    "                       bool apply_temperature=True) -> Tensor");
ops.impl("npu_gumbel_sample",
         torch::kPrivateUse1,
         &vllm_ascend::npu_gumbel_sample);
```

Compact-candidate Gumbel sampler:

```cpp
ops.def(
    "npu_gumbel_sample_from_candidates("
    "    Tensor candidate_logits, "
    "    Tensor candidate_ids, "
    "    Tensor candidate_lens, "
    "    Tensor idx_mapping, "
    "    Tensor seeds, "
    "    Tensor positions) -> Tensor");
ops.impl("npu_gumbel_sample_from_candidates",
         torch::kPrivateUse1,
         &vllm_ascend::npu_gumbel_sample_from_candidates);
```

Compact top-k/top-p candidate builder:

```cpp
ops.def(
    "npu_build_top_k_top_p_candidates("
    "    Tensor logits, "
    "    Tensor idx_mapping, "
    "    Tensor temperature, "
    "    Tensor? p=None, "
    "    Tensor? k=None, "
    "    int candidate_capacity, "
    "    bool apply_temperature=True) -> "
    "    (Tensor candidate_logits, Tensor candidate_ids, "
    "     Tensor candidate_lens, Tensor candidate_status)");
ops.impl("npu_build_top_k_top_p_candidates",
         torch::kPrivateUse1,
         &vllm_ascend::npu_build_top_k_top_p_candidates);
```

Sampler return dtype is `int32`, shape is `[num_rows]`.

The candidate builder is part of the final target, not a separate follow-up
idea. It may internally reuse or refactor the current
`npu_apply_top_k_top_p` sorting/filtering code, but its public contract must
emit compact candidate tensors rather than masked full-vocab logits.

This is intentionally different from the existing `npu_apply_top_k_top_p`
operator. The existing operator returns a full-vocab tensor with filtered
positions set to `-inf`, so the following sampler still scans the whole
vocabulary. `npu_build_top_k_top_p_candidates` must instead return compact
candidate logits plus their original token IDs. That compact output is what
allows `top_k=50` and similar rows to sample over tens of candidates rather
than `152064` vocabulary entries.

`candidate_capacity` is an internal compact-buffer capacity, not the legal
upper bound of vLLM's `top_k` serving parameter. The service-facing sampling
API must continue accepting every `top_k` value that vLLM accepts. If a row's
exact top-k/top-p candidate set does not fit in `candidate_capacity`, the
builder reports `candidate_status[row] = 2` and the adapter routes that row to
the exact full-vocab fallback. The compact fast path is an optimization; it
must never change API validity.

Temperature belongs in the candidate builder, not in
`npu_gumbel_sample_from_candidates`. For compact mode, the default path passes
raw logits and `apply_temperature=True`; the builder applies per-request
temperature before top-k/top-p filtering and before writing `candidate_logits`.
This preserves the existing semantic order: temperature first, top-k/top-p
second, Gumbel sampling last. Applying temperature inside the candidate sampler
would be wrong for top-p because the candidate set itself depends on the
temperature-scaled softmax distribution.

`apply_temperature=False` is allowed only when the caller has already applied
the same temperature transform to the input logits and has not applied
top-k/top-p filtering yet. Even with `apply_temperature=False`, the builder must
still read `temperature` so rows with `temperature == 0` short-circuit to
greedy. In other words, the flag skips division for non-greedy rows; it does
not disable greedy detection.

### 6.2 ACLNN API

The generated or hand-written ACLNN full-vocab API should be:

```cpp
aclnnStatus aclnnGumbelSampleGetWorkspaceSize(
    const aclTensor* logits,
    const aclTensor* idxMapping,
    const aclTensor* temperature,
    const aclTensor* seeds,
    const aclTensor* positions,
    bool applyTemperature,
    aclTensor* sampledTokenIds,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnGumbelSample(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

The l0 operator should allocate only `sampledTokenIds` and add one AICore launch:

```cpp
const aclTensor* GumbelSample(
    const aclTensor* logits,
    const aclTensor* idxMapping,
    const aclTensor* temperature,
    const aclTensor* seeds,
    const aclTensor* positions,
    bool applyTemperature,
    aclOpExecutor* executor)
{
    auto output = executor->AllocTensor(
        {logits->GetViewShape().GetDim(0)},
        op::DataType::DT_INT32);

    ADD_TO_LAUNCHER_LIST_AICORE(
        GumbelSample,
        OP_INPUT(logits, idxMapping, temperature, seeds, positions),
        OP_OUTPUT(output),
        OP_ATTR(applyTemperature));

    return output;
}
```

Use existing l0op implementations as concrete templates. For example,
`csrc/moe/apply_top_k_top_p_custom/op_host/op_api/apply_top_k_top_p_custom.cpp`
shows the local ACLNN pattern, but new sampling files still belong under
`csrc/sample`.

The compact-candidate ACLNN API mirrors the full-vocab API and adds candidate
IDs and valid lengths:

```cpp
aclnnStatus aclnnGumbelSampleFromCandidatesGetWorkspaceSize(
    const aclTensor* candidateLogits,
    const aclTensor* candidateIds,
    const aclTensor* candidateLens,
    const aclTensor* idxMapping,
    const aclTensor* seeds,
    const aclTensor* positions,
    aclTensor* sampledTokenIds,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

The candidate builder API returns four tensors:

```cpp
aclnnStatus aclnnBuildTopKTopPCandidatesGetWorkspaceSize(
    const aclTensor* logits,
    const aclTensor* idxMapping,
    const aclTensor* temperature,
    const aclTensor* p,
    const aclTensor* k,
    int64_t candidateCapacity,
    bool applyTemperature,
    aclTensor* candidateLogits,
    aclTensor* candidateIds,
    aclTensor* candidateLens,
    aclTensor* candidateStatus,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

`candidate_logits` and `candidate_ids` are dense `[B, candidateCapacity]`
outputs. `candidate_logits` are already temperature-scaled when
`temperature[req] > 0`; for greedy rows where `temperature[req] == 0`, the
builder emits a single max-logit candidate. `candidate_lens[row]` tells the
sampler how many entries are valid.
`candidate_status[row] == 0` means the compact row is exact and can be passed
to the candidate Gumbel sampler. Non-zero status means the row must use the
full-vocab fallback for correctness, for example because a top-p set exceeded
`candidateCapacity`, a top-k value is larger than `candidateCapacity`, or the
row has no active top-k/top-p constraint. Invalid candidate slots must contain
`-inf` logits and any valid int32 ID; the sampler masks them by length.

### 6.3 Python Wrapper

Update `vllm_ascend/worker/v2/sample/gumbel.py` so the public
`gumbel_sample()` keeps the same call surface and dispatches to the full-vocab
fused op when possible:

```python
def _can_use_gumbel_sample_op(
    logits: torch.Tensor,
    output_processed_logits: torch.Tensor | None,
    use_fp64: bool,
) -> bool:
    return (
        not use_fp64
        and output_processed_logits is None
        and hasattr(torch.ops._C_ascend, "npu_gumbel_sample")
        and logits.dim() == 2
        and logits.is_npu
        and logits.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def gumbel_sample(...):
    if _can_use_gumbel_sample_op(logits, output_processed_logits, use_fp64):
        return torch.ops._C_ascend.npu_gumbel_sample(
            logits,
            idx_mapping.to(torch.int32),
            temperature,
            seed,
            pos,
            apply_temperature,
        )

    # Existing Triton fallback remains for unsupported surfaces.
```

The fast path intentionally does not support `output_processed_logits` in the
initial integration. If logprob modes need processed logits, keep the existing
fallback or add a separate processed-logits output path inside the same PR only
after the no-logprobs path meets the performance target.

## 7. Shape, Dtype, and Format Rules

The op host validation must enforce the following.

Full-vocab sampler inputs:

| Tensor | Shape | Dtype | Format | Notes |
|---|---:|---|---|---|
| `logits` | `[B, V]` | `float16`, `bfloat16`, `float32` | `ND` | contiguous or auto-contiguous |
| `idxMapping` | `[B]` | `int32` | `ND` | row to request index |
| `temperature` | `[R]` | `float32` | `ND` | indexed by `idxMapping[row]` |
| `seeds` | `[R]` | `int64` | `ND` | indexed by request |
| `positions` | `[B]` | `int64` or `int32` | `ND` | indexed by row |

Compact-candidate Gumbel sampler inputs:

| Tensor | Shape | Dtype | Format | Notes |
|---|---:|---|---|---|
| `candidate_logits` | `[B, Cmax]` | `float16`, `bfloat16`, `float32` | `ND` | compact logits already processed by temperature and top-k/top-p |
| `candidate_ids` | `[B, Cmax]` | `int32` | `ND` | original/global token IDs |
| `candidate_lens` | `[B]` | `int32` | `ND` | valid candidate count per row |
| `idxMapping` | `[B]` | `int32` | `ND` | row to request index |
| `seeds` | `[R]` | `int64` | `ND` | indexed by request |
| `positions` | `[B]` | `int64` or `int32` | `ND` | indexed by row |

Candidate builder inputs:

| Tensor | Shape | Dtype | Format | Notes |
|---|---:|---|---|---|
| `logits` | `[B, V]` | `float16`, `bfloat16`, `float32` | `ND` | source logits |
| `idxMapping` | `[B]` | `int32` | `ND` | row to request index for temperature |
| `temperature` | `[R]` | `float32` | `ND` | applied before top-k/top-p compaction |
| `p` | `[B]` or empty optional | same as logits or `float32` | `ND` | top-p threshold |
| `k` | `[B]` or empty optional | `int32` | `ND` | top-k threshold |
| `candidateCapacity` | scalar attr | `int64` | n/a | compact output capacity, not a top-k API limit |
| `applyTemperature` | scalar attr | `bool` | n/a | divide non-greedy logits by temperature when true |

Sampler output:

| Tensor | Shape | Dtype | Format |
|---|---:|---|---|
| `sampledTokenIds` | `[B]` | `int32` | `ND` |

Candidate builder outputs:

| Tensor | Shape | Dtype | Format |
|---|---:|---|---|
| `candidate_logits` | `[B, candidateCapacity]` | same as logits | `ND` |
| `candidate_ids` | `[B, candidateCapacity]` | `int32` | `ND` |
| `candidate_lens` | `[B]` | `int32` | `ND` |
| `candidate_status` | `[B]` | `int32` | `ND` |

Validation details:

- `B > 0` and `V > 0`;
- `idxMapping.shape[0] == B` for every op that receives `idxMapping`;
- `positions.shape[0] == B`;
- for full-vocab sampling, `temperature.shape[0] == seeds.shape[0]`;
- for compact candidate sampling, `seeds.shape[0]` must cover every
  `idxMapping[row]`;
- for candidate building, `temperature.shape[0]` must cover every
  `idxMapping[row]`;
- `0 <= idxMapping[row] < temperature.shape[0]` for ops that receive
  `temperature`;
- `0 <= idxMapping[row] < seeds.shape[0]` for sampler ops that receive
  `seeds`;
- `1 <= candidate_lens[row] <= Cmax` for candidate Gumbel sampling;
- every candidate ID must be in `[0, vocab_size)` when the builder knows the
  global vocab size;
- all tensors must be on NPU;
- `logits.stride(-1) == 1` after the ACLNN contiguous step.

The C++ torch adapter should use `TORCH_CHECK` for cheap user-facing validation,
while op host must repeat validation because graph/eager calls can reach ACLNN
without the Python helper.

## 8. Tiling Design

### 8.1 Tiling Data

Create `csrc/sample/gumbel_sample/op_host/gumbel_sample_tiling.h` for the
full-vocab sampler. Mirror the same field set in
`csrc/sample/gumbel_sample_from_candidates/op_host/`
`gumbel_sample_from_candidates_tiling.h` for compact-candidate sampling:

```cpp
namespace optiling {

BEGIN_TILING_DATA_DEF(GumbelSampleTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, vocabSize);
TILING_DATA_FIELD_DEF(uint32_t, candidateCapacity);
TILING_DATA_FIELD_DEF(uint32_t, requestCount);
TILING_DATA_FIELD_DEF(uint32_t, tileSize);
TILING_DATA_FIELD_DEF(uint32_t, tileSizeAligned);
TILING_DATA_FIELD_DEF(uint32_t, numTiles);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, logitsDtypeBytes);
TILING_DATA_FIELD_DEF(uint32_t, applyTemperature);
TILING_DATA_FIELD_DEF(uint32_t, sampleMode);
TILING_DATA_FIELD_DEF(uint32_t, workspaceTileMaxOffset);
TILING_DATA_FIELD_DEF(uint32_t, workspaceRowMaxOffset);
TILING_DATA_FIELD_DEF(uint32_t, workspaceBestScoreOffset);
TILING_DATA_FIELD_DEF(uint32_t, workspaceBestIdOffset);
TILING_DATA_FIELD_DEF(uint32_t, workspaceBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GumbelSample, GumbelSampleTilingData)

struct GumbelSampleCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t ubSize = 0;
};

}  // namespace optiling
```

For `GumbelSampleFromCandidates`, use the same fields with
`GumbelSampleFromCandidatesTilingData` and register:

```cpp
REGISTER_TILING_DATA_CLASS(GumbelSampleFromCandidates,
                           GumbelSampleFromCandidatesTilingData)
```

Keeping separate CANN tiling classes avoids cross-op registration coupling.
Keep field names identical so the shared kernel helper can consume either
tiling struct.

Offsets are byte offsets from `AscendC::GetUserWorkspace(workspace)`.
`sampleMode` is `0` for full-vocab sampling and `1` for compact-candidate
sampling. In compact-candidate mode, `candidateCapacity` is `Cmax` and
`vocabSize` is retained only for validation/debug metadata if the builder
provides it.

### 8.2 Tile Size

Use a fixed first version tile size unless UB validation requires reducing it.
The same sampler core handles full-vocab and compact-candidate mode:

```text
tileSize = 1024 or 2048
tileSizeAligned = ceil_align(tileSize, 32-byte / dtype_bytes)
effectiveSize = sampleMode == full_vocab ? vocabSize : candidateCapacity
numTiles = ceil_div(effectiveSize, tileSize)
```

Recommended first value:

- `tileSize = 1024` for correctness bring-up and for `Cmax <= 1024`
  candidate Gumbel sampling;
- benchmark `2048` after the fused kernel is correct;
- do not use `8192` initially because the Triton experiment already hit UB
  pressure at that scale.

The kernel needs UB buffers for:

- logits tile cast to `float`;
- optional stable logits;
- uniform/exponential temporary;
- score tile;
- local index vector or token ID vector;
- reduction scratch.

For `float32` logits and `tileSize=1024`, this is comfortably below UB on
910B/910C even with double buffering disabled.

### 8.3 Core Assignment

Use one kernel launch with three synchronized phases:

```text
totalTiles = batchSize * numTiles
usedCoreNum = min(platformCoreNum, totalTiles)
```

Each AIV core processes tile tasks by striding:

```cpp
for (uint32_t task = coreIdx; task < totalTiles; task += usedCoreNum) {
    uint32_t row = task / numTiles;
    uint32_t tile = task % numTiles;
    ...
}
```

Then use `AscendC::SyncAll()` between phases.

This assignment keeps all rows and vocab tiles parallel without launching
`B * numTiles` blocks, and it allows the same kernel to do the final row-level
reduction.

### 8.4 Workspace Layout

For both full-vocab and compact-candidate mode:

```text
tileMax       float32 [B, numTiles]
rowMax        float32 [B]
bestScore     float32 [B, numTiles]
bestTokenId   int32   [B, numTiles]
```

In full-vocab mode, `bestTokenId` is the absolute column index. In
compact-candidate mode, `bestTokenId` is gathered from
`candidate_ids[row, best_col]` inside the sampling kernel.

Workspace bytes:

```cpp
tile_count = B * numTiles;
workspaceBytes =
    align32(tile_count * sizeof(float)) +
    align32(B * sizeof(float)) +
    align32(tile_count * sizeof(float)) +
    align32(tile_count * sizeof(int32_t));
```

For full-vocab `B=128`, `V=152064`, and `tileSize=1024`, `numTiles=149`, so
the user workspace is roughly:

```text
128 * 149 * (4 + 4 + 4) + 128 * 4 ~= 229 KiB
```

Do not allocate `int64` token IDs in workspace. Token IDs are bounded by vocab
size and should be `int32`.

For compact-candidate `Cmax <= 1024`, `numTiles=1`, so the workspace shrinks to
roughly `B * 12 + B * 4` bytes plus alignment. More importantly, Phase C scans
only candidates, not the full vocabulary.

## 9. Kernel Algorithm

Create:

```text
csrc/sample/gumbel_sample/op_kernel/gumbel_sample.cpp
csrc/sample/gumbel_sample/op_kernel/gumbel_sample.h
csrc/sample/gumbel_sample_from_candidates/op_kernel/gumbel_sample_from_candidates.cpp
csrc/sample/gumbel_sample_from_candidates/op_kernel/gumbel_sample_from_candidates.h
csrc/sample/sampling_common/counter_rng.h
csrc/sample/sampling_common/gumbel_sample_common.h
```

Both public sampler ops should share the same implementation helper. The
full-vocab entry passes `candidate_ids == nullptr` and no `candidate_lens`.
The compact-candidate entry passes `candidate_ids` and `candidate_lens`, and
uses `candidateCapacity` as the effective row width.

### 9.1 Kernel Entry

The kernel signature follows CANN custom-op convention:

```cpp
extern "C" __global__ __aicore__ void gumbel_sample(
    GM_ADDR logits,
    GM_ADDR idx_mapping,
    GM_ADDR temperature,
    GM_ADDR seeds,
    GM_ADDR positions,
    GM_ADDR sampled_token_ids,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(GumbelSampleTilingData, tilingData, tiling);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    AscendC::TPipe pipe;
    GumbelSampleKernel<DTYPE_LOGITS> op;
    op.Init(logits, idx_mapping, temperature, seeds, positions,
            sampled_token_ids, userWorkspace, &tilingData, &pipe);
    op.Process();
}
```

Add a second entry with the same helper for compact candidates:

```cpp
extern "C" __global__ __aicore__ void gumbel_sample_from_candidates(
    GM_ADDR candidate_logits,
    GM_ADDR candidate_ids,
    GM_ADDR candidate_lens,
    GM_ADDR idx_mapping,
    GM_ADDR seeds,
    GM_ADDR positions,
    GM_ADDR sampled_token_ids,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    // Same setup as gumbel_sample, but use
    // GumbelSampleFromCandidatesTilingData. Init receives candidate_ids and
    // candidate_lens, omits temperature, and sampleMode is compact_candidate.
}
```

Use `DTYPE_LOGITS` generated by the op build tool from the `logits` input.

### 9.2 Phase A: Per-Tile Row Max

For each `(row, tile)`:

1. Load the effective logits tile to UB:
   - full-vocab: `logits[row, start:start+tileSize]`;
   - compact-candidate:
     `candidate_logits[row, start:start+tileSize]`.
2. Cast to `float32`.
3. For full-vocab mode, load `req = idxMapping[row]` and
   `temp = temperature[req]`.
4. For full-vocab mode, if `applyTemperature && temp != 0 && temp != 1`,
   divide logits by `temp`.
5. For compact-candidate mode, do not apply temperature. The builder has
   already written temperature-scaled candidate logits.
6. Mask invalid elements with `-inf`:
   - full-vocab: `col < vocabSize`;
   - compact-candidate: `col < candidate_lens[row]`.
7. Reduce tile max and write `tileMax[row, tile]`.

For full-vocab `temperature == 0`, still compute max on raw logits. This
supports greedy rows using the same row-max infrastructure. For compact
candidate rows with `temperature == 0`, the builder should emit a single
candidate, so the sampler can return it without RNG.

### 9.3 Phase B: Per-Row Max

After `AscendC::SyncAll()`, reduce `tileMax[row, :]` for each row assigned to a
core:

```cpp
for (uint32_t row = coreIdx; row < batchSize; row += usedCoreNum) {
    float maxValue = -INFINITY;
    for (uint32_t tile = 0; tile < numTiles; ++tile) {
        maxValue = max(maxValue, tileMax[row * numTiles + tile]);
    }
    rowMax[row] = maxValue;
}
```

The first implementation may use scalar `GetValue` over `numTiles` because
`numTiles` is small (`149` for the target vocab with `tileSize=1024`). If this
becomes visible in profiling, replace it with a UB vector load/reduce over
`numTiles`.

### 9.4 Phase C: Per-Tile Race Score and Local Best

After another `AscendC::SyncAll()`, process each tile again:

1. Load logits and cast to `float32`.
2. Apply temperature only in full-vocab mode for non-greedy rows.
3. If full-vocab `temp == 0` or compact `candidate_lens[row] == 1`, set
   `score = logits` and skip RNG for that row.
4. Else:
   - `stable = logits - rowMax[row]`;
   - generate uniform `U` with counter RNG for each token;
   - `E = -log(max(U, eps))`;
   - `prob = exp(stable)`;
   - `score = prob / E`.
5. Mask tail elements with `-inf`.
6. Reduce the tile score to `(bestScore, bestLocalIndex)`.
7. Convert local index to output token ID and write:
   - full-vocab: `token_id = tile_start + local_index`;
   - compact-candidate:
     `token_id = candidate_ids[row, tile_start + local_index]`;
   - `bestScore[row, tile]`;
   - `bestTokenId[row, tile]`.

Tie-breaking rule:

- if scores are equal, keep the smaller token ID;
- this makes greedy rows match first-argmax behavior.

### 9.5 Phase D: Final Row Reduction

After a final `AscendC::SyncAll()`, reduce `bestScore[row, :]` and
`bestTokenId[row, :]`:

```cpp
for (uint32_t row = coreIdx; row < batchSize; row += usedCoreNum) {
    float best = -INFINITY;
    int32_t bestId = 0;
    for (uint32_t tile = 0; tile < numTiles; ++tile) {
        float candidate = bestScore[row * numTiles + tile];
        int32_t tokenId = bestTokenId[row * numTiles + tile];
        if (candidate > best || (candidate == best && tokenId < bestId)) {
            best = candidate;
            bestId = tokenId;
        }
    }
    sampledTokenIds[row] = bestId;
}
```

Again, scalar reduction over `numTiles` is acceptable for bring-up. Optimize
only if profiling shows it matters.

## 10. Counter-Based RNG

### 10.1 Determinism Contract

The random value for each `(request, position, token)` must be independent of:

- batch size;
- number of cores;
- tile size;
- scheduling order;
- whether another row in the same batch is greedy.

Use a counter key:

```text
req          = idxMapping[row]
request_seed = seeds[req]
logical_pos  = positions[row]
token_key    = global_token_id

counter_low  = lower_32_bits(token_key)
counter_high = lower_32_bits(logical_pos)
key_low      = lower_32_bits(request_seed)
key_high     = upper_32_bits(request_seed)
```

`row`, `tile`, `coreIdx`, candidate column, and compacted-row position are
explicitly forbidden in the RNG key. Full-vocab mode uses
`token_key = tile_start + local_index`. Compact-candidate mode must use
`token_key = candidate_ids[row, tile_start + local_index]`, not the local
candidate column. This makes compact sampling draw the same random number for a
token regardless of candidate order or batch partitioning.

For expanded/speculative rows, use `positions[row]` from
`V1SamplingContext.pos`. If two rows for the same request and position can be
sampled in the same call in a future path, include a logical sample slot such as
`expanded_local_pos[row]` in the Python API or encode it into `positions`
before calling the op. Do not use the physical row index for disambiguation.

The `seeds` tensor must be populated by stable request identity, not by
physical batch order. The current v1 adapter cache keyed by `req_id` is the
right shape for this requirement. Any future CPU or device-side seed path must
preserve that property; deriving a new seed from the current batch row is not
allowed.

### 10.2 First Implementation RNG

Implement a local vectorizable hash in `counter_rng.h`. The preferred first
choice is a fixed-shift Wang-style 32-bit hash because it maps well to vector
integer operations:

```cpp
// Pseudocode. Implement with AscendC vector Adds/Muls/Xor/ShiftRight.
x ^= seed_low;
x += pos_low * 0x9e3779b9u;
x = (x ^ 61u) ^ (x >> 16);
x = x * 9u;
x = x ^ (x >> 4);
x = x * 0x27d4eb2du;
x = x ^ (x >> 15);
```

Convert to `float32` uniform using the high 24 bits:

```cpp
uint32_t mantissa = (x >> 8) & 0x00ffffffu;
float u = (static_cast<float>(mantissa) + 0.5f) * 0x1.0p-24f;
```

Clamp `u` to `[2^-24, 1 - 2^-24]` before `Log`.

Implementation guidance:

- initialize an `arange` local tensor `[0, tileSize)` once per core and reuse
  it for all tile tasks;
- derive absolute token IDs with vector `Adds(arange, tileStart)`;
- avoid per-token `SetValue` in the hot loop;
- keep RNG entirely stateless; do not update `seeds` in the kernel.

If the local CANN toolkit lacks a required vector integer primitive, fall back
in this order:

1. use a CANN/AscendC stateless RNG primitive if available in the target
   toolkit;
2. implement the same hash with scalar loops only for bring-up, but do not mark
   the performance target complete;
3. temporarily support an internal debug mode that accepts pre-generated
   exponential noise, only for correctness and profiling isolation.

The release fast path must not require a host-side RNG update.

### 10.3 Statistical Validation

Add NPU tests that sample many positions with the same logits and different
`positions`, then compare empirical frequencies against softmax probabilities.

Use small shapes for the distribution test:

```text
B = 4
V = 16
positions = arange(num_trials)
num_trials >= 20000
```

Use a loose chi-square or max-error threshold because the RNG is not intended
to be bitwise identical to upstream CUDA.

## 11. Batch-Invariant Requirements

The fused sampling path must be batch-invariant when `VLLM_BATCH_INVARIANT=1`.
For a fixed request, logits, sampling parameters, seed, and logical position,
the sampled token must not change when:

- the request is batched with different neighboring requests;
- the request moves to a different physical batch row;
- rows are partitioned into compact, no-op, and overflow groups;
- tile size or AIV core scheduling changes.

Required implementation rules:

- RNG counters use only request-stable data:
  `request_seed`, `positions[row]`, and global token ID.
- `idx_mapping` always maps the current physical row to the original request
  state index. When compact rows are gathered into a smaller tensor, gather
  `candidate_logits`, `candidate_ids`, `candidate_lens`, `idx_mapping`, and
  `positions` with the same row-index tensor, and scatter sampled outputs back
  with that tensor.
- `seeds` and `temperature` remain indexed by the original request state. Do
  not renumber them after compact-row partitioning.
- The candidate builder must be deterministic for ties. For top-k and top-p,
  equal logits/probabilities must be ordered by global token ID ascending before
  choosing the retained set. This avoids candidate-set drift when the same row
  is processed in a different batch layout.
- Top-p probability calculation must use batch-invariant softmax/reduction
  behavior in batch-invariant mode. If the first implementation uses a CANN
  sort/softmax primitive whose reduction order is not proven invariant, do not
  enable the compact builder under `VLLM_BATCH_INVARIANT`.

Rollout rule:

- Full-vocab `npu_gumbel_sample` may be used in batch-invariant mode only after
  the counter-key tests pass.
- `npu_build_top_k_top_p_candidates` and
  `npu_gumbel_sample_from_candidates` may be used in batch-invariant mode only
  after candidate tie-order and top-p order tests pass.
- Until those tests exist and pass on NPU, gate the new fused path with
  `not envs.VLLM_BATCH_INVARIANT` and fall back to the existing vLLM
  batch-invariant sampler path.

## 12. Greedy and Temperature Semantics

Full-vocab sampling reads temperature by request:

```cpp
int32_t req = idxMapping[row];
float temp = temperature[req];
```

Full-vocab rules:

- `temp == 0`: greedy argmax over raw logits;
- `temp == 1` or `applyTemperature == false`: no divide;
- otherwise sample over `logits / temp`;
- reject negative temperature in op host;
- reject `NaN` temperature in op host if scalar inspection is cheap, otherwise
  document that NaN behavior is undefined and rely on Python metadata checks.

This matches vLLM's greedy semantics: a row with `temperature == 0` must return
`argmax(raw_logits)` and must not consume RNG. Top-k/top-p settings must not
change that row's selected token. They may still be present in the batch, but
the implementation must treat the row as greedy before any probability-based
top-p calculation.

The current `V1SamplerAdapter` applies top-k/top-p by calling:

```python
gumbel_ops.apply_temperature(...)
logits = apply_top_k_top_p(...)
gumbel_sample(..., apply_temperature=False)
```

The fused compact path must preserve that ordering inside
`npu_build_top_k_top_p_candidates`:

- no top-k/top-p: call `npu_gumbel_sample(..., apply_temperature=True)`;
- top-k/top-p compact path: call
  `npu_build_top_k_top_p_candidates(logits, idx_mapping, temperature, ...,
  apply_temperature=True)`;
- inside the builder, rows with `temperature == 0` short-circuit to greedy:
  write exactly one candidate, `candidate_ids[row, 0] = argmax(raw_logits[row])`,
  `candidate_lens[row] = 1`, and `candidate_status[row] = 0`;
- for non-greedy rows, the builder applies temperature first, then top-k/top-p,
  then writes compact `candidate_logits`;
- if the caller already ran `apply_temperature` and still wants to use the
  builder before top-k/top-p filtering, it may pass `apply_temperature=False`;
  the builder must still treat `temperature == 0` rows as greedy;
- call `npu_gumbel_sample_from_candidates(...)` without any temperature
  argument;
- overflow fallback rows use the existing sequence
  `apply_temperature -> apply_top_k_top_p -> npu_gumbel_sample(...,
  apply_temperature=False)`.

Do not move temperature into `npu_gumbel_sample_from_candidates`. That would
select top-p candidates from the wrong distribution and then sample from a
different one.

## 13. Build and File Layout

Place the new ops under a new `csrc/sample` directory. Do not put these files
under `csrc/moe`; the existing top-k/top-p custom op being there is historical
placement debt, not a pattern to follow.

Use one directory per CANN op so `ASCEND_OP_NAME`,
`add_op_to_compiled_list()`, generated op info, and kernel binary packaging
stay aligned with existing build conventions.

Create:

```text
csrc/sample/
  CMakeLists.txt

csrc/sample/gumbel_sample/
  CMakeLists.txt
  gumbel_sample_torch_adpt.h
  op_kernel/
    gumbel_sample.cpp
    gumbel_sample.h
  op_host/
    CMakeLists.txt
    gumbel_sample_def.cpp
    gumbel_sample_tiling.cpp
    gumbel_sample_tiling.h
    op_api/
      aclnn_gumbel_sample.cpp
      aclnn_gumbel_sample.h
      gumbel_sample.cpp
      gumbel_sample.h

csrc/sample/gumbel_sample_from_candidates/
  CMakeLists.txt
  gumbel_sample_from_candidates_torch_adpt.h
  op_kernel/
    gumbel_sample_from_candidates.cpp
    gumbel_sample_from_candidates.h
  op_host/
    CMakeLists.txt
    gumbel_sample_from_candidates_def.cpp
    gumbel_sample_from_candidates_tiling.cpp
    gumbel_sample_from_candidates_tiling.h
    op_api/
      aclnn_gumbel_sample_from_candidates.cpp
      aclnn_gumbel_sample_from_candidates.h
      gumbel_sample_from_candidates.cpp
      gumbel_sample_from_candidates.h

csrc/sample/build_top_k_top_p_candidates/
  CMakeLists.txt
  build_top_k_top_p_candidates_torch_adpt.h
  op_kernel/
    build_top_k_top_p_candidates.cpp
    build_top_k_top_p_candidates.h
  op_host/
    CMakeLists.txt
    build_top_k_top_p_candidates_def.cpp
    build_top_k_top_p_candidates_tiling.cpp
    build_top_k_top_p_candidates_tiling.h
    op_api/
      aclnn_build_top_k_top_p_candidates.cpp
      aclnn_build_top_k_top_p_candidates.h
      build_top_k_top_p_candidates.cpp
      build_top_k_top_p_candidates.h

csrc/sample/sampling_common/
  counter_rng.h
  gumbel_sample_common.h
```

`sampling_common` is header-only and is included by both sampler kernels. It
does not need its own `CMakeLists.txt`.

### 13.1 Top-Level CMake Wiring

Add `csrc/sample/CMakeLists.txt` with the same auto-discovery style used by
other operator groups:

```cmake
file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if (DEFINED ASCEND_OP_NAME AND NOT "${ASCEND_OP_NAME}" STREQUAL "")
        if (NOT "${ASCEND_OP_NAME}" STREQUAL "all"
            AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL")
            if (NOT ${SUB_DIR} IN_LIST ASCEND_OP_NAME)
                continue()
            endif()
        endif()
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/op_host/CMakeLists.txt")
        add_subdirectory(${SUB_DIR}/op_host)
    endif()
endforeach()
```

Then update `csrc/CMakeLists.txt` near the existing operator group
registrations:

```cmake
add_subdirectory(sample)
```

Place it alongside the domain directories such as `moe`, `ffn`, and
`attention`; no `OP_LIST` append is needed for these custom ACLNN ops because
each op's `op_host/CMakeLists.txt` calls `add_op_to_compiled_list()`.

Update `csrc/build_aclnn.sh::resolve_op_dir()` to check the new sample
directory before legacy locations:

```bash
for candidate_dir in \
    "${ROOT_DIR}/csrc/sample/${op_name}" \
    "${ROOT_DIR}/csrc/moe/${op_name}" \
    "${ROOT_DIR}/csrc/gmm/${op_name}" \
    ...
```

This makes `bash build.sh --pkg --ops=gumbel_sample,...` resolve the new ops
without relying on the fallback `find`.

### 13.2 Per-Op `CMakeLists.txt`

Each op directory uses the same pattern:

```cmake
file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST AND NOT BENCHMARK)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
```

### 13.3 Per-Op `op_host/CMakeLists.txt`

For `gumbel_sample`:

```cmake
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        gumbel_sample_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME GumbelSample
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE gumbel_sample ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
```

For `gumbel_sample_from_candidates`, use the same file with:

```cmake
target_sources(op_host_aclnn PRIVATE
    gumbel_sample_from_candidates_def.cpp
)

add_ops_compile_options(
    OP_NAME GumbelSampleFromCandidates
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Werror
)

add_modules_sources(OPTYPE gumbel_sample_from_candidates ACLNNTYPE aclnn)
```

For `build_top_k_top_p_candidates`, use:

```cmake
target_sources(op_host_aclnn PRIVATE
    build_top_k_top_p_candidates_def.cpp
)

add_ops_compile_options(
    OP_NAME BuildTopKTopPCandidates
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Werror
)

add_modules_sources(OPTYPE build_top_k_top_p_candidates ACLNNTYPE aclnn)
```

Use `op_host_aclnn`, not `op_host_aclnnExc`, because these ops should generate
normal public ACLNN APIs.

### 13.4 `gumbel_sample_def.cpp`

Register the op:

```cpp
namespace ops {
class GumbelSample : public OpDef {
public:
    explicit GumbelSample(const char* name) : OpDef(name)
    {
        this->Input("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("idxMapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("temperature")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seeds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("positions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("sampledTokenIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("applyTemperature").Bool();
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GumbelSample);
}  // namespace ops
```

If CANN op-build requires one dtype list entry per input/output combination,
split `positions` support into one first-version dtype (`int64`) and add
`int32` later. The Python wrapper can pass `pos.to(torch.int64)` initially.

### 13.5 `csrc/build_aclnn.sh`

Add all three ops to both custom-op arrays:

```bash
# ascend910b
CUSTOM_OPS_ARRAY=(
    ...
    "apply_top_k_top_p_custom"
    "gumbel_sample"
    "gumbel_sample_from_candidates"
    "build_top_k_top_p_candidates"
    ...
)

# ascend910_93
CUSTOM_OPS_ARRAY=(
    ...
    "apply_top_k_top_p_custom"
    "gumbel_sample"
    "gumbel_sample_from_candidates"
    "build_top_k_top_p_candidates"
    ...
)
```

Do not add it to `ascend950` until the kernel is ported and validated for that
architecture.

### 13.6 Torch Adapter

Create `csrc/sample/gumbel_sample/gumbel_sample_torch_adpt.h`:

```cpp
#ifndef GUMBEL_SAMPLE_TORCH_ADPT_H
#define GUMBEL_SAMPLE_TORCH_ADPT_H

namespace vllm_ascend {

inline at::Tensor npu_gumbel_sample(
    const at::Tensor& logits,
    const at::Tensor& idx_mapping,
    const at::Tensor& temperature,
    const at::Tensor& seeds,
    const at::Tensor& positions,
    bool apply_temperature)
{
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
    TORCH_CHECK(idx_mapping.scalar_type() == at::kInt,
                "idx_mapping must be int32");
    TORCH_CHECK(temperature.scalar_type() == at::kFloat,
                "temperature must be float32");
    TORCH_CHECK(seeds.scalar_type() == at::kLong,
                "seeds must be int64");
    TORCH_CHECK(positions.scalar_type() == at::kLong ||
                positions.scalar_type() == at::kInt,
                "positions must be int64 or int32");

    auto batch = logits.size(0);
    auto output = at::empty({batch}, logits.options().dtype(at::kInt));

    EXEC_NPU_CMD(aclnnGumbelSample,
                 logits,
                 idx_mapping,
                 temperature,
                 seeds,
                 positions,
                 apply_temperature,
                 output);
    return output;
}

}  // namespace vllm_ascend

#endif  // GUMBEL_SAMPLE_TORCH_ADPT_H
```

Include it in `csrc/torch_binding.cpp` near the existing sampling-related
include:

```cpp
#include "sample/gumbel_sample/gumbel_sample_torch_adpt.h"
#include "sample/gumbel_sample_from_candidates/gumbel_sample_from_candidates_torch_adpt.h"
#include "sample/build_top_k_top_p_candidates/build_top_k_top_p_candidates_torch_adpt.h"
```

The candidate Gumbel sampler adapter allocates `{batch}` int32 output and calls
`EXEC_NPU_CMD(aclnnGumbelSampleFromCandidates, ...)` with candidate logits,
candidate IDs, candidate lengths, `idx_mapping`, `seeds`, and `positions`. It
must not accept or apply temperature.

The candidate builder adapter validates `idx_mapping` and `temperature`,
allocates `{batch, candidate_capacity}` logits, `{batch, candidate_capacity}`
int32 IDs, `{batch}` int32 lengths, and `{batch}` int32 status, then calls
`EXEC_NPU_CMD(aclnnBuildTopKTopPCandidates, ...)` with the `apply_temperature`
attr. The normal compact fast path passes `apply_temperature=True`.

### 13.7 Meta Implementation

Add to `csrc/torch_binding_meta.cpp`:

```cpp
at::Tensor npu_gumbel_sample_meta(
    const at::Tensor& logits,
    const at::Tensor& idx_mapping,
    const at::Tensor& temperature,
    const at::Tensor& seeds,
    const at::Tensor& positions,
    bool apply_temperature)
{
    (void)idx_mapping;
    (void)temperature;
    (void)seeds;
    (void)positions;
    (void)apply_temperature;
    return at::empty({logits.size(0)},
                     logits.options().dtype(at::kInt).device(at::kMeta));
}
```

Register it:

```cpp
ops.impl("npu_gumbel_sample",
         &vllm_ascend::meta::npu_gumbel_sample_meta);
ops.impl("npu_gumbel_sample_from_candidates",
         &vllm_ascend::meta::npu_gumbel_sample_from_candidates_meta);
ops.impl("npu_build_top_k_top_p_candidates",
         &vllm_ascend::meta::npu_build_top_k_top_p_candidates_meta);
```

This is required for ACL graph capture and symbolic tracing.

## 14. Python Integration Plan

### 14.1 Fast Path Dispatch

Modify `vllm_ascend/worker/v2/sample/gumbel.py` for the no-filtering path:

```python
def _gumbel_sample_op(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
) -> torch.Tensor:
    return torch.ops._C_ascend.npu_gumbel_sample(
        logits,
        idx_mapping.to(dtype=torch.int32),
        temperature.to(dtype=torch.float32),
        seed.to(dtype=torch.int64),
        pos.to(dtype=torch.int64),
        apply_temperature,
    )
```

Then call it from `gumbel_sample()` before allocating `local_argmax` and
`local_max`.

For top-k/top-p, add a separate helper in `vllm_ascend/sample/sampler.py` or
extend the existing `vllm_ascend/worker/v2/sample/gumbel.py` module:

```python
def compact_top_k_top_p_sample(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    candidate_capacity: int,
    apply_temperature: bool = True,
) -> torch.Tensor:
    candidate_logits, candidate_ids, candidate_lens, status = (
        torch.ops._C_ascend.npu_build_top_k_top_p_candidates(
            logits,
            idx_mapping.to(dtype=torch.int32),
            temperature.to(dtype=torch.float32),
            p=p,
            k=k,
            candidate_capacity=candidate_capacity,
            apply_temperature=apply_temperature)
    )
    sampled = torch.empty((logits.shape[0],),
                          dtype=torch.int32,
                          device=logits.device)
    # Status partitioning is required for correctness; do not pass rows with
    # status != 0 to the candidate Gumbel sampler.
    compact_rows = status == 0
    sampled[compact_rows] = torch.ops._C_ascend.npu_gumbel_sample_from_candidates(
        candidate_logits[compact_rows],
        candidate_ids[compact_rows],
        candidate_lens[compact_rows],
        idx_mapping[compact_rows].to(dtype=torch.int32),
        seed.to(dtype=torch.int64),
        pos[compact_rows].to(dtype=torch.int64),
    )
    ...
    return sampled
```

`V1SamplerAdapter._sample()` should prefer this compact path when top-k/top-p
is present and `output_processed_logits` is not needed. It should keep the
existing full-vocab `apply_top_k_top_p` fallback for status `2` rows and for
processed-logits/logprob modes. The compact path must receive raw logits from
before the existing `apply_temperature` and `apply_top_k_top_p` steps; the
builder owns those two transformations for rows it compacts. If a future caller
passes pre-temperature logits, it may set `apply_temperature=False`, but it
must do so before top-k/top-p filtering and must still pass the original
`temperature` tensor for greedy-row detection.

### 14.2 Output Dtype

The fused op returns `int32`. Current `V1SamplerAdapter.__call__` already casts
the sampled result to `int32`, so this path is naturally compatible:

```python
sampled = self._sample(...).to(torch.int32)
```

If any future caller requires `int64` for indexing, cast at that call site
rather than storing `int64` inside the sampling op.

### 14.3 Fallback Conditions

Keep the existing Triton or full-vocab implementation as fallback for:

- missing custom op;
- `use_fp64=True`;
- `output_processed_logits is not None`;
- non-NPU tensors;
- unsupported dtype;
- candidate builder status `2` rows, including top-k/top-p sets whose exact
  filtered set does not fit in `candidate_capacity`;
- `envs.VLLM_BATCH_INVARIANT` is true and the corresponding batch-invariant
  NPU tests have not passed for the fused path;
- debugging environment switch, if added.

Add an environment flag only if needed for rollout:

```python
VLLM_ASCEND_ENABLE_GUMBEL_SAMPLE = 1 by default
```

If added, define it in `vllm_ascend/envs.py` according to the project rule for
environment variables.

## 15. Compact Candidate Mode

Compact candidate mode is part of the merged design. It is not a follow-up
extension. The final no-logprobs sampling pipeline should be:

```text
no top-k/top-p:
  npu_gumbel_sample(logits, ...)
    -> sampled_token_ids

top-k/top-p rows that can be compacted exactly:
  npu_build_top_k_top_p_candidates(logits, idx_mapping, temperature,
                                   p, k, candidate_capacity,
                                   apply_temperature=True)
    -> candidate_logits, candidate_ids, candidate_lens, candidate_status
  npu_gumbel_sample_from_candidates(candidate_logits, candidate_ids,
                                    candidate_lens, idx_mapping, seeds,
                                    positions)
    -> sampled_token_ids

rows that cannot be compacted exactly:
  existing full-vocab temperature + filtering fallback
    -> npu_gumbel_sample(masked_logits, ...)
```

The adapter chooses `candidate_capacity` as an internal performance knob. The
first implementation can use `1024` as the default compact capacity because it
covers common cases such as `top_k=50` and keeps workspace small. This value
must not be surfaced as a service-level `top_k` limit. Raising it increases
temporary tensor and workspace size; lowering it only reduces the set of rows
eligible for compact sampling.

Candidate mode changes the sampling kernel in these ways:

- `effectiveSize` becomes `Cmax` instead of `vocabSize`;
- validity mask uses `col < candidate_lens[row]`;
- output token ID is `candidate_ids[row, best_col]`, not `best_col`;
- workspace shape uses `[B, ceil_div(Cmax, tileSize)]`;
- for `Cmax <= 1024`, most rows finish with one tile and one final row
  reduction.

The candidate builder must be exact. It cannot silently truncate top-p rows.
It must apply temperature before top-p probability calculation and before
writing `candidate_logits`; otherwise top-p rows would compact a different
candidate set from the one used by the existing sampler. The candidate sampler
never divides candidate logits by temperature.

When `applyTemperature=false`, the builder assumes non-greedy logits are
already temperature-scaled and starts from top-k/top-p filtering. This mode is
for compatibility and tests, not for the default compact fast path. It is
invalid to pass logits that have already been top-k/top-p filtered into the
builder; that would conflate candidate construction with fallback filtering and
can produce different status handling.

Rows with `temperature == 0` are special: the builder must not compute softmax,
top-p cumulative probability, or random-sampling candidates for them. It must
emit a one-element exact candidate set containing the raw-logits argmax token.
This aligns compact mode with vLLM greedy behavior and avoids division by zero.

Use this status contract:

```text
candidate_status[row] = 0: exact compact candidates; pass to candidate Gumbel sampler
candidate_status[row] = 1: no active filtering; use full-vocab sampler
candidate_status[row] = 2: valid filtered set exceeds candidate_capacity; use
                           full-vocab masked fallback
```

For pure top-k with `k[row] <= candidate_capacity`, the builder should gather the
top-k logits and global token IDs directly, then write temperature-scaled
candidate logits. For pure top-k with `k[row] > candidate_capacity`, the row is
not compactable and must be marked `candidate_status[row] = 2`; it remains a
valid vLLM request and must be sampled by the full-vocab fallback. For top-p,
the builder should sort or reuse the current sorted path on temperature-scaled
logits, compute cumulative probability from those scaled logits, and compact
only when the valid set fits in `candidate_capacity`.

Mixed batches should be handled by the Python/C++ adapter by partitioning rows:

- compactable rows go through `npu_gumbel_sample_from_candidates`;
- rows with status `1` go through
  `npu_gumbel_sample(original_logits, ..., apply_temperature=True)`;
- rows with status `2` go through the existing masked full-vocab filtering path
  followed by `npu_gumbel_sample(..., apply_temperature=False)`;
- outputs are scattered back to the original row order.

This keeps correctness for unbounded top-p/no-op rows while still avoiding a
full-vocab sampling scan for common compact rows such as `top_k=50`.

## 16. Testing Plan

### 16.1 Build Tests

On an NPU machine:

```bash
SOC_VERSION=910b COMPILE_CUSTOM_KERNELS=1 pip install -e .
```

For custom op iteration only:

```bash
cd csrc
bash build.sh --pkg \
  --ops=gumbel_sample,gumbel_sample_from_candidates,build_top_k_top_p_candidates \
  --soc=ascend910b
./build/cann-ops-transformer*.run \
  --install-path=/path/to/vllm-ascend/vllm_ascend/_cann_ops_custom
```

Then rebuild the Python extension if `torch_binding.cpp` changed.

### 16.2 Unit Correctness

Add `tests/ut/sample/test_gumbel_sample_op.py`, skipped when NPU or the
custom op is unavailable.

Required cases:

1. **Greedy rows**

   ```python
   temperature = torch.zeros(B, device="npu", dtype=torch.float32)
   out = torch.ops._C_ascend.npu_gumbel_sample(...)
   assert torch.equal(out, logits.argmax(dim=-1).to(torch.int32))
   ```

2. **Determinism**

   Same logits, seeds, positions, and temperature must produce identical output
   across repeated calls.

3. **Counter independence**

   Changing `positions` must be able to change output. Changing batch size must
   not change a row's sample when that row has the same seed, position, and
   logits.

4. **Temperature**

   Compare `apply_temperature=True` against a reference call using
   pre-divided logits and `apply_temperature=False` with the same seeds and
   positions.

5. **Shape and dtype errors**

   Validate wrong rank, wrong dtype, and mismatched batch dimensions.

6. **Distribution smoke**

   For small `V`, run many positions and compare empirical frequency against
   `softmax(logits)` with a loose tolerance.

7. **Compact candidate equivalence**

   Build candidates for top-k and top-p rows, sample with
   `npu_gumbel_sample_from_candidates`, and compare the empirical
   distribution against sampling from the corresponding masked logits where
   temperature was applied before filtering.

8. **Temperature and top-p order**

   Use logits where top-p membership differs before and after temperature
   scaling. Verify `npu_build_top_k_top_p_candidates` matches the reference
   order `apply_temperature -> apply_top_k_top_p`, not
   `apply_top_k_top_p -> apply_temperature`.

9. **Builder `apply_temperature` flag**

   Compare builder output with `apply_temperature=True` on raw logits against
   `apply_temperature=False` on pre-temperature logits for non-greedy rows.
   Include `temperature == 0` rows and verify they still emit a single raw
   argmax candidate in both modes.

10. **Mixed greedy and random rows**

   Use a batch where some rows have `temperature == 0` and others have
   `temperature > 0`, with top-k/top-p present. Verify greedy rows return
   `argmax(raw_logits)`, have `candidate_lens == 1` in compact mode, do not
   consume RNG, and are unaffected by neighboring random rows.

11. **Candidate overflow fallback**

   Force `candidate_capacity` smaller than a top-p valid set and smaller than a
   valid top-k value. Verify `candidate_status == 2`, then verify the adapter
   routes that row through the full-vocab masked fallback without rejecting the
   request.

12. **Batch invariant**

    Run the same target request with identical logits, seed, position,
    temperature, top-k/top-p settings, and request ID in at least three layouts:
    alone, first row in a larger batch, and later row in a larger batch. Verify
    full-vocab sampling, compact sampling, no-op status rows, and overflow
    fallback all return the same token for the target request. Also verify
    `candidate_ids` are identical for compact top-k/top-p rows with tied logits.

### 16.3 Performance Tests

Extend `tests/ut/sample/test_gumbel_sample_perf.py`:

```python
BenchImpl("gumbel_sample_full_vocab")
BenchImpl("gumbel_sample_top_k_50")
```

Measure:

- golden exponential race;
- old Triton gumbel;
- new fused Gumbel sample;
- compact top-k candidate builder plus candidate Gumbel sampler.

Target cases:

```text
B in (1, 8, 32, 128)
V = 152064
temperature = 1
top-k/top-p = None
top-k = 50
top-k = 1024
mixed compact/full rows
```

Report:

```text
batch vocab impl                         ms speedup_vs_golden
```

Do not merge the fast path as default unless it beats the acceptance targets in
section 2.1. A result that is faster than the current Triton `gumbel_sample` but
slower than the exponential-race reference is not enough for default enablement;
keep it behind an env/config flag and treat the profiling data as a follow-up
optimization input.

### 16.4 CI and Style

Run:

```bash
bash format.sh ci
pytest -sv tests/ut/sample/test_gumbel_sample_op.py
pytest -sv tests/ut/sample/test_v1_sampler_adapter.py
```

Run the NPU performance test manually because it is benchmark-oriented and
requires NPU hardware:

```bash
pytest -sv tests/ut/sample/test_gumbel_sample_perf.py
```

## 17. Profiling Checklist

Use msprof or CANN profiling to confirm:

- only one sampling custom op appears for the fast path;
- there is no Python-side `argmax` or `gather` after the custom op;
- workspace writes are `float32 + int32`, not `int64`;
- full-vocab `temperature == 0` rows and compact single-candidate rows skip
  RNG and transcendental ops;
- `idxMapping`, `seeds`, and `positions` stay device-side;
- no host-to-device seed copy occurs inside the op;
- row-reduction time is not dominating at small batch sizes;
- tail tile masking does not produce out-of-range token IDs.
- top-k/top-p compact rows do not execute the full-vocab sampling kernel;
- candidate builder status counts are visible in debug logs or benchmark
  output.

Important counters to inspect:

- AIV vector active cycles;
- MTE2/MTE3 bandwidth;
- UB usage and spills;
- scalar instruction ratio in RNG;
- number of `Log` and `Exp` vector instructions;
- SyncAll overhead between phases.

## 18. Risks and Mitigations

### RNG quality or speed is insufficient

Mitigation:

- start with deterministic unit tests and distribution smoke tests;
- profile scalar ratio;
- replace Wang hash with Philox4x32 or a toolkit RNG primitive if needed;
- keep the old Triton path as fallback until statistical and performance tests
  are accepted.

### Extra row-max pass costs too much

Mitigation:

- benchmark a direct Gumbel score variant:

  ```text
  score = logits / temp - log(-log(U))
  ```

- choose by measured latency, not by operation count alone;
- retain the same op interface so the internal formula can change.

### `SyncAll` overhead hurts small batch

Mitigation:

- add a single-row tiling key where one row is reduced by one core group with
  fewer global workspace phases;
- specialize `B == 1` if profiling shows synchronization dominates.

### Candidate builder dominates latency

Mitigation:

- specialize pure top-k before generalized top-p;
- keep `Cmax <= 1024` in the hot path;
- reuse existing sorted data only when it avoids extra full-vocab passes;
- report builder time and candidate Gumbel sampler time separately in the
  benchmark.

### Processed logits and logprobs need old semantics

Mitigation:

- keep `output_processed_logits` on fallback initially;
- later add a separate processed-logits op or optional output only if logprob
  profiling shows it matters;
- avoid writing full processed logits in the default no-logprobs path.

## 19. Single-PR Implementation Phases

This work should land as one PR. The phases below are implementation and review
checkpoints inside that PR, not separate PRs.

1. **Phase 1: buildable op skeletons**
   - add all three op directories, CMake files, `build_aclnn.sh` entries,
     torch bindings, and meta functions;
   - implement greedy-only full-vocab and candidate Gumbel samplers;
   - verify custom-op packaging and greedy correctness.
2. **Phase 2: shared random sampling core**
   - add `sampling_common/counter_rng.h`;
   - implement probability-race sampling for full-vocab and candidate modes;
   - add determinism, temperature, and distribution tests.
3. **Phase 3: compact candidate builder**
   - implement exact top-k compaction first;
   - implement top-p compaction with `candidate_status` overflow reporting;
   - add candidate equivalence and overflow fallback tests.
4. **Phase 4: Python integration**
   - route no-filtering rows to `npu_gumbel_sample`;
   - route compactable top-k/top-p rows to candidate builder plus candidate
     sampler;
   - route overflow/no-op rows through the full-vocab fallback and scatter all
     outputs back to original order.
5. **Phase 5: performance closure**
   - benchmark full vocab, top-k=50, top-k=1024, and mixed rows;
   - report candidate builder and candidate Gumbel sampler latency separately;
   - keep fallback gates only for unsupported correctness surfaces, not for the
     intended fast paths.

The PR is complete only when both full-vocab and compact-candidate Gumbel
sampling are implemented, tested, and wired into the no-logprobs sampling path.
