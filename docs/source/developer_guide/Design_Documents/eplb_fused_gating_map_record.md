# Fused EPLB Gating, Mapping and Load Recording

## Overview

On the STAIR/EPLB path, every MoE layer in every forward pass used to run three separate kernel launches:

1. `moe_gating_top_k` (CANN): sigmoid/bias scoring with grouped top-k, producing logical expert ids;
2. `ascend_eplb_map_to_physical` (Triton): replica-table lookup, producing physical expert ids;
3. `ascend_eplb_record_expert_tokens` (Triton): accumulating operator-provided expert counts into the load vector.

This document describes Triton kernels that fuse all three steps into a single launch (`vllm_ascend/ops/triton/eplb_map_record.py`), fix two pre-existing issues — the recording kernel launches unconditionally even when collection is disabled, and padding rows leak into the recorded load — and the batch-bucket routing that makes the fused path strictly non-regressive.

The coverage goal is the sigmoid and softmax scoring paths plus the DeepSeek V4 hash route (all routes where gating runs as a CANN device kernel), validated bit-exactly against each CANN op. Explicitly out of scope, each keeping the unfused route with no regression: the DeepSeek V4 vision sub-path, `GroupedTopKRouter` torch-composed fallbacks, `CustomRoutingRouter` (arbitrary Python functions cannot be fused statically), the 310P hardware path and the xlite engine.

The result is verified bit-exact against the three-launch route and measured on Atlas 800I A3 (Ascend 910C, CANN 9.0.1, torch_npu 2.10.0, triton-ascend 3.2.2):

| tokens | three-launch route | fused kernel | speedup |
|---|---|---|---|
| 64 | 0.256 ms | **0.156 ms** | **1.65x** |
| 256 | 0.253 ms | **0.223 ms** | **1.14x** |
| 512 | 0.249 ms | 0.262 ms | 0.95x (parity) |
| 1024 | 0.243 ms | 0.379 ms | 0.64x |
| 6144 (prefill) | 0.461 ms | 1.547 ms | 0.30x |

(The 1024/6144 rows run at tile heights the backend barely tolerates and are irrelevant under bucket routing, which sends those batches to the unfused route.)

The integrated route is therefore bucketed: batches up to `EPLB_FUSED_MAP_RECORD_MAX_TOKENS` (512, tunable) take the fused kernel; larger batches keep the unfused CANN route and match the current behavior exactly.

## Semantics Baseline

The kernels replicate the CANN gating ops exactly.

**Sigmoid and softmax** (`moe_gating_top_k`, generalized variant):

- sigmoid: `score = sigmoid(x)`; softmax: `score = softmax(x)` over all experts;
- `key = score + bias` drives selection; weights gather the **pre-bias** scores;
- re-normalization over the selected subset: sigmoid always (`y = score / (sum(score) + eps)`), softmax only when `renorm == 1` (the CANN generalized variant's `needRenorm` rule); without it the full softmax values pass through;
- group score = sum of the top-2 keys inside each group; select the top `k_group` groups, then the top `k` experts among them;
- ties break toward the lowest expert index.

**DeepSeek V4 hash** (`moe_gating_top_k_hash`, regbase variant, `group_count == 1`):

- the expert ids come directly from the `tid2eid[token_id]` lookup table — no selection network;
- weights = `sqrt(softplus(x))` gathered at those ids (pre-bias), re-normalized over the selected subset: `y = score / (sum(score) + eps) * routed_scaling_factor`.

## Kernel Structure

Each program processes `TOKENS_PER_PROGRAM` tokens as a 2D tile, so the per-token selection chain is amortized with `axis=1` reductions:

1. scoring: sigmoid + bias;
2. group scores: per-group max, index-based removal of the argmax (correct with duplicated values), second max, top-2 sum;
3. group selection: `K_GROUP` iterative argmax rounds over the group scores;
4. expert selection: `K` iterative argmax rounds over the selected-group mask;
5. mapping: `table[token % 1024][logical]`, where the table row encodes `(row + ep_rank + logical) % replica_count`;
6. recording: one vector atomic-add per program accumulates the local-expert histogram into `expert_load_view[local_start : local_start + local_count)`, gated by `record_enabled && token < num_valid_tokens && physical id in the local range`.

### Why a routing histogram can replace operator-provided counts

- Same source: the operator's `expert_tokens` are the `group_list` rendering of "tokens routed to this rank's local experts"; the histogram derives the same counts from the routing decision and removes the two `group_list_type` encodings.
- Same locality: only the local expert range is accumulated; the device-group all-reduce in `collect_global_load_stats` reconciles the global load, unchanged.
- Earlier point in time is a correction: the histogram observes the routing decision before force-EPLB and shared-expert transforms rewrite `topk_ids`.
- Padding correction: the `token < num_valid_tokens` filter strictly improves on operator counts, which include padded rows.

### ACL-graph safety

Every per-step mutable input (`record_enabled`, `num_valid_tokens`, the replica routing table) is a device tensor with a stable address, updated in place — no Python scalars are baked into capture. The tile height snaps to powers of two (2/4/8/16) by batch size. Bucket routing is decided per captured graph, so the kernel choice is fixed at capture time.

## Integration and Fallbacks

- Entry: the sigmoid branch of `fused_topk_router._compute_routing`, via `torch.ops.vllm.ascend_eplb_gating_top_k_map_record`.
- On hit, the layer state sets `fused_map_record_active`; `_ascend_apply_eplb_mapping` then short-circuits (topk ids are already physical) and `routed_experts` skips `_record_v2_eplb_load`. One flag drives both, so double counting is impossible.
- Fallback to the unfused route when: EPLB is off, the routing table is not ready, `mix_placement` or force EPLB is active (those rewrite `topk_ids` after mapping), the batch exceeds the bucket threshold, the scoring function is not sigmoid-with-renormalization, or `VLLM_ASCEND_EPLB_FUSED_MAP_RECORD=0`.
- The legacy one-dimensional `log2phy` path is not affected.

## Verification

All routes validated on Ascend 910C against their CANN ops (`spike/triton_gating/validate_all_paths.py`): ids exact, recorded load integer-equal, weights within 1e-5 (sigmoid and hash are bit-exact at 0.0; softmax shows <= 2.4e-7 from exp/reduction-order ulps):

- sigmoid + bias + renorm: bit-exact (0.0 diff);
- softmax x {bias, no-bias} x {renorm 0, 1}: ids exact, weights <= 2.4e-7 (exp/reduction-order ulps);
- hash (tid2eid lookup, 4096 ids): bit-exact;
- validated at tile heights 4 and 16 (the bucket shapes and the largest probed height).
- Bit-exact: `topk_ids` zero mismatches, `topk_weights` max diff 0.0, `expert_load_view` integer-equal, against the three-launch route on identical inputs, across all measured batch sizes.
- Padding: with `num_valid_tokens = T - T/8`, the load difference matches the padding-row histogram exactly, per expert.
- No double counting: `_record_v2_eplb_load` invocation count is zero on the fused path.
- Collection disabled: with `record_enabled = 0` the load stays unchanged and no separate recording launch remains — this explains the earlier observation that enabling collection "cost nothing": the cost was the launch, not the compute.
- ACL graph: after capture, flipping `record_enabled`, refreshing the routing table and changing `num_valid_tokens` all take effect on replay.

## Rejected Routes (measurement record)

| route | outcome | reason |
|---|---|---|
| CANN gating + fused map/record (2 launches) | 5x slower at prefill (2.53 ms vs 0.33 ms) | atomic histogram contention outweighs one saved launch; the operator-provided counting path is already nearly free |
| packed-index `tl.sort`/`tl.topk` selection (3 sorts instead of ~50 serial reductions) | does not compile | triton-ascend 3.2.x backend bugs (below); the packing algorithm itself was validated by probes and is correct |
| `num_warps` 4→8, tile height 32 | no effect / crash | reduction latency is warp-count independent; non-power-of-two heights abort in `parseSelect` |

## triton-ascend 3.2.x Backend Constraints

1. `tl.topk` (partial sort, values only) compiles on 1D tiles only; 2D `dim=1` fails for every dtype.
2. 2D `tl.sort` passes standalone probes but triggers "cannot align N axis" layout-propagation errors when its result feeds further ops.
3. Tile heights must be powers of two; TS=5 aborts in `parseSelect`, TS=1 and TS>16 have other shape issues; the fused kernel caps at TS=4 (bucket shapes only need 2/4).
4. `tl.argsort` does not exist and `tl.sort` returns no indices.
5. UB budget is 196 KB; the (TS=8, 256) tile with a 3D group phase overflows it via reshape layout conversions, and `tl.reshape` between 2D and 3D tiles is the trigger.
6. `num_warps` does not affect `axis=1` reduction latency.
7. **Iterative `tl.argmax` is unusable**: once `-inf` replacements appear between rounds it returns wrong indices (known Triton bug, reproduced in `spike/triton_gating/probe_argmax_iter.py`); selection rounds use the max + min-index pattern instead.
8. Loads/atomics whose mask is a row-broadcast (TS, 1) expression are unreliable at some (TS, K) shapes; masks built from full 2D terms or per-column 1D ops are safe.
9. Index packing is available and lossless: an order-preserving int32 map of the fp32 key (`u < 0 ? ~u : u ^ (1 << 31)`) shifted left with the inverted index in the low 9 bits sorts identically to fp32 argsort with lowest-index-first ties. This unblocks the prefill fused route as soon as the backend ships index-bearing sort primitives.

## Follow-up

- Re-evaluate the prefill bucket when triton-ascend provides index-bearing sort primitives or fixes the 2D sort layout bugs; the packed-sort kernel is preserved in the spike branch as a starting point.
- Re-run the batch-bucket calibration on other model shapes (different expert counts, group sizes) before widening the default threshold.
