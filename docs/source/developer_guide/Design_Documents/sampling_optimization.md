# Sampling Optimization Design

## 1. Background

RFC 9269 can be summarized as follows: reduce the end-to-end cost of
sampling and speculative decoding rejection sampling in vLLM Ascend, unblock
the mainline migration to ModelRunnerV2, and obtain part of the performance
benefit while ModelRunnerV1 is still the primary execution path.

The current ModelRunnerV1 sampling path mainly reuses the existing Ascend
sampler and rejection sampler. The path is functionally complete, but it has
several performance problems: random sampling and Gumbel generation are often
serialized with model execution, rejection sampling still performs expensive
dense vocab work, and the sampling path is tightly coupled with logits
processing details.

ModelRunnerV2 has a cleaner sampling layout and a rejection sampling flow that
is closer to upstream vLLM. Its structure is more suitable for external random
number generation, V2-style input packing, and future migration to the V2
mainline. Therefore, the goal of this design is not to maintain another
long-term ModelRunnerV1-only sampling implementation. Instead, ModelRunnerV1
should reuse a V2-style sampling and rejection sampling path where it is safe
to do so.

This allows vLLM Ascend to get performance gains in ModelRunnerV1 first, while
keeping the implementation direction aligned with the future ModelRunnerV2
mainline.

## 2. Value

1. The ModelRunnerV2-based sampling design provides a large optimization
   opportunity for Gumbel-related operators, which is one of the major
   blockers for switching the mainline path to ModelRunnerV2.
2. The ModelRunnerV2 rejection sampling flow is more efficient. Reusing it in
   ModelRunnerV1 allows us to get the performance gain before V2 fully replaces
   V1.
3. Optimizing around the V2 rejection sampling flow makes the work easy to move
   forward in later versions.
4. Compared with reduced sampling, this path can still provide benefits even
   when there is no `top_k` parameter.
5. Compared with reduced sampling, this path does not need to understand the
   internals of logits processing. It is more flexible and easier to maintain.
6. This design can be combined with reduced sampling later. Reduced sampling
   can reduce the candidate space, while this design optimizes random number
   generation, sampling, and rejection sampling.

## 3. Terms and Current State

- **Regular sampling**: sampling one output token from target model logits
  after logits processing, without speculative decoding.
- **Speculative decoding**: a draft model, MTP layer, or another proposer
  generates draft tokens first; the target model verifies multiple tokens in
  one forward pass; rejection sampling decides how many draft tokens can be
  accepted and which recovery token should be sampled after rejection.
- **Rejection sampling**: accepts draft tokens according to the probability
  relation between target and draft distributions. Once a token is rejected,
  the sampler draws a recovery token from the residual distribution.
- **Gumbel sampling**: represents random sampling as
  `argmax(logits + gumbel)`. This form is friendly to fused kernels and
  external random number generation.
- **Reduced sampling**: reduces the candidate token set using top-k/top-p or
  similar mechanisms. It is useful, but it depends on candidate sets produced
  by logits processing. This design does not require reduced sampling.

## 4. Prototype Experiments

### 4.1 Setup

The prototype was validated on a real NPU environment. The performance unit
test used the following settings:

- vocab size: `151936`
- batch sizes: `1, 4, 16, 64, 128`
- speculative tokens: `4`
- warmup iterations: `5`
- measurement iterations: `20`
- old path golden: the existing Ascend rejection sampler, with async random
  number generation simulated for fairness
- new path: ModelRunnerV1 routes probabilistic rejection sampling through a
  V2-style rejection sampler and uses external acceptance uniform and recovery
  Gumbel tensors

Test command:

```bash
python -m pytest -q -s tests/ut/sample/test_upstream_rejection_sampler_poc.py
```

### 4.2 Effective Prototype Result

After removing candidate resampling, the dense vocab recovery path produced the
following result:

| Batch size | Old async random ms | New external random ms | Speedup |
|---:|---:|---:|---:|
| 1 | 2.967 | 2.264 | 1.310x |
| 4 | 2.931 | 2.251 | 1.302x |
| 16 | 5.595 | 4.194 | 1.334x |
| 64 | 19.672 | 17.125 | 1.149x |
| 128 | 38.410 | 33.841 | 1.135x |

Conclusions:

- Even when the old path also simulates async random number generation, the new
  path still has stable performance gains.
- Small batches benefit more, but batch sizes 64 and 128 still show meaningful
  gains.
- The gain mainly comes from the V2-style rejection sampling flow, external
  random number generation, fewer Python-side intermediate constructions, and a
  more compact sampling path.

### 4.3 Current Implementation Puncture Result

After implementing the ModelRunnerV1 bridge, the puncture benchmark compared
three paths for both regular sampling and draft-free speculative
rejection sampling:

1. `v1_original`: the current ModelRunnerV1 Ascend sampler or rejection
   sampler path;
2. `v2_native`: the V1 bridge using the current V2 NPU native sampling helper;
3. `our_optimized`: the V1 bridge using prefetched random tensors and the
   optimized draft-free operator.

The benchmark was run on a real NPU with NPU event timing, 5 warmup iterations,
and 20 measured iterations. Timing covers the sampling critical path only. The
`our_optimized` rows assume random tensors have already been prefetched, which
matches the intended ModelRunnerV1 overlap design.

Test command:

```bash
python benchmarks/ops/bench_sampling_paths.py \
  --batch-size 128 --vocab-size <vocab_size> --spec-steps 3 \
  --warmups 5 --iterations 20
```

Regular sampling:

| Batch size | Vocab size | Path | Mean ms | Median ms | P90 ms | Speedup vs v1 |
|---:|---:|---|---:|---:|---:|---:|
| 128 | 32000 | v1 original | 0.946 | 0.943 | 0.967 | 1.00x |
| 128 | 32000 | v2 native | 33.316 | 33.260 | 33.535 | 0.03x |
| 128 | 32000 | our optimized | 0.249 | 0.248 | 0.265 | 3.80x |
| 128 | 151936 | v1 original | 3.120 | 3.106 | 3.129 | 1.00x |
| 128 | 151936 | v2 native | 152.799 | 152.789 | 152.848 | 0.02x |
| 128 | 151936 | our optimized | 0.755 | 0.754 | 0.762 | 4.13x |

Speculative rejection sampling without draft probabilities:

| Batch size | Vocab size | Path | Mean ms | Median ms | P90 ms | Speedup vs v1 |
|---:|---:|---|---:|---:|---:|---:|
| 128 | 32000 | v1 original | 2.773 | 2.771 | 2.788 | 1.00x |
| 128 | 32000 | v2 native | 33.809 | 33.798 | 33.849 | 0.08x |
| 128 | 32000 | our optimized | 1.998 | 2.001 | 2.061 | 1.39x |
| 128 | 151936 | v1 original | 11.462 | 11.460 | 11.484 | 1.00x |
| 128 | 151936 | v2 native | 153.445 | 153.437 | 153.553 | 0.07x |
| 128 | 151936 | our optimized | 6.491 | 6.535 | 6.739 | 1.77x |

Standalone random prefetch cost, not included in the critical-path speedup
above:

| Batch size | Vocab size | Prefetch tensor | Mean ms | Median ms | P90 ms |
|---:|---:|---|---:|---:|---:|
| 128 | 32000 | regular sampling Gumbel | 0.657 | 0.653 | 0.665 |
| 128 | 32000 | rejection uniform + recovery Gumbel | 0.761 | 0.758 | 0.773 |
| 128 | 151936 | regular sampling Gumbel | 2.610 | 2.638 | 2.659 |
| 128 | 151936 | rejection uniform + recovery Gumbel | 2.716 | 2.719 | 2.769 |

Additional batch sweep for vocab size `151936`:

```bash
python benchmarks/ops/bench_sampling_paths.py \
  --batch-size <batch_size> --vocab-size 151936 --spec-steps 3 \
  --warmups 5 --iterations 20
```

Regular sampling batch sweep:

| Batch size | V1 original ms | V2 native ms | V2 native speedup | Our optimized ms | Our speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.659 | 1.559 | 0.42x | 0.217 | 3.03x |
| 8 | 0.647 | 9.856 | 0.07x | 0.223 | 2.90x |
| 32 | 0.996 | 38.637 | 0.03x | 0.221 | 4.50x |
| 64 | 1.650 | 76.697 | 0.02x | 0.249 | 6.62x |
| 96 | 2.412 | 114.786 | 0.02x | 0.526 | 4.59x |

Speculative rejection sampling batch sweep:

| Batch size | V1 original ms | V2 native ms | V2 native speedup | Our optimized ms | Our speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.627 | 2.872 | 0.91x | 1.854 | 1.42x |
| 8 | 2.047 | 10.746 | 0.19x | 1.592 | 1.29x |
| 32 | 3.371 | 39.028 | 0.09x | 1.962 | 1.72x |
| 64 | 5.959 | 77.021 | 0.08x | 3.365 | 1.77x |
| 96 | 8.848 | 115.378 | 0.08x | 5.209 | 1.70x |

Conclusions:

- The optimized regular sampling path shows strong speedup because the
  critical path only needs processed logits plus prefetched Gumbel tensors.
- The optimized rejection path also improves over the V1 original path, and the
  gain increases at the larger vocab size.
- Directly routing through the current V2 NPU native helper is not a viable V1
  bridge for this case. Its in-kernel random path is dominated by dense-vocab
  random generation and is much slower than both the V1 original path and the
  prefetched-random optimized path.
- If random prefetch cannot overlap with model execution, the standalone random
  generation cost should be accounted for separately. The design depends on
  moving this work before sampling and overlapping it with model forward.

### 4.4 Experiments That Should Not Be Productized

#### Candidate resampling

The prototype once tried to replace dense vocab resampling with candidate
resampling. The idea was to run `topk` on processed logits first and then
sample only from the top-k candidate set. This direction should not be used in
the formal implementation.

The reason is that `torch.topk(processed_logits, k)` scans
`num_logits * vocab`, while dense resampling only scans `num_reqs * vocab`.
When the number of speculative tokens is 4, `num_logits` is roughly
`5 * num_reqs`. The extra top-k cost is larger than the resampling cost it
saves, especially for batch sizes 64 and 128.

Candidate resampling result:

| Batch size | Old async random ms | Candidate new ms | Speedup |
|---:|---:|---:|---:|
| 1 | 3.019 | 2.410 | 1.253x |
| 4 | 3.009 | 2.399 | 1.254x |
| 16 | 5.643 | 4.469 | 1.263x |
| 64 | 19.668 | 18.122 | 1.085x |
| 128 | 38.443 | 35.907 | 1.071x |

Future work should only revisit candidate resampling if the candidate set can
be reused from logits processing without an extra dense top-k, or if candidate
selection is performed only for the actual rejected rows after rejection is
known.

#### Skipping resampling

Skipping resampling was useful for diagnosis and showed that resampling has a
visible cost. However, it is not mathematically equivalent to rejection
sampling and must not be used as the final solution.

#### Saving draft logits or draft probabilities

The prototype also tested paths that store draft logits. The formal
ModelRunnerV1 path should avoid saving draft logits or constructing draft
probabilities in the first implementation.

The main reason is memory and copy overhead. Full draft logits have shape
similar to `[max_reqs, num_spec_tokens, vocab_size]`. Storing and copying this
tensor can easily offset the sampling benefit.

The formal V1 path should instead use a new draft-free rejection sampling
operator that explicitly supports the draft-free mode. This avoids
changing the existing ModelRunnerV2 operator semantics.

## 5. Design Goals

1. Do not add a user-facing switch. ModelRunnerV1 should use the new branch by
   default when all constraints are satisfied.
2. Reuse the ModelRunnerV2 sampling layout and rejection sampling idea, but do
   not break the existing ModelRunnerV2 operator.
3. Support both regular sampling and probabilistic speculative decoding.
4. Support all common logits processing parameters in the final implementation,
   including temperature, top-k, top-p, min-p, presence penalty, frequency
   penalty, repetition penalty, bad words, allowed token ids, structured
   output, and greedy sampling.
5. Do not support logprobs in the first version. If logprobs are requested,
   fallback to the old path.
6. Generate random tensors before model graph execution and overlap random
   generation with model execution on a separate NPU stream.
7. Keep the engineering implementation small and focused. Avoid broad
   abstractions and long defensive code. If an internal invariant is violated,
   fail loudly instead of silently producing unexpected behavior.

## 6. New Branch Enablement Conditions

No new configuration switch should be added. ModelRunnerV1 should check whether
the new sampling path can be used at each sampling step.

The new path can be used only when:

- the current device is NPU;
- the request does not require logprobs;
- the current sampling inputs can be packed into the V2-style batch format;
- all logits processing parameters used by the request are supported by the
  new branch;
- speculative decoding is disabled, or speculative decoding uses the
  `probabilistic` rejection sampling method;
- the current parallel or sharding mode has clear global vocab semantics.

If any condition is not satisfied, fallback to the old path.

Suggested shape of the helper:

```python
def _can_use_v2_sampling_path(self, sampling_metadata, spec_decode_metadata) -> bool:
    if sampling_metadata.max_num_logprobs is not None:
        return False
    if self.input_batch.sampling_metadata.logprobs is not None:
        return False
    if spec_decode_metadata is not None:
        return self.speculative_config.rejection_sample_method == "probabilistic"
    return True
```

The exact field names should be adjusted to the current `SamplingMetadata`
definition. Keep the function short.

## 7. Overall Architecture

The new design has four layers:

1. **Entry selection**: a small check near `NPUModelRunner._sample` decides
   whether to use the new branch.
2. **Input adaptation**: ModelRunnerV1 `logits`, `SamplingMetadata`, and
   `SpecDecodeMetadata` are packed into tensors expected by the V2-style
   sampler and rejection sampler.
3. **Random prefetch**: random tensors are allocated or reused on the default
   stream before model graph execution, then filled on a separate NPU stream.
4. **Sampling operators**: add draft-free V2-style sampling and rejection
   sampling operators. The rejection sampling operator explicitly supports the
   draft-free mode.

Data flow:

```text
execute_model
  ├─ allocate or reuse random buffers on the default stream
  ├─ fill acceptance_uniform / sampling_gumbel / recovery_gumbel on random stream
  ├─ run model graph on the default stream
  └─ _sample
       ├─ check whether the new branch is available
       ├─ build V2-style input batch
       ├─ run logits processing
       ├─ run regular sampling or rejection sampling
       └─ return SamplerOutput
```

## 8. Input Format Adaptation

### 8.1 Regular Sampling

When speculative decoding is disabled, the input format is simple:

- `target_logits`: `[num_reqs, vocab_size]`
- `idx_mapping`: `[num_reqs]`
- `temperature/top_k/top_p/...`: per-request tensors
- `sampling_gumbel`: `[num_reqs, vocab_size]`, or another layout required by the
  operator

Output:

- `sampled_token_ids`: `[num_reqs, 1]`

Regular sampling should use the same processed-logits and Gumbel-argmax path so
that it can benefit from the same optimization even when speculative decoding
is not enabled.

### 8.2 Speculative Decoding Rejection Sampling

For speculative decoding, ModelRunnerV1 `SpecDecodeMetadata` should be
converted to:

- `target_logits`: `[num_logits, vocab_size]`
- `draft_tokens`: `[num_logits]`
- `cu_num_logits`: `[num_reqs + 1]`
- `idx_mapping`: `[num_reqs]`
- `expanded_idx_mapping`: `[num_logits]`
- `expanded_local_pos`: `[num_logits]`
- `positions`: `[num_logits]`
- `acceptance_uniform`: `[num_logits]`
- `recovery_gumbel`: `[num_reqs, vocab_size]`

Meanings:

- `cu_num_logits[i]` and `cu_num_logits[i + 1]` describe the logits range of
  request `i`.
- `expanded_idx_mapping[row]` maps a logits row to the request index.
- `expanded_local_pos[row]` maps a logits row to its speculative step inside
  the request.
- `draft_tokens` is gathered from `input_ids[logits_indices]` and is used in
  the acceptance test.

The prototype verified that `cu_num_logits`, `idx_mapping`,
`expanded_idx_mapping`, and `expanded_local_pos` can be preallocated and reused.
This avoids repeated `arange`, `repeat_interleave`, and `empty` calls on the
hot path.

The formal implementation should keep this cache in a small helper, for
example:

```python
class SamplingInputBuilder:
    def build_spec_inputs(...)
    def build_sampling_inputs(...)
```

Avoid scattering temporary sampling state across the main `NPUModelRunner`
class.

## 9. Draft Probabilities Strategy

The formal implementation should not save draft logits and should not
construct draft probabilities.

Reasons:

- full draft logits have shape similar to
  `[max_reqs, num_spec_tokens, vocab_size]`, which is expensive in memory;
- saving draft logits requires extra copies during proposal generation;
- the first ModelRunnerV1 implementation should focus on getting the V2-style
  sampling performance benefit, not on expanding full probability semantics.

Implementation strategy:

- add a draft-free rejection sampling operator, for example
  `sample_with_rejection`;
- make the draft-free mode explicit in the operator name or
  interface;
- use target distribution and external Gumbel tensors for recovery sampling;
- call this operator only from the ModelRunnerV1 new branch;
- do not change the existing ModelRunnerV2 rejection sampler.

If full draft probabilities are needed later, they should be added in a
separate change with performance data that proves the extra memory and copy
cost is justified.

## 10. Logits Processing Support

The first formal implementation should cover common sampling parameters:

- `temperature`
- `top_k`
- `top_p`
- `min_p`
- presence penalty
- frequency penalty
- repetition penalty
- bad words
- allowed token ids
- structured output or grammar mask
- greedy sampling

Recommended approach:

1. Reuse the existing logits processing path as much as possible.
2. Let the new sampling branch consume processed logits instead of
   reimplementing every logits processor.
3. Only lower a parameter into the V2-style sampler when an equivalent V2
   entry point already exists.
4. If equivalence is unclear, fallback to the old path.

The key principle is that this change should optimize sampling and rejection
sampling. It should not become another implementation of logits processing.

## 11. Async Random Number Generation

This is the most important implementation detail.

### 11.1 Why Random Generation Must Be Moved Earlier

Sampling needs two classes of random numbers:

- acceptance uniform numbers for rejection sampling;
- Gumbel or exponential numbers for regular sampling and recovery sampling.

If these tensors are generated inside `_sample`, random generation is
serialized after model execution. The correct design is to start random number
generation before model graph execution in `execute_model`, after the runtime
knows the current batch can use the new sampling branch.

### 11.2 Avoid the Known `enable_async_exponential` Pitfall

The existing `enable_async_exponential` idea is useful, but its implementation
has a pitfall: allocating tensors inside the async stream can conflict with the
default stream memory lifecycle and cached allocator reuse.

The required rule is:

1. allocate or reuse random tensors on the default stream;
2. make the random stream wait for the default stream;
3. fill tensor values on the random stream;
4. make the default stream wait for a ready event before sampling.

Pseudo-code:

```python
def _prepare_sampling_noise(self, num_logits, num_reqs, vocab_size):
    acceptance_uniform = self._get_buffer("acceptance_uniform", (num_logits,), torch.float32)
    sampling_gumbel = self._get_buffer("sampling_gumbel", (num_logits, vocab_size), torch.float32)
    recovery_gumbel = self._get_buffer("recovery_gumbel", (num_reqs, vocab_size), torch.float32)

    current_stream = torch.npu.current_stream()
    random_stream = self.v2_sampling_random_stream
    random_stream.wait_stream(current_stream)

    with torch.npu.stream(random_stream):
        acceptance_uniform.uniform_()
        acceptance_uniform.clamp_(min=1e-20)
        sampling_gumbel.exponential_()
        sampling_gumbel.log_().neg_()
        recovery_gumbel.exponential_()
        recovery_gumbel.log_().neg_()
        self.v2_sampling_random_ready_event.record()

    return acceptance_uniform, sampling_gumbel, recovery_gumbel
```

Before sampling:

```python
torch.npu.current_stream().wait_event(self.v2_sampling_random_ready_event)
```

The implementation should generate only the random tensors needed by the
current path. For example, do not unconditionally allocate a large
`recovery_gumbel` tensor for regular sampling.

## 12. Operator Design

Do not directly modify the rejection sampler operator currently used by
ModelRunnerV2.

Reasons:

- the existing V2 functionality must not be affected by a V1 transition path;
- ModelRunnerV1 needs the draft-free mode first, which is not the
  same as the full V2 semantics;
- a draft-free operator can have fewer branches and a smaller interface.

Suggested files:

```text
vllm_ascend/sample/sampling_bridge.py
vllm_ascend/sample/rejection_ops.py
```

If a new directory is not desired, place the helper near
`vllm_ascend/worker/model_runner_v1.py`, but avoid putting long kernels and
input builders directly into `model_runner_v1.py`.

Suggested rejection sampling interface:

```python
def sample_with_rejection(
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    cu_num_logits: torch.Tensor,
    positions: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    acceptance_uniform: torch.Tensor,
    recovery_gumbel: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

`draft_probs=None` is the ModelRunnerV1 path because MRv1 does not keep draft
probabilities. A non-`None` `draft_probs` input is aligned with
`target_logits` rows, kept for future reuse, and must follow the standard
probabilistic rejection and residual recovery rules.

Suggested regular sampling interface:

```python
def sample_processed_logits(
    processed_logits: torch.Tensor,
    temperature: torch.Tensor,
    sampling_gumbel: torch.Tensor,
) -> torch.Tensor:
    ...
```

## 13. ModelRunnerV1 Integration Steps

### Step 1: Add an Input Builder

The input builder should:

- build `cu_num_logits` from `SpecDecodeMetadata`;
- reuse `idx_mapping`;
- fill `expanded_idx_mapping` and `expanded_local_pos`;
- return a lightweight `SimpleNamespace` or dataclass.

Keep this state local to the builder and avoid polluting `NPUModelRunner`.

### Step 2: Add Random Buffers and Stream State

Initialize the following in `NPUModelRunner.__init__`:

- a random stream;
- a ready event;
- a small buffer holder.

Do not allocate large tensors in `__init__`. Allocate lazily and reuse buffers
by slicing them to the actual runtime shape.

### Step 3: Prefetch Random Numbers Before Model Execution

Once the current step is known to be eligible for the new sampling path, call
`_prepare_sampling_noise` before model graph execution.

This check cannot depend on model output logits. It can depend on
`sampling_metadata`, `spec_decode_metadata`, batch size, vocab size, and
whether logprobs are requested.

### Step 4: Update `_sample`

Inside `_sample`:

1. check again whether the new path is available;
2. wait for the random ready event;
3. reuse the existing logits processing path to produce processed logits;
4. call `sample_processed_logits` for regular sampling;
5. call `sample_with_rejection` for speculative decoding;
6. return `SamplerOutput`.

### Step 5: Fallback

Fallback to the old path for:

- logprobs requests;
- unsupported logits processing parameters;
- unsupported speculative methods;
- reduced sampling paths that are not adapted yet;
- TP or global vocab cases whose token id semantics are unclear.

The fallback should be simple. Do not introduce a complex state machine.

## 14. Relationship With Reduced Sampling

This design is orthogonal to reduced sampling:

- reduced sampling optimizes the candidate space and depends on top-k/top-p or
  similar candidate pruning;
- this design optimizes random number generation, the sampling flow, and the
  rejection sampling operator.

Do not force the two designs to merge in the first stage. They can be combined
later when:

- logits processing naturally produces candidate ids and candidate scores;
- candidate ids can be passed into the sampler without an extra dense top-k;
- residual distribution semantics are clearly defined.

The prototype showed that running an extra dense top-k over full processed
logits and then doing candidate resampling is not profitable for large batch
sizes. Avoid repeating that direction.

## 15. Test Plan

### 15.1 Unit Tests

Add or extend unit tests for:

- regular sampling: greedy, temperature, top-k, top-p, and min-p;
- speculative sampling: probabilistic rejection sampling without
  `draft_probs`;
- fallback: logprobs, unsupported parameters, and unsupported speculative
  methods;
- input builder: different batch sizes, different speculative token counts,
  and different `cu_num_sampled_tokens`.

### 15.2 Correctness Tests

Compare with the old path using:

- fixed random tensors;
- fixed logits and draft tokens;
- sampled token ids;
- accepted length or `sample_counts`.

For random sampling, do not compare a single random output unless the random
tensors are fixed. Otherwise, verify distribution properties statistically.

### 15.3 Performance Tests

Keep the core settings from the prototype performance UT:

- vocab size: `151936`
- batch sizes: `1, 4, 16, 64, 128`
- warmup iterations: `5`
- measurement iterations: `20`
- the old path golden should also simulate async random number generation

The performance comparison method must be fixed so that random generation,
first-time compilation, or synchronization details are not incorrectly counted
as algorithmic speedup:

1. **Compare in one process**: construct fixed logits, draft tokens, and
   sampling metadata in one pytest case, then run old and new paths in the same
   process.
2. **Use old path as golden**: the old path should use the existing Ascend
   rejection sampler. For fairness, both paths should have async random number
   generation capability.
3. **Fix inputs and random tensor semantics**: target logits, draft tokens,
   temperature, top-k, top-p, acceptance uniform, and recovery Gumbel should
   have the same shapes or equivalent semantics.
4. **Check correctness before timing**: assert sampled tokens or accepted
   lengths before printing performance numbers.
5. **Warm up before timing**: warm up each path 5 times and call
   `torch.npu.synchronize()` before starting the timer.
6. **Use clear timing boundaries**: measure only the target sampling or
   rejection sampling path. Do not include test data construction, first-time
   tensor allocation, or logging.
7. **Report averages and speedup**: run 20 measured iterations and print
   `old_ms`, `new_ms`, and `old_ms / new_ms`. For PR data, repeat at least
   three rounds and report both average and worst round.
8. **Cover representative batch sizes**: at least cover `1, 4, 16, 64, 128`.
   The prototype showed that some ideas look good for small batches but regress
   for large batches.
9. **Separate regular sampling and speculative sampling**: do not reuse
   speculative sampling data as regular sampling evidence. Speculative sampling
   should also report different speculative token counts.

Suggested output format:

```text
sampling_optimization batch_size=64 vocab_size=151936
warmup_iters=5 num_iters=20 old_async_random=True new_async_random=True
old_ms=19.672 new_ms=17.125 speedup=1.149x
```

Add regular sampling performance tests for:

- no speculative decoding;
- no top-k, only temperature/top-p;
- both top-k and top-p enabled.

Acceptance criteria:

- speculative sampling keeps stable gains against the old async-random golden;
- regular sampling also shows benefit when speculative decoding is disabled;
- large batch sizes must not regress because of extra dense top-k or similar
  work.

## 16. Risks and Constraints

1. **Mathematical semantics**: the draft-free mode must be explicit
   in the operator name and call condition. Do not present it as full
   probabilistic rejection sampling.
2. **Random tensor lifetime**: allocate random tensors on the default stream
   and fill them on the random stream. Do not allocate tensors inside the async
   random stream.
3. **ModelRunnerV2 compatibility**: do not directly change the existing V2
   rejection sampler for the V1 transition path.
4. **Logits processing compatibility**: the new branch should consume processed
   logits and avoid understanding every logits processor internally.
5. **Large tensor memory**: `[num_reqs, vocab_size]` Gumbel buffers are large.
   Reuse buffers and slice them to the actual runtime shape.
6. **TP and global vocab semantics**: if logits are local vocab shards, token id
   semantics must be explicit. Fallback if correctness is unclear.

## 17. Non-goals

- Do not implement candidate resampling.
- Do not skip resampling.
- Do not support logprobs in the first stage.
- Do not reimplement all logits processing.
- Do not change the existing ModelRunnerV2 rejection sampler into a draft-free
  operator.
- Do not add a new user-facing switch.

## 18. Recommended Implementation Order

1. Add the draft-free sampling input builder.
2. Add the draft-free `draft_probs=None` rejection sampling operator.
3. Add external random tensors and validate correctness with synchronous
   generation first.
4. Move random generation before model execution and add random stream overlap.
5. Add the regular sampling path.
6. Extend supported logits processing parameters.
7. Add fallback and performance tests.
8. Revisit integration with reduced sampling only after the base path is
   stable.

## 19. Code Style Requirements

- Keep the path cohesive: input builder, random manager, and sampling kernel
  should be separated. `NPUModelRunner` should only orchestrate them.
- Do not write long defensive code. Fallback when conditions are not met, and
  raise an error for broken internal invariants.
- Do not add magic switches.
- Do not introduce unnecessary abstractions.
- Do not call `.item()` on device tensors in the hot path.
- Do not allocate large tensors in the hot path.
- Do not allocate tensors inside async streams.

## 20. References

- [vLLM speculative decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html)
- vLLM Ascend speculative decoding documentation:
  `docs/source/user_guide/feature_guide/speculative_decoding.md`
- vLLM Ascend MTP rejection sampling documentation:
  `docs/source/user_guide/feature_guide/Multi_Token_Prediction.md`
- Prototype branch: `codex/v1-upstream-rejection-sampler-poc`
- Prototype performance UT: `tests/ut/sample/test_upstream_rejection_sampler_poc.py`
