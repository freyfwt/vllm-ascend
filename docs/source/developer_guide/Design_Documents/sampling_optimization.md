# Sampling Optimization Design

## Overview

This design adds a lightweight NPU fast sampling path for ModelRunnerV1. The
goal is to reduce sampling overhead without introducing a ModelRunnerV2 input
adapter inside the V1 runner.

The fast path is intentionally narrow:

- regular sampling uses Gumbel sampling on processed logits;
- speculative decoding uses an NPU rejection sampling operator;
- random tensors are generated on a separate NPU stream when they are needed;
- unsupported cases fall back to the existing V1 Ascend sampler and rejection
  sampler.

The implementation must not change the semantics of the existing
`enable_async_exponential` option. When `enable_async_exponential=True`, the
new fast path is disabled and ModelRunnerV1 keeps using the original Ascend
sampling path.

## Motivation

The original V1 sampling path is functionally complete, but sampling can become
a visible part of decode latency, especially with large vocabularies and
speculative decoding. The optimized path focuses on the operations that are
known to be expensive on NPU:

- random number generation serialized with sampling;
- dense-vocabulary rejection sampling work;
- Python-side bridge logic around sampler state;
- extra adaptation layers that are not needed by ModelRunnerV1.

The previous bridge-style design tried to reuse upstream GPU/V2 sampler
structures from the V1 runner. That approach made the implementation difficult
to maintain because it introduced V2-style input packing, request state
construction, and sampler inheritance that were not native to the V1 execution
path. The new design removes that bridge and keeps only the core NPU
optimizations.

## Scope

The fast path is owned by ModelRunnerV1 and uses existing V1 runtime state. It
does not add a public user-facing option and does not require a new environment
variable.

The main code boundaries are:

| Component | Responsibility |
| --- | --- |
| `NPUModelRunner._can_fast_sample` | Decide whether the current step can use the fast path. |
| `NoiseManager` | Allocate reusable random buffers and fill them on a separate NPU stream. |
| regular fast sampler | Sample one token from processed logits with argmax or Gumbel argmax. |
| `rejection_ops.py` | Run speculative decoding rejection sampling with reusable NPU workspaces. |
| existing Ascend samplers | Remain the fallback for unsupported cases. |

The design does not cover the separate async scheduling bubble caused by
`_copy_valid_sampled_token_count`. That issue should be fixed independently.

## Fast Path Selection

`NPUModelRunner._can_fast_sample` is the single entry check. It returns `False`
for unsupported or legacy paths so the runner falls back to the existing V1
sampler implementation.

The fast path is disabled when any of the following is true:

- the fast sampler is not initialized for the current runner;
- `enable_async_exponential=True`;
- logprobs are requested;
- `logprob_token_ids` are requested;
- the sampler must compute NaNs;
- lmhead tensor parallel sampling is enabled;
- reduced sampling is enabled;
- the current step contains discarded requests;
- speculative decoding is active but the rejection fast path is unavailable.

The `enable_async_exponential` rule is a compatibility guard. That option
belongs to the original V1 Ascend sampler path. Enabling it means the user has
explicitly selected the legacy async exponential behavior, so the new Gumbel
fast path must not intercept the request.

## Random Number Generation

`NoiseManager` owns random buffers, an NPU stream, and a ready event. It prepares
the random tensors before sampling, records the ready event after filling them,
and lets the sampling stream wait only when a real random tensor was generated.

The manager must short-circuit all-greedy batches before launching any random
operation:

- all-greedy regular sampling does not allocate or fill a Gumbel tensor;
- all-greedy speculative decoding does not fill acceptance uniform or recovery
  Gumbel tensors;
- no event is recorded for all-greedy random work, so the sampling stream does
  not wait on an empty random task.

When random sampling is required, the buffers are reused across iterations and
sliced to the current shape. The random stream waits for the current stream
before overwriting a reused buffer, then fills the buffer and records an event.
The sampling stream waits on that event before consuming the random tensor.

## Regular Sampling

The regular fast path consumes processed logits and the existing V1 sampling
metadata. It does not construct a V2 `GpuInputBatch`, `RequestState`, or
sampler state object.

The sampling rule is:

- if all requests are greedy, return `argmax(processed_logits)`;
- if all requests are random, return `argmax(processed_logits + gumbel)`;
- for mixed batches, compute both greedy and random tokens, then select by the
  per-request temperature.

The Gumbel tensor is generated only for non-greedy batches. The fast path should
avoid host synchronization and avoid per-step allocation in the sampling hot
path.

## Speculative Decoding

The speculative fast path uses a native NPU rejection sampling operator. It
consumes V1 speculative decoding metadata directly instead of adapting the
batch to upstream GPU/V2 sampler classes.

The operator receives:

- target model logits for draft and bonus positions;
- draft token ids from the existing input batch;
- cumulative logits offsets from speculative metadata;
- request index mappings and local draft positions;
- per-request temperature;
- acceptance uniform values when probabilistic acceptance is needed;
- recovery Gumbel values when a random recovery token is needed;
- a reusable `RejectionWorkspace`.

The rejection path returns sampled token ids and the number of accepted tokens
for each request. Unsupported speculative configurations fall back to the
existing V1 rejection sampler.

## Removed Bridge Design

The implementation should not keep the V2 bridge layer that existed in the
prototype. In particular, the final design removes:

- V2-style input structure construction;
- `GpuBatchView`;
- `RequestState` construction for the V1 runner;
- request state update and refresh logic that only served the bridge;
- inheritance from upstream `GpuSampler`;
- inheritance from upstream `GpuRejectionSampler`.

The V1 runner should call small NPU-owned helper functions directly. This keeps
the fast path close to the state that already exists in ModelRunnerV1 and makes
fallback behavior easy to audit.

## Fallback Behavior

Fallback is part of the design, not an error path. If a request needs behavior
outside the fast path contract, ModelRunnerV1 calls the existing Ascend sampler
or rejection sampler.

Fallback preserves:

- existing `enable_async_exponential` behavior;
- logprob behavior;
- reduced sampling behavior;
- lmhead tensor parallel behavior;
- any speculative decoding case not covered by the NPU rejection fast path.

This keeps the fast path performance-focused while avoiding semantic drift in
less common sampling modes.

## Testing

Unit tests should cover both fast path behavior and fallback decisions:

- `_can_fast_sample` returns `False` when `enable_async_exponential=True`;
- `_can_fast_sample` returns `False` for logprobs, reduced sampling, lmhead TP,
  discarded requests, and unsupported speculative metadata;
- all-greedy regular sampling does not request a Gumbel tensor;
- random regular sampling uses Gumbel argmax;
- mixed regular sampling respects per-request temperature;
- speculative rejection sampling matches the reference sampler for supported
  draft-free cases;
- unsupported speculative cases fall back to the existing rejection sampler.

Benchmarks should compare the original V1 path with the V1-native fast path.
They should not describe the removed V2 bridge as a supported implementation
strategy.

## Documentation

User-facing configuration documentation for `enable_async_exponential` should
continue to describe the original V1 async exponential sampler behavior. This
design document should describe the fast path as an internal ModelRunnerV1
optimization that respects that option by falling back when it is enabled.

No new environment variable is introduced.
