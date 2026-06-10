# Sampling Optimization Design

## Overview

This design keeps ModelRunnerV1 on its existing Ascend sampler path and applies
three focused optimizations inside the existing components:

- regular random sampling uses Gumbel argmax instead of softmax plus exponential
  division;
- random tensors are generated on a separate NPU stream before sampling when
  possible;
- speculative decoding uses a native NPU rejection sampling operator that
  samples the bonus token only after draft acceptance is known.

The design intentionally does not introduce a V2 bridge, `FastSampler`, V2-style
input packing, or request-state adaptation. Regular sampling still enters
`AscendSampler`, and speculative decoding still enters `AscendRejectionSampler`.

## Compatibility

The optimization is internal and does not add a user-facing flag or environment
variable. Existing behavior is preserved for:

- logprobs and `logprob_token_ids`;
- reduced sampling;
- lmhead tensor parallel sampling;
- discarded requests;
- legacy `enable_async_exponential`.

When `enable_async_exponential=True`, ModelRunnerV1 keeps using the original V1
async exponential behavior. The new Gumbel path does not intercept that mode.
`enable_reduce_sample=True` and sampled-token logprobs are treated as an invalid
combination and fail with an assertion instead of falling back silently.

## Regular Sampling

`AscendTopKTopPSampler.forward_native` remains the regular random sampling
entry. It still applies top-k/top-p first and still returns processed logits or
processed logprobs when requested.

The random token selection changes from:

```text
softmax(processed_logits) / exponential_random -> argmax
```

to:

```text
processed_logits + gumbel -> argmax
```

For reduced sampling, the same rule is applied to the candidate logits returned
by `apply_top_k_top_p`; the sampled candidate position is mapped back through
candidate token ids. Gumbel noise is used only for token selection and does not
modify logits returned for logprobs.

All-greedy batches return before random sampling, so no Gumbel tensor is
generated for them.

## Async Random Generation

`NPUModelRunner` prepares random tensors before the forward pass:

- regular path calls `AscendSampler.prepare_async_gumbel`;
- speculative path calls `AscendRejectionSampler.prepare_async_rejection_random`;
- all-greedy batches skip random preparation;
- `enable_async_exponential=True` calls the original
  `AscendSampler.do_async_exponential`.

The optimized sampling branches consume prefetched tensors prepared by
ModelRunnerV1. Benchmarks call the same preparation methods before timed sampler
execution so random generation is not measured as part of the sampling kernels.

## Speculative Decoding

`AscendRejectionSampler.forward` keeps the existing V1 entry, but the optimized
branch uses the V2-style bonus flow:

1. gather draft and bonus logits in `metadata.logits_indices` order;
2. reuse the existing V1 draft helper for draft rows and the existing V1 sampler
   helper with `predict_bonus_token=True` for bonus rows, then reassemble the
   processed rows in joint-logits order;
3. apply speculative sampling constraints to draft and bonus rows together;
4. run rejection sampling over draft rows;
5. if every draft token is accepted for a request, sample the bonus token from
   that request's bonus row using the prefetched recovery Gumbel row;
6. otherwise insert the recovered token at the rejection position;
7. build logprobs with the existing `_get_logprobs_tensors` path.

Step 4 uses the native NPU rejection operator when the required V1 tensors are
available. The operator consumes compact `metadata.draft_token_ids` and
draft+bonus logits rows; it samples the bonus row only after all draft tokens for
that request are accepted. The optimized path no longer calls `AscendSampler` to
pre-sample bonus tokens.

Reduced sampling is supported by passing candidate token ids into the operator.
Dense vocab mode indexes logits directly by token id; reduced mode searches the
candidate token ids and returns global token ids. Sampled-token logprobs are
supported in dense vocab mode for both raw and processed logprobs modes by
reusing the existing V1 logprobs tensor builder.

All-greedy speculative batches also use the optimized operator. They do not
prepare random tensors; the operator uses greedy argmax for draft verification
and bonus sampling. In reduced sampling mode, the optimized path first builds
top-1 tensor-parallel candidates with global token ids and passes those
candidates into the operator.

Penalties, bad words, allowed-token masks, and V1 non-argmax logits processors
are handled before the optimized operator by reusing the existing V1 helpers
separately for draft and bonus rows. This keeps the request-history semantics
identical to the original V1 path while still allowing the optimized rejection
operator to own bonus sampling.

## Fallbacks

Fallback is used for cases where the optimized operator should not own the
behavior:

- `enable_async_exponential=True`;
- speculative calls outside the ModelRunnerV1 optimized entrance;
- draft-probability rejection;
- argmax-invariant logits processors, because those are applied by the regular
  sampler after temperature scaling and before top-k/top-p rather than by the V1
  logits-processor helpers reused by this design;
- `enable_reduce_sample=True` together with sampled-token logprobs, which is
  rejected with an assertion.

The fallback path is still the existing V1 Ascend sampler or rejection sampler.

## Testing

Unit tests should cover:

- Gumbel sampling does not mutate logits used for logprobs;
- reduced regular sampling maps candidate positions back to token ids;
- `enable_reduce_sample=True` with sampled-token logprobs raises an assertion;
- optimized rejection is not disabled by raw or processed logprobs in dense vocab
  mode;
- optimized rejection handles all-greedy speculative batches, including reduced
  sampling with global token ids;
- optimized rejection matches the V1 fallback when V1 logits processors such as
  allowed-token masks are active;
- `enable_async_exponential=True` disables the optimized rejection branch;
- optimized rejection samples bonus tokens from bonus logits after draft
  acceptance;
- ModelRunnerV1 prepares regular Gumbel random tensors and speculative rejection
  random tensors in the expected cases;
- all-greedy batches do not prepare random tensors.

Benchmarks should compare the existing sampler entry without the optimized
operator against the optimized sampler entry.

## Benchmark Results

The benchmark tables below report mean sampling latency in milliseconds.
`no_async_exponential` is the V1 path without prefetched exponential random
tensors; `async_exponential` is the V1 path with prefetched exponential random
tensors; `optimized` is the Gumbel/rejection-operator path.

### Speculative Decoding

| Batch size | no_async_exponential (ms) | async_exponential (ms) | optimized (ms) | Speedup vs no_async_exponential | Speedup vs async_exponential |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.4993 | 2.1779 | 2.0710 | 1.2068x | 1.0516x |
| 8 | 2.6355 | 2.4032 | 2.0417 | 1.2908x | 1.1771x |
| 32 | 5.7979 | 5.3139 | 4.2505 | 1.3641x | 1.2502x |
| 64 | 10.5919 | 9.5223 | 7.6919 | 1.3770x | 1.2380x |
| 96 | 15.9505 | 14.3173 | 11.1793 | 1.4268x | 1.2807x |

### Regular Sampling

| Batch size | no_async_exponential (ms) | async_exponential (ms) | optimized (ms) | Speedup vs no_async_exponential | Speedup vs async_exponential |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6201 | 0.5966 | 0.5510 | 1.1254x | 1.0828x |
| 8 | 0.7555 | 0.6113 | 0.5607 | 1.3476x | 1.0903x |
| 32 | 1.5284 | 1.0148 | 0.9696 | 1.5764x | 1.0466x |
| 64 | 2.8859 | 1.8548 | 1.7807 | 1.6207x | 1.0416x |
| 96 | 4.4680 | 2.8327 | 2.7114 | 1.6478x | 1.0447x |
