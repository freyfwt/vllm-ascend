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

- regular path calls `AscendSampler.do_async_gumbel`;
- speculative path calls `AscendRejectionSampler.do_async_rejection_random`;
- all-greedy batches skip random preparation;
- `enable_async_exponential=True` calls the original
  `AscendSampler.do_async_exponential`.

The sampler consumes a prefetched tensor when it is available and falls back to
generating random tensors on the sampling stream when the caller did not prefetch
or the prefetched shape is too small. This keeps direct sampler tests and
benchmarks usable without ModelRunnerV1.

## Speculative Decoding

`AscendRejectionSampler.forward` keeps the existing V1 entry, but the optimized
branch uses the V2-style bonus flow:

1. gather draft and bonus logits in `metadata.logits_indices` order;
2. apply speculative sampling constraints to draft and bonus rows together;
3. run rejection sampling over draft rows;
4. if every draft token is accepted for a request, sample the bonus token from
   that request's bonus row using the prefetched recovery Gumbel row;
5. otherwise insert the recovered token at the rejection position;
6. build logprobs with the existing `_get_logprobs_tensors` path.

Step 4 uses the native NPU rejection operator when the required V1 tensors are
available. The operator consumes compact `metadata.draft_token_ids` and
draft+bonus logits rows; it samples the bonus row only after all draft tokens for
that request are accepted. The optimized path no longer calls `AscendSampler` to
pre-sample bonus tokens.

Reduced sampling is supported by passing candidate token ids into the operator.
Dense vocab mode indexes logits directly by token id; reduced mode searches the
candidate token ids and returns global token ids.

The optimized branch is intentionally limited to the hot path without penalties,
bad words, allowed-token masks, or custom logits processors. Those features
remain on the V1 fallback path because V1 handles bonus logits processors with
`predict_bonus_token=True`, which requires different request history semantics
from draft-row processing.

## Fallbacks

Fallback is used for cases where the optimized operator should not own the
behavior:

- `enable_async_exponential=True`;
- speculative calls outside the ModelRunnerV1 optimized entrance;
- all-greedy speculative batches, where the existing path keeps the cheaper
  greedy-specific rejection flow;
- draft-probability rejection, penalties, bad words, allowed-token masks, or
  custom logits processors;
- processed-logprobs mode when sampled-token logprobs are requested.

The fallback path is still the existing V1 Ascend sampler or rejection sampler.

## Testing

Unit tests should cover:

- Gumbel sampling does not mutate logits used for logprobs;
- reduced regular sampling maps candidate positions back to token ids;
- optimized rejection is not disabled by logprobs or reduced sampling;
- `enable_async_exponential=True` disables the optimized rejection branch;
- optimized rejection samples bonus tokens from bonus logits after draft
  acceptance;
- ModelRunnerV1 prepares regular Gumbel random tensors and speculative rejection
  random tensors in the expected cases;
- all-greedy batches do not prepare random tensors.

Benchmarks should compare the existing sampler entry without the optimized
operator against the optimized sampler entry.
