# Swift Policy Integration for Model Runner V2 EPLB

## Status

This document records the agreed two-stage plan for integrating the existing
Ascend Swift EPLB placement policy with Model Runner V2. It is a design and
experiment plan, not a description of the current implementation.

The immediate objective is to determine whether the Swift placement policy can
reduce Model Runner V2's expert migration volume to the Model Runner V1 range.
Communication, asynchronous scheduling, workspace commit, and routing-table
refresh behavior must remain unchanged during the first-stage experiment so
that any performance change can be attributed to the policy.

## Motivation

In the equivalent Qwen3-30B-A3B W8A8 experiment, the Model Runner V1 and Model
Runner V2 baselines without EPLB were nearly identical. The large regression
was isolated to Model Runner V2 EPLB:

| Metric | Model Runner V1 Swift | Model Runner V2 default |
| --- | ---: | ---: |
| Remote experts received | 53 | 12,482 |
| Remote bytes received | 0.233 GiB | 54.97 GiB |
| Active remote layer events | 19 | 193 |

Model Runner V2 therefore moved approximately 235.5 times as many bytes as
Model Runner V1 in the measured windows. Its Gloo transfer and main-thread
commit costs were measured under this abnormal migration volume. Those costs
must be re-profiled after the policy produces a bounded delta; optimizing them
first would risk optimizing work that should not exist.

The existing Swift policy already has the desired behavior: it retains the old
placement by default, skips sufficiently balanced layers, accepts only
improving layer placements, limits pairwise cross-rank communication, and
preserves same-rank slots where possible.

## Goals

1. Run the same Swift algorithm used by Model Runner V1 from the Model Runner
   V2 EPLB policy call site.
2. Make the first experiment a policy-only change with the smallest practical
   implementation and review surface.
3. Preserve the current Model Runner V2 load collection, async worker,
   transfer, map commit, and routing-table lifecycle.
4. Keep the upstream default policy as the default and make Swift an explicit
   Ascend opt-in until validation is complete.
5. Use the first-stage data to decide whether a shared policy core and a
   separate policy process are justified.

## Non-goals for Stage 1

Stage 1 must not:

- replace Gloo with HCCL;
- modify `vllm.distributed.eplb.async_worker`;
- change the single-buffer producer/consumer protocol;
- add an unchanged-layer fast path;
- optimize `_move_to_workspace`, map commit, or routing-table refresh;
- change `window_size`, `step_interval`, or load-collection semantics;
- rewrite or improve the Swift algorithm;
- introduce a new multiprocessing policy executor.

These changes would make the experiment harder to attribute. They remain
possible follow-up work after migration volume is corrected and the pipeline is
profiled again.

## Existing Interface Mismatch

The legacy Swift entry point is
`vllm_ascend/eplb/core/policy/policy_swift_balancer.py::SwiftBalanceEplb.rebalance_experts`.
It accepts:

- a placement tensor shaped `[layers, ranks, slots_per_rank]`;
- a physical-slot load tensor with the same shape;
- and returns `(change, priority, new_placement)`.

The Model Runner V2 policy contract is
`vllm.distributed.eplb.policy.abstract.AbstractEplbPolicy.rebalance_experts`.
It receives:

- global logical-expert load shaped `[layers, logical_experts]`;
- the old flattened physical-to-logical map shaped
  `[layers, physical_replicas]`;
- replica and topology counts;
- and returns a flattened CPU placement tensor shaped
  `[layers, physical_replicas]`.

Model Runner V2 has already summed the loads of all physical replicas into
logical-expert load before invoking the policy. The adapter must not aggregate
those loads a second time.

## Stage 1: Minimal Compatibility Adapter

### Decision

Stage 1 adds a thin Model Runner V2 policy adapter around the existing
`SwiftBalanceEplb`. The existing Swift implementation remains unchanged. The
adapter reconstructs the legacy input representation, calls the legacy policy,
and converts its placement result back to the Model Runner V2 contract.

This intentionally accepts a small amount of CPU-side representation overhead
in exchange for preserving the exact policy implementation used by Model
Runner V1. The converted arrays contain only placement and load metadata and
are created once per rebalance cycle, not per request, forward, or MoE layer.

### Data flow

The adapter receives the Model Runner V2 inputs:

```text
logical_load:                  [L, E]
old_physical_to_logical_map:   [L, P]
```

It reshapes the old map to `[L, R, S]`, where `P = R * S`. For each logical
expert, it counts the replicas in the old placement. Each physical slot is then
assigned an equal share of its logical expert's load:

```text
slot_load[layer, rank, slot]
    = logical_load[layer, expert] / replica_count[layer, expert]
```

The legacy Swift implementation immediately sums the slot loads belonging to
the same logical expert. It therefore reconstructs the original Model Runner
V2 logical load, subject only to normal floating-point representation. The
adapter passes `is_node_redundant=False` in Stage 1 to match the Model Runner V1
behavior used by the reference experiment.

The returned placement is converted to a contiguous CPU tensor and flattened
to `[L, P]`. Model Runner V2's existing `transfer_layer` logic remains the sole
owner of placement differencing and weight movement.

### Proposed code ownership

| Name or file | Action | Responsibility |
| --- | --- | --- |
| `SwiftBalanceEplb` | Keep unchanged | Existing Model Runner V1 policy and reference behavior |
| `vllm_ascend/distributed/eplb_policy.py` | Add | Model Runner V2-to-legacy Swift compatibility adapter |
| `SwiftEplbPolicyAdapter` | Add | Implement the upstream V2 policy call signature and data conversion |
| `AscendEplbState.add_model` | Extend narrowly | Select the adapter after upstream state initialization |
| `placement_policy` | Add | Explicit Ascend policy override; unset means upstream default |

`AscendEplbState` must select the effective policy after
`super().add_model()` because upstream `EplbState.add_model()` resets
`self.policy` from its own registry. The same path is used by
`EplbState.from_mapping()`, so the selection remains consistent across normal
initialization and mapping-based initialization.

The implementation must not mutate the upstream `EPLB_POLICIES` global, patch
`run_rebalance_experts`, or maintain a second placement state. Upstream
`EplbState` remains the only Model Runner V2 placement owner.

### Configuration

Add the following optional Ascend-specific field under
`additional_config.eplb_config`:

```text
placement_policy: null | "swift"
```

When it is unset, Model Runner V2 continues to use the upstream-selected
policy. When it is `"swift"`, `AscendEplbState` selects the Swift adapter and
logs the effective policy once during initialization.

The legacy numeric `eplb_policy_type=2` must not be reused for Model Runner V2.
Its existing default is already `2`; reusing it would silently switch every
Model Runner V2 EPLB deployment to Swift and imply support for legacy policies
that have not been adapted.

Stage 1 supports only fixed-topology EPLB. `placement_policy="swift"` combined
with elastic EP must fail during configuration validation rather than silently
falling back to another policy. The current asynchronous Ascend EPLB path
already rejects elastic EP.

### Swift semantics to preserve

The Stage 1 adapter must preserve the existing Swift defaults and decisions:

- `imbalance_threshold = 1.01`;
- `increment = 0.01`;
- `num_max_com = 1`;
- `max_swap_times = 100`;
- balanced layers keep their old placement;
- a proposed layer is accepted only when it improves that layer's imbalance;
- same-rank slots are preserved where possible;
- layers are transferred in the existing Model Runner V2 order;
- the legacy `priority` result does not reorder layers;
- the legacy aggregate `change` flag does not gate the returned map.

The final point is required for parity. Model Runner V1 computes an aggregate
five-percent `change` flag, but its worker ignores the flag and consumes the
returned placement. Enforcing that flag in Model Runner V2 would implement a
different policy.

### Execution context and GIL risk

The legacy Model Runner V1 policy runs in a subprocess. Model Runner V2 invokes
its policy from the async worker thread. Swift contains Python loops and set
operations, so running it in that thread may contend for the GIL even though it
does not run on the main thread.

Stage 1 must record CPU `policy_compute_ms` without adding an NPU event or
device synchronization. A new process must not be introduced before this is
measured. If policy computation creates visible main-thread latency spikes,
that result becomes an explicit trigger for Stage 2's policy executor work.

### Stage 1 tests

Unit tests must cover:

1. Adapter output shape, dtype, CPU residency, and contiguity.
2. The input load and old placement are not modified in place.
3. Logical load expanded to slot load and re-aggregated by legacy Swift retains
   the original logical load within an appropriate numerical tolerance.
4. A balanced placement remains unchanged.
5. Four redundant experts remain valid and every logical expert retains at
   least one replica.
6. Repeated calls with identical inputs produce identical placement.
7. The adapter result matches direct legacy Swift execution for equivalent
   placement and load fixtures.
8. `placement_policy` defaults to no override, accepts `"swift"`, and rejects
   unknown values and unsupported elastic-EP combinations.
9. Normal construction and `from_mapping()` select the same effective policy.

### Stage 1 benchmark

Run the same Model Runner V2 case twice while changing only the policy:

| Group | Policy | Transport | Other settings |
| --- | --- | --- | --- |
| A | upstream default | Gloo | unchanged |
| B | Swift adapter | Gloo | identical to A |

Record:

- imbalance before and after policy execution;
- policy compute time;
- changed layers;
- cross-rank moved experts;
- moved bytes;
- active remote layer events;
- TTFT, TPOT, request E2E, benchmark wall time, and throughput;
- buffer-ready, ready-to-consume, main-thread consume/commit, and full cycle
  duration using the existing asynchronous probe methodology.

The logging and probe path must remain asynchronous and must not add NPU
synchronization to serving.

### Stage 1 success and exit criteria

Stage 1 succeeds when:

- moved experts and bytes return to the Model Runner V1 order of magnitude;
- Swift provides a material imbalance improvement over the old placement;
- its balance result is acceptably close to the upstream default policy for the
  test load;
- policy computation does not introduce a new visible TTFT or TPOT stall;
- inference correctness and expert-map consistency are preserved.

After success, the Gloo and main-thread commit timings must be measured again.
The earlier 91.4 ms transfer and 29.6 ms consume values were collected under a
235.5-times larger byte volume and must not be treated as fixed costs.

If Stage 1 does not reduce migration volume, stop. Do not proceed to transport
or commit optimization; first compare the exact V1 and V2 load/map snapshots
and resolve the policy-input semantic difference.

## Stage 2: Shared Swift Core and Production Hardening

Stage 2 is conditional on Stage 1 proving that Swift materially improves Model
Runner V2. It removes the temporary representation adapter and turns Swift into
a reusable, runner-independent policy implementation.

### Shared core

Extract a pure CPU core from the existing legacy policy:

| Name or file | Action | Responsibility |
| --- | --- | --- |
| `vllm_ascend/eplb/core/policy/swift_core.py` | Add | Runner-independent Swift placement algorithm |
| `SwiftBalancer` | Add | Accept logical load, old placement, and explicit topology |
| `SwiftPlan` | Add | Return placement plus policy diagnostics |
| `SwiftBalanceEplb` | Retain as thin wrapper | Convert legacy V1 physical-slot load and preserve its return contract |
| `SwiftEplbPolicy` | Replace Stage 1 adapter | Thin V2 wrapper calling the shared logical-load core directly |

The shared core accepts logical load `[L, E]` and placement `[L, R, S]`
directly. Model Runner V1 performs its existing physical-to-logical load
aggregation before calling the core. Model Runner V2 passes its already
aggregated logical load without creating slot-load arrays.

The core must:

- remove direct `torch_npu` imports and NPU API calls;
- take `num_ranks` and `num_nodes` explicitly;
- derive ranks per node from the supplied topology rather than
  `torch.npu.device_count()`;
- avoid class-level mutable state such as `DynamicTable`;
- avoid mutating input arrays;
- use stable ordering and remove unordered output construction;
- allocate only the proposed placement and small policy diagnostics once per
  rebalance cycle;
- preserve Stage 1/V1 placement results through golden tests before making any
  algorithmic improvement.

Any stricter global gain threshold, total movement budget, or revised
multi-node behavior is a separate policy change. It must not be hidden inside
the extraction refactor.

### Policy execution process

Do not add a process merely because Model Runner V1 used one. Add a persistent
policy executor only if Stage 1 proves that Swift's Python execution causes GIL
contention or unacceptable policy latency.

If required, the process boundary must live in a policy-execution component,
not in `SwiftBalancer`. It sends only CPU logical-load and placement snapshots
and returns a CPU placement plan. The V2 async worker may wait for that result;
the serving main thread must not. Process startup, shutdown, worker failure,
timeout, and stale-result handling must be part of the state lifecycle.

### Upstream extension point

The long-term upstream shape is an official policy factory or registration
hook. Once available, vLLM Ascend should register `SwiftEplbPolicy` through
that interface and delete the local post-`add_model()` selection override.

Upstream extension work is not a prerequisite for Stage 1 and must not expand
the first experimental patch.

### Stage 2 validation

Stage 2 must rerun the Stage 1 golden policy fixtures and performance case. For
the same logical load and old placement, the shared core must produce the same
placement and movement counters as the successful Stage 1 adapter before any
additional optimization is considered.

## Follow-up Decision Tree

After Stage 1 or Stage 2 corrects migration volume, re-profile the remaining
pipeline in this order:

1. If main-thread consume/commit remains material, split weight copy, map
   commit, and Ascend routing-table refresh timings and optimize the dominant
   component.
2. If Gloo remains material for the reduced byte volume, evaluate direct HCCL
   communication independently.
3. If unchanged layers still pay meaningful commit overhead, add an
   unchanged-layer fast path.
4. If ready-to-consume delay limits rebalance freshness, separate forward
   overlap from readiness collective time before changing the single-buffer
   protocol.

None of these follow-ups should be bundled into the Swift integration patch.

## Final Decision

The implementation proceeds in two stages:

1. **Minimal validation:** keep the legacy Swift implementation unchanged and
   connect it to Model Runner V2 through a thin, explicit Ascend adapter.
2. **Production cleanup:** only after the policy hypothesis is proven, extract
   a shared logical-load core, remove legacy device assumptions, and introduce
   a policy process or upstream registration hook only when measurements
   justify them.

This order minimizes the first change, preserves a trustworthy Model Runner V1
reference, and prevents unrelated communication or state-machine work from
obscuring the policy experiment.
