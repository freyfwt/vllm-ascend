# 采样优化方案设计

## 1. 背景

RFC 9269 的核心诉求可以概括为：降低 vLLM Ascend 解码阶段采样与投机解码拒绝采样的端到端开销，为主线切换到 ModelRunnerV2 扫清性能障碍，同时在 ModelRunnerV1 仍作为主要执行路径期间提前拿到收益。

当前 ModelRunnerV1 的采样链路主要复用 Ascend 侧已有 sampler 和 rejection sampler。它能跑通功能，但在随机采样、Gumbel 生成、拒绝采样恢复分布计算等环节存在多次 dense vocab 访问、随机数生成与模型执行串行、以及采样代码与 logits processing 代码耦合较强的问题。ModelRunnerV2 的采样设计更接近上游 vLLM 的新流程，拒绝采样 kernel 结构也更利于把随机数外置、把采样输入整理成统一格式、并在后续继续向 V2 主线平滑迁移。

本方案的目标不是继续维护一条长期分叉的 ModelRunnerV1 专用采样实现，而是在 ModelRunnerV1 内接入一套与 ModelRunnerV2 对齐的采样/拒绝采样路径。这样可以先在 V1 获得性能收益，后续主线迁移到 V2 时复用同一套核心算子与输入组织方式。

## 2. 方案价值

1. 基于 ModelRunnerV2 的采样方案可以显著优化 Gumbel 相关路径，缓解当前阻碍主线切换到 V2 的关键性能问题。
2. ModelRunnerV2 的拒绝采样流程整体性能更优，将其接入 ModelRunnerV1 后，可以在 V2 完全替换 V1 前提前获得收益。
3. 直接围绕 V2 的拒绝采样流程做优化，后续版本可以把同一套算子与数据结构平滑前移到 V2，避免一次性重写。
4. 相比 reduced sampling，本方案即使没有 `top_k` 参数也可以通过随机数外置、拒绝采样流程优化获得收益。
5. 相比 reduced sampling，本方案不感知 logits processing 的内部实现，不需要侵入 top-k/top-p/min-p/penalty 等处理逻辑，因此更灵活、更可维护。
6. 理论上本方案可以与 reduced sampling 叠加：先用 reduced sampling 缩小候选空间，再使用本方案优化随机数、拒绝采样与输出整理，后续仍有进一步收益空间。

## 3. 术语和现状

- **普通采样**：不开投机解码时，从目标模型 logits 经 logits processing 后采样一个 token。
- **投机解码**：draft model 或 MTP 等 proposer 先生成若干 draft token，目标模型一次验证多个 token，再通过拒绝采样决定接受长度和恢复 token。
- **拒绝采样**：比较 target distribution 和 draft distribution，接受满足概率测试的 draft token；遇到拒绝时从 residual distribution 中重采样。
- **Gumbel 采样**：用 `argmax(logits + gumbel)` 表达随机采样，适合放入融合 kernel 或外置随机数路径。
- **reduced sampling**：通过 top-k/top-p 等方式减少后续采样候选数量。它依赖 logits processing 的结果和候选集合，本方案不把它作为前置条件。

## 4. 穿刺实验总结

### 4.1 实验设置

穿刺实验在真实 NPU 环境运行，性能 UT 使用：

- vocab size: `151936`
- batch sizes: `1, 4, 16, 64, 128`
- speculative tokens: `4`
- warmup: `5` 次
- measurement: `20` 次平均
- old path golden: 现有 Ascend rejection sampler，并模拟旧路径也具备异步随机数能力
- new path: ModelRunnerV1 内接入 V2 风格 probabilistic rejection sampler，外置 acceptance uniform 与 resample Gumbel

测试入口：

```bash
python -m pytest -q -s tests/ut/sample/test_upstream_rejection_sampler_poc.py
```

### 4.2 最终有效穿刺结果

清理掉 candidate resample 后，dense vocab resample 路径的数据如下：

| Batch size | Old async random ms | New external random ms | Speedup |
|---:|---:|---:|---:|
| 1 | 2.967 | 2.264 | 1.310x |
| 4 | 2.931 | 2.251 | 1.302x |
| 16 | 5.595 | 4.194 | 1.334x |
| 64 | 19.672 | 17.125 | 1.149x |
| 128 | 38.410 | 33.841 | 1.135x |

结论：

- 在 old path 也模拟异步随机数的情况下，new path 仍有稳定收益。
- 小 batch 收益更明显，大 batch 仍有 7% 到 15% 左右收益。
- 当前收益主要来自 V2 风格 rejection sampler 的流程组织、外置随机数、减少 Python 侧中间构造和更紧凑的采样流程。

### 4.3 已放弃或仅用于定位的实验

#### candidate resample

曾尝试把 dense vocab resample 改为 candidate resample，即先对 processed logits 做 `topk`，再只在 top-k 候选内重采样。该方向已放弃。

原因是新增的 `torch.topk(processed_logits, k)` 需要扫描 `num_logits * vocab`，而省掉的 dense resample 只扫描 `num_reqs * vocab`。在 speculative tokens 为 4 时，`num_logits` 约等于 `5 * num_reqs`，新增 top-k 成本超过 resample 节省，尤其在 batch 64 和 128 时性能明显劣化。

candidate 版本数据：

| Batch size | Old async random ms | Candidate new ms | Speedup |
|---:|---:|---:|---:|
| 1 | 3.019 | 2.410 | 1.253x |
| 4 | 3.009 | 2.399 | 1.254x |
| 16 | 5.643 | 4.469 | 1.263x |
| 64 | 19.668 | 18.122 | 1.085x |
| 128 | 38.443 | 35.907 | 1.071x |

因此正式实现不应引入 candidate resample，除非未来能在拒绝后只对实际 resample 行做候选选择，或能复用 logits processing 已经产生的候选集合。

#### skip resample

曾跳过 resample 做诊断，性能更好，但不满足拒绝采样数学语义，不能作为实现方案。它只用于定位 resample 阶段开销。

#### 不使用 draft probabilities

从上游行为和 POC 结果看，在 ModelRunnerV1 的当前需求中，可以先支持不传 `draft_probs` 的路线。这能避免保存完整 draft logits/draft probabilities 的巨大内存和拷贝成本。正式方案应新写 V1 专用算子，显式支持 `draft_probs=None`，避免修改 V2 原有功能语义。

## 5. 设计目标

1. ModelRunnerV1 在满足约束时默认走新采样分支，不再新增用户开关。
2. 新分支复用 ModelRunnerV2 的输入组织和拒绝采样思想，但不直接破坏 V2 原有算子。
3. 支持不开投机解码的普通采样，也支持 probabilistic speculative decoding 的拒绝采样。
4. 支持现有 logits processing 参数，最终覆盖 temperature、top-k、top-p、min-p、presence penalty、frequency penalty、repetition penalty、bad words、allowed token ids、structured output 等 ModelRunnerV1 已有能力。
5. 不支持 logprobs。检测到请求 logprobs 时直接 fallback 到旧路径。
6. 提前下发随机数生成任务，用独立流与模型执行 overlap。
7. 工程实现保持小而集中，不引入过度抽象和复杂 fallback。能明确判断不满足约束时走旧路径；出现理论上不应发生的状态时直接抛异常。

## 6. 新分支启用条件

不再引入配置开关。ModelRunnerV1 在每次采样前判断是否可以使用新分支：

必须满足：

- 当前设备是 NPU。
- 当前请求不需要 logprobs。
- 当前采样输入可以整理成 V2 风格 batch。
- 当前 logits processing 参数均已被新分支支持。
- speculative decoding 为空，或 speculative decoding 的 rejection sample method 为 `probabilistic`。
- 不处于尚未适配的并行/分片/特殊路径，例如当前实现无法正确构造全局 vocab 语义时。

不满足上述条件时直接走旧路径，不记录静默降级之外的复杂状态。开发阶段可以用 debug 日志或断言辅助定位，但正式路径不要堆过多容错代码。

建议封装一个局部判断函数：

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

实际实现时要按当前 `SamplingMetadata` 字段命名补齐 logprobs 判断，但保持函数短小。

## 7. 总体架构

新方案分为四层：

1. **入口选择层**：位于 `NPUModelRunner._sample` 或更靠近采样入口的位置，判断是否走新分支。
2. **输入适配层**：把 ModelRunnerV1 的 `logits`、`SamplingMetadata`、`SpecDecodeMetadata` 整理成 V2 sampler/rejection sampler 需要的 tensor 格式。
3. **随机数预取层**：在 `execute_model` 中，模型图下发前，提前申请随机数 tensor，并在独立 NPU stream 上填充 uniform/Gumbel。
4. **采样算子层**：新增 V1 专用 V2-style sampling/rejection sampling 算子，支持 `draft_probs=None`，不修改 V2 原始算子。

数据流：

```text
execute_model
  ├─ 默认流申请 random buffers
  ├─ random stream 填充 accept_uniform / sample_gumbel / resample_gumbel
  ├─ 默认流执行模型图得到 logits
  └─ _sample
       ├─ 判断是否走新分支
       ├─ 构造 V2-style input batch
       ├─ logits processing
       ├─ 普通采样或拒绝采样
       └─ 输出 SamplerOutput
```

## 8. 输入格式适配

### 8.1 普通采样

不开投机时，输入最简单：

- `target_logits`: `[num_reqs, vocab_size]`
- `idx_mapping`: `[num_reqs]`
- `temperature/top_k/top_p/...`: per request tensors
- `sample_gumbel`: `[num_reqs, vocab_size]` 或按算子需要组织

输出：

- `sampled_token_ids`: `[num_reqs, 1]`

普通采样也应走同一套 logits processing 和 Gumbel argmax 逻辑，以便不开投机也能获得 V2 采样优化收益。

### 8.2 投机解码拒绝采样

投机时需要把 V1 的 `SpecDecodeMetadata` 转成以下格式：

- `target_logits`: `[num_logits, vocab_size]`
- `draft_sampled`: `[num_logits]`
- `cu_num_logits`: `[num_reqs + 1]`
- `idx_mapping`: `[num_reqs]`
- `expanded_idx_mapping`: `[num_logits]`
- `expanded_local_pos`: `[num_logits]`
- `positions`: `[num_logits]`
- `accept_uniform`: `[num_logits]`
- `resample_gumbel`: `[num_reqs, vocab_size]`

其中：

- `cu_num_logits[i]` 和 `cu_num_logits[i + 1]` 标识第 `i` 个 request 的 logits 范围。
- `expanded_idx_mapping[row]` 标识第 `row` 行 logits 属于哪个 request。
- `expanded_local_pos[row]` 标识第 `row` 行 logits 是当前 request 的第几个 speculative step。
- `draft_sampled` 从 `input_ids[logits_indices]` 得到，用于接受测试。

当前 POC 已验证可以预分配并复用 `cu_num_logits`、`idx_mapping`、`expanded_idx_mapping`、`expanded_local_pos`，避免在关键路径上反复 `arange/repeat_interleave/empty`。

正式实现可以保留这种缓存，但建议集中在一个小型 helper 中，例如：

```python
class V1V2SamplingInputBuilder:
    def build_spec_decode_inputs(...)
    def build_sampling_inputs(...)
```

不要把大量临时字段散落在 `NPUModelRunner` 主体逻辑中。

## 9. draft probabilities 策略

正式方案不保存 draft logits，也不构造 draft probabilities。

原因：

- 完整 draft logits 形状约为 `[max_reqs, num_spec_tokens, vocab_size]`，内存占用很高。
- 保存 draft logits 需要在 proposer 每一步额外 copy，容易抵消采样收益。
- 当前 ModelRunnerV1 目标是先拿到 V2-style sampling 的性能收益，不要求一次性完成全概率语义扩展。

实现策略：

- 新写 V1 专用拒绝采样算子，例如 `v1_rejection_sample_without_draft_probs`。
- 算子接口显式接受 `draft_probs=None` 或不提供 `draft_probs` 参数。
- acceptance 阶段按当前 V1 可支持语义处理 draft token。
- resample 阶段使用 target distribution 与外置 Gumbel 完成恢复采样。
- 仅 ModelRunnerV1 的新分支调用该算子；ModelRunnerV2 原有 rejection sampler 不改，避免影响 V2 功能。

如果后续要恢复完整 draft probabilities 语义，应另起增量方案，并用性能数据证明收益没有被 draft logits 保存成本抵消。

## 10. 后处理参数支持

第一版正式实现必须覆盖现有常见采样参数：

- `temperature`
- `top_k`
- `top_p`
- `min_p`
- presence penalty
- frequency penalty
- repetition penalty
- bad words
- allowed token ids
- structured output / grammar mask
- greedy sampling

推荐做法：

1. 尽量复用已有 logits processor，使新分支只消费 processed logits。
2. 新分支不要理解每一种 logits processing 的内部细节。
3. 只有在 V2 sampler 已经提供等价入口时，才把参数下沉到 V2-style sampler。
4. 对无法确认等价的参数，先 fallback 到旧路径，不做静默近似。

关键原则是：新采样分支应该优化采样和拒绝采样，不应该重新实现一套 logits processing。

## 11. 异步随机数设计

这是正式方案的关键优化点。

### 11.1 为什么要提前下发

采样需要两类随机数：

- acceptance uniform：用于拒绝采样接受测试。
- Gumbel / exponential：用于普通随机采样和 resample。

如果这些随机数在 `_sample` 内同步生成，就会和模型执行串行。正确做法是在 `execute_model` 中确认当前 step 可以走新采样分支后，在模型图下发前提前启动随机数生成，让随机数生成与模型执行 overlap。

### 11.2 避免 enable_async_exponential 的坑

已有 `enable_async_exponential` 思路可以参考，但不能照搬其中“在异步流内部申请 tensor”的写法。异步流中申请 tensor 可能和默认流内存生命周期/缓存复用发生踩踏。

必须遵守：

1. 在默认流上申请或复用 random tensor。
2. 在 random stream 上等待默认流完成申请。
3. 在 random stream 上填充数值。
4. 在采样前让默认流等待 random stream 的事件。

伪代码：

```python
def _prepare_sampling_randoms(self, num_logits, num_reqs, vocab_size):
    accept_uniform = self._get_buffer("accept_uniform", (num_logits,), torch.float32)
    sample_gumbel = self._get_buffer("sample_gumbel", (num_logits, vocab_size), torch.float32)
    resample_gumbel = self._get_buffer("resample_gumbel", (num_reqs, vocab_size), torch.float32)

    current_stream = torch.npu.current_stream()
    random_stream = self.v2_sampling_random_stream
    random_stream.wait_stream(current_stream)

    with torch.npu.stream(random_stream):
        accept_uniform.uniform_()
        accept_uniform.clamp_(min=1e-20)
        sample_gumbel.exponential_()
        sample_gumbel.log_().neg_()
        resample_gumbel.exponential_()
        resample_gumbel.log_().neg_()
        self.v2_sampling_random_ready_event.record()

    return accept_uniform, sample_gumbel, resample_gumbel
```

采样前：

```python
torch.npu.current_stream().wait_event(self.v2_sampling_random_ready_event)
```

实现时可以根据普通采样/投机采样分别决定是否需要 `sample_gumbel` 和 `resample_gumbel`，不要无条件生成超大 tensor。

## 12. 算子设计

不要直接修改 ModelRunnerV2 当前使用的 rejection sampler 算子。新增 V1 专用算子，原因：

- V2 现有功能不能被 POC/过渡方案破坏。
- V1 当前需要 `draft_probs=None` 的路线，和 V2 完整语义不完全一致。
- 新算子可以更激进地裁剪参数和分支，保持高内聚。

建议新增文件：

```text
vllm_ascend/worker/v1/sample/v2_sampling_adapter.py
vllm_ascend/worker/v1/sample/rejection_sampler_without_draft_probs.py
```

如果不想新增目录，也可以放在现有 `vllm_ascend/worker/model_runner_v1.py` 邻近模块，但不要继续把长 kernel 和 input builder 堆进 `model_runner_v1.py`。

算子接口建议：

```python
def rejection_sample_without_draft_probs(
    target_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    positions: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    accept_uniform: torch.Tensor,
    resample_gumbel: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

普通采样算子接口建议：

```python
def sample_from_processed_logits(
    processed_logits: torch.Tensor,
    temperature: torch.Tensor,
    sample_gumbel: torch.Tensor,
) -> torch.Tensor:
    ...
```

## 13. ModelRunnerV1 接入步骤

### Step 1: 抽象输入 builder

实现一个小 helper，负责：

- 从 `SpecDecodeMetadata` 构造 `cu_num_logits`。
- 复用 `idx_mapping`。
- 填充 `expanded_idx_mapping` 和 `expanded_local_pos`。
- 返回一个轻量 `SimpleNamespace` 或 dataclass。

这部分应包含最少状态，避免污染 ModelRunnerV1。

### Step 2: 加入随机数 buffer 和 stream

在 `NPUModelRunner.__init__` 中创建：

- random stream
- ready event
- buffer holder

只初始化对象，不在初始化时申请大 tensor。大 tensor 按最大实际 shape 懒分配并复用。

### Step 3: 在 execute_model 早期预取随机数

在确认本 step 可以走新采样分支后，模型图执行前调用 `_prepare_sampling_randoms`。

注意：判断逻辑不能依赖模型输出 logits，但可以依赖 `sampling_metadata`、`spec_decode_metadata`、batch size、vocab size、是否 logprobs 等已知信息。

### Step 4: 改造 `_sample`

在 `_sample` 内：

1. 再次检查是否可以走新分支。
2. 等待 random ready event。
3. 复用已有 logits processing 得到 processed logits。
4. 普通采样调用 `sample_from_processed_logits`。
5. 投机采样调用 `rejection_sample_without_draft_probs`。
6. 返回 `SamplerOutput`。

### Step 5: fallback

以下情况直接旧路径：

- 请求 logprobs。
- 未支持的 logits processing 参数。
- 未支持的 speculative method。
- reduced sampling 相关路径尚未适配。
- TP/global vocab 语义无法确认。

fallback 应简洁，不要引入复杂状态机。

## 14. 与 reduced sampling 的关系

本方案和 reduced sampling 是正交关系：

- reduced sampling 优化的是候选空间，依赖 top-k/top-p 等候选裁剪。
- 本方案优化的是随机数生成、采样流程和拒绝采样算子。

第一阶段不要强行融合二者，避免把 logits processing 和 rejection sampling 重新耦合。后续可以在以下条件满足时叠加：

- logits processing 已经自然产出候选集合和 candidate scores。
- candidate indices 可以无额外 dense topk 成本地传给采样算子。
- resample 的数学语义明确，尤其是 residual distribution 的归一化与随机采样范围。

本轮 POC 已证明，额外对 full processed logits 做 top-k 再 candidate resample 不划算，应避免重走这条路。

## 15. 测试计划

### 15.1 单元测试

新增或扩展 UT：

- 普通采样：greedy、temperature、top-k、top-p、min-p。
- 投机采样：probabilistic rejection sampling，无 `draft_probs`。
- fallback：logprobs、未支持参数、未支持 speculative method。
- 输入 builder：不同 batch size、不同 speculative token 数、不同 `cu_num_sampled_tokens`。

### 15.2 正确性测试

与旧路径对齐：

- 固定随机数。
- 固定 logits 和 draft tokens。
- 对比 sampled token ids。
- 对比 accepted length / num sampled。

对随机采样不要比较随机输出分布的单次结果，应通过固定外置随机数或统计分布验证。

### 15.3 性能测试

保留当前 POC 性能 UT 的核心设置：

- vocab size: `151936`
- batch sizes: `1, 4, 16, 64, 128`
- warmup: `5`
- iterations: `20`
- old path golden 同样模拟异步随机数

性能对比方法必须固定，避免把随机数生成、首次编译或异步同步误算进新旧路径差异：

1. **同一进程内对比**：同一个 pytest case 内先构造一份固定 logits、draft token、sampling metadata，再分别执行 old path 和 new path，避免不同进程初始化、编译缓存和环境噪声影响结论。
2. **old path 作为 golden**：old path 使用现有 Ascend rejection sampler；为了公平，新旧路径都要具备异步随机数能力。不能拿“旧路径同步生成随机数”对比“新路径异步随机数”，否则收益会被放大。
3. **固定输入和随机数形态**：target logits、draft logits/draft tokens、temperature、top-k、top-p、acceptance uniform、resample Gumbel 的 shape 必须一致或语义等价。需要比较正确性时，使用固定随机数或预填充随机 tensor。
4. **先校验输出再统计性能**：每个 batch size 必须先 `assert torch.equal(new_out, old_out)` 或校验 accepted length/token ids，再打印性能结果。
5. **warmup 后再计时**：每条路径先 warmup 5 次，warmup 后 `torch.npu.synchronize()`，再开始计时。
6. **计时边界明确**：只统计采样/拒绝采样目标路径，不把测试数据构造、首次 tensor 分配、远端日志打印算进去。计时循环结束后再次 `torch.npu.synchronize()`。
7. **取平均并报告 speedup**：真实执行 20 次，输出 `old_ms`、`new_ms` 和 `old_ms / new_ms`。如果要提交 PR 性能数据，建议每组至少重复 3 轮，报告均值和最差轮结果。
8. **覆盖代表性 batch**：至少覆盖 `1, 4, 16, 64, 128`，避免只看小 batch；本轮 POC 已证明部分方案在小 batch 看起来有效，但大 batch 会回退。
9. **单独拆分普通采样和投机采样**：不开投机的普通采样 benchmark 不能复用投机采样数据；投机采样也要单独报告不同 speculative token 数。

建议输出格式：

```text
sampling_optimization batch_size=64 vocab_size=151936
warmup_iters=5 num_iters=20 old_async_random=True new_async_random=True
old_ms=19.672 new_ms=17.125 speedup=1.149x
```

新增普通采样性能 UT：

- 不开投机。
- 无 top-k，仅 temperature/top-p。
- top-k/top-p 同时开启。

验收标准：

- 投机采样在 old async random golden 下保持稳定收益。
- 普通采样不开投机时也有收益。
- 不允许 candidate resample 这类额外 dense top-k 导致大 batch 回退。

## 16. 风险和约束

1. **数学语义风险**：不传 draft probabilities 必须在算子命名和调用条件上明确，不能伪装成完整 probabilistic rejection sampling。
2. **随机数生命周期风险**：random tensor 必须在默认流申请，在 random stream 填充，避免异步流申请导致内存踩踏。
3. **V2 功能风险**：不要直接改 V2 原算子，新增 V1 专用算子。
4. **logits processing 兼容风险**：新分支应消费 processed logits，避免感知每个 processor 内部逻辑。
5. **大 tensor 内存风险**：`[num_reqs, vocab_size]` Gumbel buffer 很大，必须复用，并按实际 batch shape 切片。
6. **TP/global vocab 风险**：如果当前 logits 是 local vocab，新算子必须明确 token id 语义；无法确认时 fallback。

## 17. 非目标

- 不实现 candidate resample。
- 不实现 skip resample。
- 不在第一阶段支持 logprobs。
- 不重写全部 logits processing。
- 不把 V2 原有 rejection sampler 改成 V1 专用语义。
- 不引入新的用户开关。

## 18. 推荐落地顺序

1. 新增 V1 专用 sampling input builder。
2. 新增 V1 专用 `draft_probs=None` rejection sampling 算子。
3. 接入外置随机数参数，先同步生成，验证正确性。
4. 在 `execute_model` 前移随机数生成，并引入 random stream overlap。
5. 接入普通采样路径。
6. 扩展 logits processing 参数覆盖范围。
7. 增加 fallback 和性能 UT。
8. 再评估是否与 reduced sampling 叠加。

## 19. 代码风格要求

- 代码路径高内聚：input builder、random manager、sampling kernel 分开，ModelRunnerV1 只做调度。
- 不写长篇容错逻辑。不满足条件就 fallback；内部不变量被破坏就抛异常。
- 不加魔法开关。
- 不引入不必要抽象。
- 不在 hot path 做 `.item()` 或 CPU/NPU 同步。
- 不在 hot path 申请大 tensor。
- 不在异步 stream 中申请 tensor。

## 20. 参考资料

- vLLM speculative decoding documentation: https://docs.vllm.ai/en/latest/features/spec_decode.html
- vLLM Ascend speculative decoding documentation: `docs/source/user_guide/feature_guide/speculative_decoding.md`
- vLLM Ascend MTP rejection sampling documentation: `docs/source/user_guide/feature_guide/Multi_Token_Prediction.md`
- POC branch: `codex/v1-upstream-rejection-sampler-poc`
- POC performance UT: `tests/ut/sample/test_upstream_rejection_sampler_poc.py`
