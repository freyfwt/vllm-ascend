# STAIR：Model Runner V2 纯均衡 EPLB 融合 Policy 设计

## 状态

本文档记录 STAIR 在真实负载回放校正后的实现设计。STAIR 是 **Statistical Temporal-Aware Incremental Rebalancing** 的缩写，配置值为 `stair`，核心类为 `StairEplbPolicy`，V2 adapter 为 `StairEplbPolicyAdapter`。

STAIR 不建立搬运成本模型，不依赖离线 profiling，也不增加用户参数。它复用 Swift 生成完整、稳定、满足现有搬运约束的候选 placement，再使用 FlashLB 的时间序列评分和跨窗口 hysteresis 决定是否接受候选层。

## 回放校正

第一版 STAIR 自行使用当前 50-step 窗口的均值、方差和协方差重新分配冗余副本，并把 Swift 的 `num_max_com=1` 误实现为整层硬搬运上限。真实 V2 负载回放证明这两个设计会放大搬运量：

- 七个连续窗口中，第一版 STAIR 选择了 210 个 layer transfer，48 层全部发生过变化，多数层重复变化 5～7 次；
- 336 个“层×窗口”替换候选中，213 个因硬通信上限只完成了部分目标；
- 即使将每层目标在同一窗口补齐，仍会产生 188 个 layer transfer；
- 相邻 50-step 窗口有 43～47 层的冗余专家目标变化，说明主要问题是统计目标追逐短窗口噪声；
- 对相同负载执行“Swift 完整候选 + FlashLB 时间序列收益校验”，七个窗口只接受 19 个完整 layer transfer。

因此，最终设计不再维护第二套副本分配和 placement planner。STAIR 的融合边界改为：Swift 负责候选生成，FlashLB 负责时序接受。该组合直接复用已验证的 Swift 搬运语义，同时阻止聚合窗口上的偶然改善转化为没有时序收益的真实搬运。

## 目标与非目标

### 目标

1. 原生接收 Model Runner V2 的 `[T,L,E]` 逻辑专家负载时间序列和 `[L,P]` physical-to-logical map。
2. 保留 Swift 的层级失衡门槛、局部交换收益门槛、通信约束和完整 placement 生成行为。
3. 使用完整时间序列评价 Swift 候选，只提交真实改善均衡分数的层。
4. 保留 FlashLB 的跨窗口 hysteresis，并只记录最终实际提交的 placement。
5. 保持上游 Model Runner V2 的异步 load collection、policy worker、传输、commit 和 routing table 刷新流程不变。
6. 不增加 NPU 同步，不在 forward 热路径增加 Python 逻辑。

### 非目标

本方案不包含：

- 搬运耗时预测、收益回本模型或硬件成本参数；
- 离线或启动时 profiling；
- 用户可配置的 changed-layer 或 changed-expert 搬运预算；
- 独立 policy 子进程；
- Gloo/HCCL、D2H、H2D 或 ready-to-consume 生命周期修改；
- Swift、FlashLB 或 Model Runner V1 的行为修改；
- Swift 的整轮 5% 全局收益门槛。

TTFT、TPOT、端到端耗时和搬运量只作为实验指标，不反馈到 policy 的目标函数。

## 输入、输出与均衡分数

STAIR 通过 Model Runner V2 的 `AbstractEplbPolicy` 接口运行，并声明 `uses_expert_load_time_series = True`：

```text
expert_load:       [T, L, E]
current_placement: [L, P]
new_placement:     [L, P]
```

其中 `P = R * S`，`R` 为 EP rank 数，`S` 为每个 rank 的槽位数。输入和输出均位于 CPU；输出保持 `torch.long`、连续内存和原 shape，输入不得原地修改。

逻辑专家负载平均分摊给其物理副本。第 `t` 个样本中某层某 rank 的负载为：

```text
rank_load[t,r]
    = sum(expert_load[t,e] / replica_count[e]
          for e placed on rank r)
```

每层均衡分数为：

```text
instant_score[t] = max(rank_load[t,:]) / mean(rank_load[t,:])
balance_score    = mean(instant_score[:])
```

分数越小越好，理论最优为 `1.0`。总负载为零的样本按 `1.0` 处理。

## 决策流程

### 1. 输入校验与快速返回

adapter 校验输入维数、CPU residency、layer 数、replica 数、rank/node 整除关系和当前 placement shape。窗口负载全零时直接返回当前 placement，不进入 Swift，也不更新 hysteresis。

最终候选逐层验证：

- 专家 ID 位于 `[0,E)`；
- 每个逻辑专家至少有一个副本；
- 同一逻辑专家的副本数不超过 rank 数；
- 同一 rank 不含重复专家；
- 继续保留在同一 rank 的专家不发生无意义的槽位换位。

非法候选层回退到当前 placement，并记录 rejected layer 数。

### 2. Swift 生成完整候选

adapter 将时间序列沿窗口维求和，得到 `[L,E]` 聚合逻辑负载。该操作只发生在 CPU policy worker 中，不触发设备同步。

为兼容现有 Swift 输入，逻辑专家负载按当前副本数均分到每个物理槽位，再调用现有 `SwiftBalanceEplb`。STAIR 不复制或重写 Swift planner，因此继承其完整语义：

- 层级 aggregate peak/average 失衡门槛为 `1.01`；
- 单步 swap 至少降低 aggregate average load 的 `1%`；
- `num_max_com=1` 约束优选冗余分配和后续 swap；
- 当优选阶段受通信额度限制时，Swift 的 fallback 会在当前窗口补齐剩余冗余槽位；
- 继续存在于本 rank 的专家保留原槽位；
- 只有 aggregate placement 改善的层才写入候选。

`num_max_com=1` 不是“整层最多搬一个专家”的硬预算。把它解释成硬预算会把一个完整层更新拆到多个 EPLB 窗口，而运行时传输开销按层支付，最终造成同一层反复搬运。

Swift 返回的整轮 `change` 标志包含 5% 全局收益判断。STAIR 与现有 V2 Swift adapter 一致，忽略该全局标志，只使用逐层候选。这样不会因其它层收益不足而否决少数确有价值的层。

### 3. FlashLB 时间序列接受

对 Swift 实际改变的每一层，STAIR 使用完整 `[T,E]` 时间序列分别计算当前 placement 和 Swift 候选的 `balance_score`。只有满足以下条件才接受：

```text
current_score - candidate_score > 1e-6
```

这一步是减少短窗口抖动的关键。Swift 的聚合负载可能认为候选改善，但完整时间序列会揭示不同时间点上的峰值被抵消、反转或仅是采样偶然。未获得时序收益的候选层保持当前 placement。

STAIR 同时保留 FlashLB hysteresis。首次观察某层时允许评估；提交后记录该候选的 average-to-peak ratio。后续仅在以下任一条件满足时重新接受搜索结果：

```text
current_ratio < past_ratio * relative_threshold
current_ratio < absolute_threshold
```

当 `R < 32` 时，阈值为 `0.95` 和 `0.9`；否则为 `0.9` 和 `0.85`。该门槛限制稳定层重复更新，但不替代候选的正时序收益校验。

### 4. Commit 一致性

adapter 完成最终 validator 后才调用 `commit()`。STAIR 只为实际变化层记录新的 ratio，并保存预期 placement。若下一窗口观察到的当前 placement 与预期结果不一致，说明异步 commit、validator 或外部状态没有按预测推进，policy 会清空历史，避免内部状态领先于真实 routing table。

## 数据和线程行为

STAIR 只处理上游已经交给 policy worker 的 CPU tensor：

- `weight.sum(dim=0)`、slot-load 展开、Swift 计算和时间序列评分均在 CPU 执行；
- 不读取 NPU tensor，不调用 `torch.npu.synchronize()`，也不使用设备 tensor `.item()`；
- 不向日志添加 NPU event 或同步计时；
- 只跨窗口保存每层一个 ratio 和一份预期 placement；
- 不保存负载窗口，不维护搬运成本状态。

因此该实现不会把异步调度重新同步化。policy CPU 耗时仍需在真实实验中记录，但可与前台 NPU forward 并行。

## 组件边界

```text
V2 logical load time series
    -> Ascend STAIR adapter
    -> aggregate load + existing Swift candidate generator
    -> STAIR temporal score and FlashLB hysteresis
    -> final placement validator
    -> existing upstream async diff/transfer/commit pipeline
```

实现只修改 vLLM Ascend：

| 位置 | 责任 |
| --- | --- |
| `vllm_ascend/eplb/core/policy/policy_stair.py` | 时间序列评分、hysteresis 和 commit 状态 |
| `vllm_ascend/distributed/eplb_policy.py` | V2 contract、Swift 候选生成、最终 validator 和日志 |
| `vllm_ascend/distributed/eplb_state.py` | 根据 Ascend 配置选择 STAIR adapter |
| `vllm_ascend/ascend_config.py` | 接受 `placement_policy: stair` |

不修改 `/Users/freyfwt/Projects/vllm` 上游接口，也不改变 `SwiftBalanceEplb` 本身。

## 配置

用户仅通过现有 Ascend 配置选择 STAIR：

```text
additional_config.eplb_config.placement_policy: stair
```

不增加其它用户可见配置。

## 测试与性能验收

CPU 单元测试至少证明：

- Swift adapter 仍与直接调用 legacy Swift 的结果一致；
- STAIR 使用完整窗口聚合负载生成 Swift 候选；
- 只接受时间序列分数严格改善的候选层；
- FlashLB 相对和绝对 hysteresis 阈值生效；
- history 只对最终提交层更新，observed placement 不一致时清空；
- 输入不被修改，输出满足 shape、dtype、CPU 和 contiguity contract；
- 非法 placement 被明确拒绝或逐层回退。

真实 NPU 实验沿用此前 V1/V2 等价 case：TP2、EP、异步调度、`FULL_AND_PIECEWISE`、4 个冗余专家、`window_size=50`、等价 interval、并发 8、输出 256。记录：

- TTFT、TPOT、端到端耗时和吞吐；
- policy cycle 数和 CPU policy 耗时；
- 每轮 Swift proposed layer、STAIR accepted layer 和 rejected layer；
- 实际 global layer transfer 总数；
- 与 V1 Swift 开启 EPLB 的同 case 差异。

验收重点不是某一轮恰好零变化，而是完整测量窗口内将搬运量压回 V1 同一数量级，同时保持 validator rejection 为零，并使 TPOT、端到端耗时和吞吐显著接近 V1。
