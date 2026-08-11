# STAIR：Model Runner V2 纯均衡 EPLB 融合 Policy 设计

## 状态

本文档记录 Model Runner V2 STAIR EPLB placement policy 的已确认设计，作为实现、测试和性能实验的依据。当前分支已包含实验实现；是否保留该实现由后续 NPU 等价实验决定。

STAIR 是 **Statistical Temporal-Aware Incremental Rebalancing** 的缩写。本方案融合 Swift 和 FlashLB 的有效部分，但不直接串联两套现有实现。其中，Statistical Temporal-Aware 对应 FlashLB 的时间序列统计和均衡评分，Incremental Rebalancing 对应 Swift 基于当前 placement 的增量构造；副本分配、合法性约束和局部优化由 STAIR 重新实现。

正式命名已经确认：配置值为 `stair`，算法核心类为 `StairEplbPolicy`，V2 adapter 类为 `StairEplbPolicyAdapter`，核心文件为 `policy_stair.py`。本文后续使用 “STAIR” 指代该实现。

## 结论

新 policy 只优化负载均衡，不建立搬运成本模型。它不依赖离线 profiling，不进行在线成本校准，也不增加任何与硬件带宽、专家字节数或预期回本周期有关的用户配置。

整个决策过程只有一个主目标：最小化采样窗口内各 rank 峰值负载相对平均负载的比值。placement 变化量只在两个候选的均衡效果等价时作为次级排序条件，不能阻止一个均衡效果更好的合法候选被采用。

## 背景

当前 Swift 和 FlashLB 各自解决了问题的一部分：

- Swift 从当前 placement 出发重新分配冗余专家，并在构造过程中避免同一专家重复出现在同一 rank。它的增量构造和本地槽位保留更接近异步 EPLB 的真实 placement 约束，但只使用窗口汇总负载，没有跨时间样本的风险信息，当前副本负载递推和跨层平均负载计算也存在问题。
- FlashLB 使用负载时间序列计算均值、方差和协方差，并在真实时间样本上评价候选 placement。它能够减少稳定窗口中的重复更新，但当前副本分配没有限制单专家副本数不超过 rank 数，placement 生成器在不存在合法目标 rank 时仍继续写入结果，且内部状态可能在 adapter 拒绝非法层之前被更新。

在当前等价实验中，Swift 产生了 198 个 global layer transfer；FlashLB 产生了 32 个，并在完成前两轮调整后连续七轮保持零变化。该结果说明时间序列评分和更新门槛有价值，但 FlashLB 当前生成 placement 的方式不能直接作为生产实现。新 policy 因此采用新的单一流程，而不是在一个 policy 的输出上调用另一个 policy 做修补。

## 目标与非目标

### 目标

1. 原生接收 Model Runner V2 的逻辑专家负载时间序列和当前 physical-to-logical map。
2. 使用窗口时间序列评价当前 placement 和候选 placement 的真实均衡效果。
3. 生成满足专家覆盖、rank 容量和同 rank 唯一性约束的 placement。
4. 从当前 placement 增量构造候选，并在均衡效果等价时保留更多现有专家和槽位。
5. 保持算法无跨窗口可变状态，使当前 committed placement 始终是唯一权威输入。
6. 不修改 Model Runner V2 现有 load collection、异步 worker、传输、commit 和 routing table 刷新流程。
7. 不增加 NPU 同步，不在 forward 热路径增加 Python 逻辑。

### 非目标

本方案不包含：

- 搬运耗时预测或收益回本模型；
- D2H、Gloo、H2D 或 ready-to-consume 成本参数；
- 离线或启动时 profiling；
- 在线传输成本拟合；
- changed-layer 或 changed-expert 搬运预算；
- 跨节点搬运 penalty；
- 独立 policy 子进程；
- Gloo/HCCL 传输机制修改；
- Swift 或 FlashLB 现有实现的兼容性重构；
- Model Runner V1 policy 行为修改。

TTFT、TPOT、端到端耗时和搬运量仍作为实验指标记录，但不反馈到新 policy 的目标函数。

## 输入、输出与负载模型

新 policy 通过 Model Runner V2 的 `AbstractEplbPolicy` 接口运行，并声明需要时间序列输入。主要输入为：

```text
expert_load:       [T, L, E]
current_placement: [L, P]
```

其中：

- `T` 是采样窗口中的时间点数量；
- `L` 是 MoE 层数；
- `E` 是逻辑专家数；
- `P` 是每层物理专家槽位总数；
- `P = R * S`，`R` 是 EP rank 数，`S` 是每个 rank 的槽位数。

输出为连续 CPU tensor：

```text
new_placement: [L, P]
```

输出 dtype、device、shape 和 contiguity 与 Model Runner V2 当前 policy contract 保持一致。输入 `expert_load` 和 `current_placement` 不允许原地修改。

与现有两套 policy 一致，新 policy 假设逻辑专家的路由负载平均分摊给它的所有物理副本。设第 `l` 层逻辑专家 `e` 的副本数为 `C[l,e]`，rank `r` 上的专家集合由 placement `P` 决定，则时间点 `t` 的 rank 负载为：

```text
rank_load[t,l,r]
    = sum(expert_load[t,l,e] / C[l,e]
          for e placed on rank r)
```

该假设是当前 Model Runner V2 逻辑专家负载统计能够支持的最精确信息。新 policy 不把逻辑负载重新展开为 legacy physical-slot load，也不伪造不同副本之间的负载差异。

## 纯均衡目标

每层 placement 的不均衡分数定义为：

```text
instant_score[t,l]
    = max(rank_load[t,l,:]) / mean(rank_load[t,l,:])

balance_score[l]
    = mean(instant_score[:,l])
```

分数越小越好，理论最优值为 `1.0`。当某个时间点该层总负载为零时，该时间点分数按 `1.0` 处理，不能因除零产生无穷大或 NaN。

候选层只在以下条件成立时更新：

```text
current_score - candidate_score > BALANCE_EPSILON
```

`BALANCE_EPSILON` 是内部数值容差，初始使用 `1e-6`，不暴露为用户配置。两个候选分数之差不超过该容差时，依次选择：

1. changed slots 更少的候选；
2. 跨 rank 变化更少的候选；
3. 专家 ID 和 rank ID 字典序更小的候选。

后两项只保证等价结果稳定和可复现，不构成搬运成本目标。

## 算法流程

### 1. 输入校验与快速返回

policy 每个 rebalance window 执行一次，不在 request、forward 或单个 MoE 层的设备热路径执行。

入口首先验证：

- `expert_load` 为三维 CPU tensor；
- `current_placement` 为二维 CPU tensor；
- layer 维一致；
- `P` 能被 `R` 整除；
- placement 中专家 ID 位于 `[0, E)`；
- 每个逻辑专家在每层至少出现一次；
- 每个 rank 的当前 placement 不包含重复逻辑专家。

以下情况直接返回当前 placement：

- 窗口负载全部为零；
- 当前层已经达到数值意义上的完全均衡；
- 候选生成失败或候选没有超过 `BALANCE_EPSILON` 的均衡收益。

即使 `P == E`、没有冗余槽位，也不能直接跳过整层。此时目标副本数固定为每个专家一个，但跨 rank 交换仍可能改善均衡，因此仍应进入局部均衡优化。

输入契约错误必须抛出明确异常。某一层在合法输入上无法构造更好的合法候选时只回退该层，不影响其它层。编程错误和内部数组越界不能被静默吞掉。

### 2. 时间序列统计

每层从 `[T,E]` 逻辑专家负载计算：

```text
mean[e]
variance[e]
covariance[e1,e2]
risk[e] = mean[e] + Z_SCORE * sqrt(variance[e])
```

`Z_SCORE` 沿用 FlashLB 当前默认值 `0.674`，作为候选生成启发式的内部常量，不作为用户配置。最终更新决定仍由完整时间序列的 `balance_score` 作出，因此统计近似不能单独决定是否采用 placement。

统计只为当前 rebalance window 创建，不保存跨窗口 ring buffer、EWMA 或历史最优分数。这样不需要 policy singleton、pending placement 或两阶段状态提交，也不会出现预测状态领先于真实 committed placement 的问题。

### 3. 目标副本数量分配

每个逻辑专家从一个副本开始，剩余 `P - E` 个冗余槽位逐个分配。每一步选择当前单位风险最高且仍可增加副本的专家：

```text
unit_risk[e] = risk[e] / replica_count[e]
```

副本分配必须始终满足：

```text
1 <= replica_count[e] <= R
sum(replica_count) == P
```

当多个专家的 `unit_risk` 相同时，优先专家 ID 更小者，保证所有 rank 对相同输入生成相同结果。

如果约束下无法分配完所有物理槽位，说明输入拓扑本身不可行，policy 必须报错，不能像当前 FlashLB 一样让 placement 阶段处理一个不可能满足的副本目标。

该步骤替换 Swift 的副本负载递推公式和 FlashLB 当前无 rank 上限的 `min_max_replica()`。第一版不使用 FlashTree 的分组树搜索；目标副本数由单一、可解释且确定性的 min-max 风险分配产生。

### 4. 增量 placement 构造

目标副本数确定后，placement planner 从当前 placement 开始，而不是生成一张与当前 rank 排列无关的新表。

#### 4.1 替换多余副本

planner 不先批量删除全部多余副本，因为独立删除可能使剩余空槽位无法满足同 rank 唯一性。它把一次“删除多余副本并填入缺失副本”作为原子替换：每次只从当前副本数高于目标值的专家选择一个槽位，并立即填入当前副本数低于目标值的专家。

每个候选替换后，planner 通过确定性的容量流检查验证剩余缺失副本仍能映射到剩余多余槽位。流图同时约束每个缺失专家的需求量、每个物理槽位最多使用一次、每个多余专家允许删除的副本数，以及目标 rank 不得已有待填入专家。无法完成全部剩余替换的候选不得进入评分。

#### 4.2 选择替换位置

按 `unit_risk` 从高到低处理副本不足的专家。对每一个待增加副本，只考虑满足以下条件的多余副本槽位：

- 目标 rank 尚未包含该专家；
- 被替换专家的当前副本数高于其目标副本数；
- 放置后所有专家覆盖和 rank 容量约束仍可完成。

在合法候选 rank 中，选择加入该专家后设备风险最小的 rank。设备风险使用专家均值、方差和同 rank 专家间协方差计算。风险等价时，优先保留更多当前 placement 的方案，再按 rank ID 和槽位 ID 决定。

如果某个缺失副本没有合法目标 rank，该层候选判定为不可行并回退到当前 placement，不能使用 `-1` rank 继续写入。

### 5. 有界局部均衡优化

初始候选构造完成后，执行最多 `MAX_REFINEMENT_STEPS` 次跨 rank 专家交换。初始值沿用 Swift 的 `100` 次上限，但它只是 policy CPU 计算量的安全边界，不是通信或搬运限制。

每轮优先检查当前窗口中贡献最高的 rank，对合法专家对执行交换模拟。交换必须保持：

- 每个 rank 槽位数不变；
- 同一专家不在同一 rank 重复；
- 每个逻辑专家的目标副本数不变；
- placement 中不存在空槽位或非法专家 ID。

只有完整窗口 `balance_score` 严格改善超过 `BALANCE_EPSILON` 时才接受交换。若一轮不存在可接受交换，立即停止。候选选择遵循统一 tie-break 规则，不能依赖 Python `set` 的遍历顺序。

该阶段借鉴 Swift 的 bounded swap，但删除 `num_max_com`、跨节点成本和发送方向限制，因为这些约束可能阻止纯均衡目标选择更优 placement。

### 6. 槽位对齐和最终校验

局部优化后，在不改变每个 rank 专家集合的前提下对槽位重新排列。当前 rank 上继续存在的专家必须保持原槽位，新增专家只填入被删除专家留下的槽位。

返回前逐层验证：

1. shape、dtype、device 和 contiguity 满足 V2 contract；
2. 所有专家 ID 合法；
3. 每个逻辑专家至少一个副本；
4. 每个逻辑专家副本数与目标副本数一致；
5. 单专家副本数不超过 rank 数；
6. 同 rank 不存在重复专家；
7. 每个 rank 槽位数不变；
8. 保留在同一 rank 的专家没有发生无意义的槽位换位；
9. 被采用层的 `candidate_score` 优于 `current_score`。

第一版保留 adapter 侧的独立 validator 作为防御性检查。planner 内部校验和 adapter validator 应使用同一组不变量；adapter 出现 rejected layer 视为实现缺陷，而不是正常控制流。被拒绝层回退当前 placement，并记录层号和失败规则，但日志不得引入设备同步。

## 数据和内存行为

新 policy 只处理 CPU 侧统计数据。它不读取 NPU tensor，不调用 `.item()` 访问设备数据，也不增加 `torch.npu.synchronize()` 或 NPU event。

实现应直接消费 V2 已收集的逻辑负载时间序列：

- contiguous CPU tensor 转 NumPy 时优先使用共享内存 view；
- 非 contiguous 输入只允许在 policy 入口创建一次 contiguous 副本；
- 不创建 `[T,L,R,S]` physical-slot load；
- current placement 保持只读；
- 输出从 current placement 的 CPU clone 开始，只覆盖确认采用的层；
- 均值、方差、协方差和 rank load 是单次 rebalance 的临时数组；
- 不创建跨窗口 buffer 或全局可变 policy 实例。

协方差矩阵是算法中最大的临时结构，形状为 `[E,E]`。实现按层计算并复用 scratch storage，避免同时保留 `[L,E,E]`。这使额外内存从 `O(L*E^2)` 降为 `O(E^2)`。

## 组件边界和集成方式

新实现保持上游 Model Runner V2 EPLB 生命周期不变：

```text
V2 logical load window
    -> Ascend policy adapter
    -> pure-balance planner
    -> validated [L,P] placement
    -> existing upstream diff/transfer pipeline
    -> existing async commit and routing refresh
```

Ascend 侧新增一个 V2-native policy core 和薄 adapter。core 负责统计、副本规划、placement 构造、局部优化、评分和内部合法性检查；adapter 只负责 V2 contract 校验、tensor/NumPy 边界、最终防御性校验和日志。

现有 Swift、FlashLB 和 upstream default policy 保持不变，继续作为对照组。新 policy 不修改 upstream `EPLB_POLICIES` 全局注册表，而是沿用 `AscendEplbState` 当前的 Ascend-specific policy 选择路径。

预期修改边界如下，具体新名称必须在编码前确认：

| 位置 | 动作 | 责任 |
| --- | --- | --- |
| `vllm_ascend/eplb/core/policy/policy_stair.py` | 新增 | V2-native STAIR 纯均衡算法核心 |
| `vllm_ascend/distributed/eplb_policy.py` | 扩展 | 新 policy 的 V2 adapter 和最终 validator |
| `vllm_ascend/distributed/eplb_state.py` | 扩展 | 根据 Ascend 配置选择新 adapter |
| `vllm_ascend/ascend_config.py` | 扩展 | 接受确认后的单一 policy 配置值 |
| `vllm_ascend/platform.py` | 扩展 | 保持现有 V2 EPLB/elastic EP 配置校验 |
| `tests/ut/eplb/core/policy/` | 扩展 | 核心算法性质和边界验证 |
| `tests/ut/distributed/test_eplb_policy.py` | 扩展 | V2 contract、adapter 和 validator 验证 |
| `tests/ut/distributed/test_eplb_state.py` | 扩展 | policy 选择验证 |
| `tests/ut/test_ascend_config.py`、`tests/ut/test_platform.py` | 扩展 | 配置接受与错误组合验证 |

第一版不修改 `/Users/freyfwt/Projects/vllm` 上游仓库。V2 时间序列输入和当前 placement 已经能够通过现有接口提供。

## 配置与命名

用户只需要通过现有 `additional_config.eplb_config.placement_policy` 选择 STAIR：

```text
placement_policy: stair
```

除该 policy 名称外不新增任何用户可见配置。

正式名称如下：

| 类型 | 名称 | 状态 |
| --- | --- | --- |
| 算法全称 | Statistical Temporal-Aware Incremental Rebalancing | 已确认 |
| 算法简称 | STAIR | 已确认 |
| 配置值 | `stair` | 已确认 |
| 算法核心类 | `StairEplbPolicy` | 已确认 |
| V2 adapter 类 | `StairEplbPolicyAdapter` | 已确认 |
| 核心文件 | `policy_stair.py` | 已确认 |
| 纯均衡评分函数 | `compute_balance_score` | 建议保留，职责明确 |
| 副本分配函数 | `allocate_replicas` | 建议保留，职责明确 |
| placement 构造函数 | `build_placement` | 建议保留，职责明确 |
| 局部优化函数 | `refine_placement` | 建议保留，职责明确 |

`hybrid`、`flash_swift`、`swift` 和 `flashlb` 不作为 STAIR 的别名，避免用户误认为它与某个现有 policy 完全兼容。

## 正确性与测试策略

### CPU 单元测试

核心算法必须证明以下性质：

- 副本总数守恒，所有逻辑专家至少一个副本；
- 单专家副本数不超过 rank 数；
- 同一 rank 不出现重复专家；
- 每个 rank 槽位数不变；
- 输入 tensor 不被修改；
- 相同输入产生完全一致的输出；
- 不同层总负载独立计算，不复用第 0 层平均值；
- 单副本、最大副本、全零负载和均匀负载行为正确；
- 不可行目标被明确识别，不产生 `-1` rank 或非法 placement；
- 每个实际变化层的窗口均衡分数严格改善；
- 分数等价时选择变化更少的 placement；
- 超过局部优化步数上限时仍返回合法候选。

adapter 和配置测试必须证明：

- `uses_expert_load_time_series` 已启用；
- 输入输出满足 V2 shape、dtype、CPU residency 和 contiguity contract；
- 不再展开 legacy slot load；
- planner 非法输出会被 validator 拒绝并逐层回退；
- 未选择新 policy 时默认路径不变；
- 新配置值只在启用 EPLB 的 Model Runner V2 下生效；
- 不支持的 elastic EP 组合在初始化阶段失败。

### 记录负载回放

使用 Swift 和 FlashLB 实验中保存的相同 `[T,L,E]` 负载窗口和初始 placement 离线回放，对比：

- 当前 placement、Swift、FlashLB 和新 policy 的 `balance_score`；
- 变化层数和 changed slots；
- 是否出现非法或 validator rejected layer；
- policy CPU 计算时间；
- 连续相同窗口是否稳定返回相同 placement。

回放首先验证算法，不以某次端到端耗时替代均衡正确性。

### NPU 端到端实验

沿用此前 V1/V2 等价实验的模型、DP/TP/EP、图模式、异步调度、荣誉专家数、采样窗口、policy interval、请求并发和输出长度。至少比较：

1. EPLB 关闭；
2. V2 Swift；
3. V2 FlashLB；
4. 新 policy。

记录：

- 更新前后 `balance_score`；
- policy 计算耗时；
- policy cycle 数；
- 实际变化层数；
- changed experts 和 global layer transfer 数；
- validator rejected layer 数；
- 平均 TTFT；
- 平均 TPOT；
- 请求端到端耗时；
- benchmark wall time 和吞吐。

所有观测日志必须使用 CPU 时间或现有异步生命周期事件，不能为了测量增加设备同步。

## 验收标准

新 policy 进入后续优化前必须同时满足：

1. 所有 CPU 正确性和确定性测试通过；
2. 记录负载回放中没有非法 placement；
3. NPU 实验中 validator rejected layer 为零；
4. 每个变化层的新 `balance_score` 严格优于旧 placement；
5. 稳定或重复窗口不产生由非确定性排序导致的 placement 抖动；
6. EPLB 关闭、upstream default、Swift 和 FlashLB 路径行为不受影响；
7. policy 不增加 NPU 同步或 forward 热路径工作。

端到端性能必须记录并与三个对照组比较，但本方案不承诺搬运量小于等于 FlashLB 的 32 次。纯均衡目标允许一个明显更均衡的 placement 产生更多变化；changed slots 只在均衡效果等价时参与选择。如果实验表明纯均衡收益无法覆盖真实搬运开销，应重新讨论目标函数，而不能在本实现中偷偷加入未确认的成本参数或搬运预算。

## 实施顺序

1. 编写独立的 STAIR CPU 纯算法核心和性质测试，不接入运行时。
2. 使用记录负载回放比较 Swift、FlashLB 和 STAIR，先解决所有非法 placement 和非确定性问题。
3. 增加 V2 adapter、配置选择和防御性 validator，不修改异步传输链路。
4. 运行 CPU 测试和格式检查。
5. 在指定 NPU 服务器运行等价端到端实验，记录均衡、搬运和性能结果。
6. 根据结果决定是否保留实验 policy、继续优化候选搜索，或回退实现。

命名确认和第一版运行时代码已经完成；最终状态以本文档记录的验收实验为准。
