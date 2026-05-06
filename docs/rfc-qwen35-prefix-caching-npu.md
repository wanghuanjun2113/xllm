# RFC: xLLM Qwen3.5 Prefix Caching on NPU

| 字段 | 值 |
|---|---|
| **RFC ID** | 001 |
| **标题** | xLLM NPU 上支持 Qwen3.5 混合注意力架构的 Prefix Caching |
| **状态** | Draft |
| **日期** | 2026-05-06 |
| **目标评审人** | xLLM 推理引擎团队 / NPU 算子团队 |

---

## 1. Summary

本提案为 xLLM 在 NPU 上实现 Qwen3.5 系列模型的 Prefix Caching。Qwen3.5 采用 Full Attention + Linear Attention（Gated DeltaNet）混合架构，Prefix Caching 需要同时复用两种路径的状态：Full Attention 路径复用 KV Cache，Linear Attention 路径复用 conv state 与 ssm state。

核心挑战在于 Linear State 体积远大于 Full KV（单 checkpoint 约 60–183 MiB），且 `chunk_gated_delta_rule` 仅输出 final state，无法在任意位置缓存中间状态。因此 Linear State 只能在 chunk 边界缓存，checkpoint interval 必须与调度切片边界对齐。

## 2. Motivation

### 2.1 背景

Qwen3.5 系列引入了混合注意力机制：部分层使用标准 Full Attention，其余层使用 Linear Attention（Gated DeltaNet）。与纯 Full Attention 模型不同，Linear Attention 层不维护逐 token 的 KV Cache，而是维护 conv state 和 ssm state 两种递推状态。

在多轮对话、系统 prompt 复用等场景下，Prefix Caching 能显著降低首 token 延迟（TTFT）和重复计算开销。现有的 prefix caching 方案仅处理 Full KV，无法覆盖 Linear Attention 的状态复用。

### 2.2 问题

1. **Linear State 无法逐 token 缓存**：conv state 和 ssm state 是递推状态，不像 KV Cache 可以按 token 边界切片。`chunk_gated_delta_rule` 只输出 final state，不支持输出中间 state。
2. **Linear State 显存开销大**：单次 checkpoint（1024 token 间隔）的 Linear State 约 60–183 MiB，是 Full KV 同区间开销的 2–3 倍，必须采用稀疏 checkpoint 策略。
3. **两种状态需联动管理**：Full KV 和 Linear State 必须同时命中才能复用 prefix，淘汰策略需协调一致，避免出现 Full KV 命中但 Linear State 丢失（或反之）导致的无效命中。

### 2.3 目标

- 支持 Qwen3.5-27B、Qwen3.5-35B-A3B、Qwen3.5-397B-A17B 三个模型规格的 Prefix Caching。
- Linear State 稀疏 checkpoint，默认 1024 token 间隔，显存开销可控。
- Full KV 与 Linear State 联动命中与淘汰，保证 prefix 复用的正确性。
- NPU 显存三池分离布局，所有状态数据常驻显存，避免动态分配带来的性能抖动。

## 3. Detailed Design

### 3.1 稀疏 Checkpoint

Linear State 按 `checkpoint_interval`（默认 1024 tokens）间隔缓存。复用现有参数 `long_prefill_token_threshold` 作为 interval 参数，不引入独立配置：

```
checkpoint_interval = long_prefill_token_threshold > 0
                      ? long_prefill_token_threshold
                      : 1024
checkpoint position = k × checkpoint_interval   (k = 1, 2, 3, ...)
```

每个 checkpoint 包含所有 Linear 层的 conv state + ssm state + cumulative prefix hash。

调度层面，Chunked Prefill 单步 token 数向下对齐到 `checkpoint_interval` 边界，保证 chunk 结束位置恰好可缓存。

**权衡**：间隔越小，命中率越高但显存占用越大；间隔越大则相反。1024 是默认平衡点，可通过 `long_prefill_token_threshold` 调整。

### 3.2 三池分离的显存布局

NPU HBM 采用三池分离布局，所有数据常驻显存：

```
HBM: | Full KV Pool | Conv State Pool | SSM State Pool |
```

每个 State Pool（Conv/SSM）内部分为 live 区域和 checkpoint 区域：

```
Conv State Pool:  | live_0 | live_1 | ... | ckpt_0 | ckpt_1 | ... |
SSM State Pool:   | live_0 | live_1 | ... | ckpt_0 | ckpt_1 | ... |
```

| 区域 | 用途 | 生命周期 |
|---|---|---|
| **Full KV Pool** | 按 token block 粒度管理，复用现有 block manager | block ref-count = 0 时可淘汰 |
| **Live Slot** | 每个运行中的请求占一个，保存当前 prefill/decode 的 Linear Attention 状态 | 请求结束即释放 |
| **Checkpoint Slot** | 缓存 checkpoint 边界的 Linear State 快照，供后续请求 prefix hit 复用 | LRU 淘汰 |

**数据流**：
- Cache hit 时：checkpoint slot → live slot（NPU 显存拷贝）
- Prefill 到 checkpoint 边界时：live slot → checkpoint slot（NPU 显存拷贝）

### 3.3 显存分配策略

按 Full KV 与 Linear State 的缓存成本比例切分可用 HBM：

```
buffer_ratio = full_kv_bytes_per_interval : conv_bytes_per_checkpoint : ssm_bytes_per_checkpoint
```

以 Qwen3.5-27B（默认 1024 interval）为例，比例约为 64 : 2.8 : 144。

**分配流程**：

1. Profile 模型加载后的可用显存，扣除 graph / workspace / safety margin。
2. 按 `buffer_ratio` 比例切分为 Full KV、Conv、SSM 三份。
3. Conv/SSM Pool 内先满足 live slot 下限：`live_slots = 1 + max_running_requests`，剩余空间分配给 checkpoint slot。
4. 校验 `effective_prefix_capacity = min(full_kv_token_capacity, linear_state_token_capacity)`，若瓶颈明显则调整 interval 或分配权重。

**容量校验示例**（Qwen3.5-27B，1024 interval）：

| 组件 | 每 interval 大小 | 占比 |
|---|---|---|
| Full KV | 64 KiB × 1024 = 64 MiB | 30.3% |
| Conv State | 2.8 MiB | 1.3% |
| SSM State | 144 MiB | 68.3% |
| **合计** | **210.8 MiB** | — |

SSM state 占比超过 95%（相对于 Linear State 总量），是显存规划的主要约束。

### 3.4 共享 Hash 链与命中策略

Full KV 与 Linear State 共用同一条 cumulative prefix hash 链。Match 结果取 `min(H_full, H_linear)` 作为有效命中长度。

**Prefix Match 流程**：

```
1. 按 token block 构建 cumulative hash
2. Full KV Cache 查询最长 block hit → H_full
3. Linear State Cache 查询 ≤ H_full 的最大
   checkpoint_interval 对齐 checkpoint → H_linear
4. H_effective = min(H_full, H_linear)
5. 模型从 H_effective 开始 prefill
```

**Fallback 场景**：

| 场景 | 策略 |
|---|---|
| Full KV miss | 从 position 0 prefill，Linear State 使用 zero state |
| Full KV hit + Linear State miss | 从 position 0 prefill（已命中 KV blocks 丢弃，Linear State 无法从 KV 重建） |
| 全命中 + 需 logits | 从最近 checkpoint 恢复，重算 suffix 到最后 token |

### 3.5 联动驱逐策略

Full KV 和 Linear State 必须同时命中才能复用 prefix，因此驱逐需联动决策：

**驱逐原则**：

1. **Full KV Block 驱逐**：沿用 ref-count + LRU。驱逐后，所有晚于该 position 的 Linear checkpoint 变为孤立 entry（Full KV 不连续导致无法从该位置开始 prefill），应级联淘汰以释放 checkpoint slot。
2. **Linear State Checkpoint 驱逐**：以 checkpoint slot 为单位 LRU 淘汰。active request 正在使用的 checkpoint 不可淘汰（ref-count > 0）。Linear checkpoint 被淘汰后，对应的 Full KV blocks 仍然保留——其他请求可通过更早的 Linear checkpoint 或 zero state 复用 Full KV。
3. **驱逐优先级**（从高到低）：
   - 孤立 Linear checkpoint（对应 Full KV 已不连续）
   - 普通 Linear checkpoint（ref-count = 0，最久未命中）
   - Full KV block（ref-count = 0）
   - Request preemption

**核心不变式**：淘汰 Full KV 时级联清理失效的 Linear checkpoint；淘汰 Linear checkpoint 时保留 Full KV。`min(H_full, H_linear)` 作为运行时兜底，确保 match 结果始终有效。

## 4. 算子改动

| 算子 | 改动描述 | 复杂度 |
|---|---|---|
| `chunk_gated_delta_rule` | 支持非零 initial state（修复 `fill_(0.0)`），保持只输出 final state，写入 live state slot | 低 |
| `linear_state_copy`（新增） | checkpoint slot ↔ live slot 之间的 NPU 显存拷贝 | 低 |

**算子要求**：

- conv state 与 ssm state 均支持批量拷贝。
- NPU tensor 地址稳定适配 ACL Graph replay。
- 热路径避免 CPU/NPU 同步（如 `tensor.item()`）。
- forward 中不动态分配大 tensor。

## 5. 配置参数

本方案复用现有参数 `long_prefill_token_threshold` 作为 checkpoint interval，不引入新配置项：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `long_prefill_token_threshold` | 1024 | 同时作为 Linear State 的 checkpoint interval |

> **讨论点**：是否需要引入独立的 `linear_checkpoint_interval` 参数，使其与 `long_prefill_token_threshold` 解耦？优势是可以独立调优 Linear State 的缓存粒度；劣势是增加配置复杂度。

## 6. Alternatives Considered

### 6.1 逐 token 缓存 Linear State

放弃稀疏 checkpoint，改为与 Full KV 相同的逐 token 粒度。

**否决原因**：需要修改 `chunk_gated_delta_rule` 输出每个 token 的中间 state，改动量大且当前算子不支持。即使支持，逐 token 的 Linear State 显存开销约为 Full KV 的 2–3 倍（61–183 KiB/token vs 20–64 KiB/token），不经济。

### 6.2 Linear State 与 Full KV 统一管理

将 Linear State checkpoint 纳入现有的 block manager 统一管理，而非独立 State Pool。

**否决原因**：Linear State 是固定大小的 checkpoint（不按 token 增长），与 block manager 的按 token block 动态分配模型不匹配。独立池化管理更简洁，且能避免 Linear State 与 Full KV 之间的显存碎片问题。

### 6.3 独立 Hash 链

Full KV 和 Linear State 使用独立的 prefix hash 链，分别匹配后取交集。

**否决原因**：增加 hash 计算与存储开销。共享 hash 链已能正确表达 prefix 的语义等价性，且 Linear State 只需在 checkpoint 边界记录 hash 值，无需额外存储。

## 7. Open Questions

1. **Checkpoint interval 可调性**：是否需要独立于 `long_prefill_token_threshold` 的配置参数？不同模型的最优 interval 可能不同。
2. **SSM State dtype 优化**：当前 ssm state 使用 float32，占 Linear State 的 >95%。是否可以安全地降精度到 bfloat16？需要评估对模型精度的影响。
3. **多 NPU 卡间 Checkpoint 迁移**：当前方案假设所有状态在同一张 NPU 上。多卡张量并行场景下，checkpoint 需要拆分/聚合，如何处理？
4. **Checkpoint 压缩**：是否需要对不活跃的 Linear State checkpoint 进行压缩（如量化到 int8）以降低显存占用？
5. **Full KV hit + Linear State miss 场景优化**：当前策略是丢弃已命中的 Full KV 从头 prefill。是否可以通过保留 Full KV、仅对 Linear Attention 层从头计算来优化？

## 8. References

- xLLM Qwen3.5 Prefix Caching on NPU 设计文档 v2（本 RFC 的设计来源）
- `chunk_gated_delta_rule` 算子实现
- Qwen3.5 模型规格（27B / 35B-A3B / 397B-A17B）
