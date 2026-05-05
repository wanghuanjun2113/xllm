# xLLM Qwen3.5 Prefix Caching on NPU 设计

## 1. 总体方案

为 xLLM 在 NPU 上支持 Qwen3.5 混合注意力架构的 Prefix Caching。Full Attention 路径复用 KV Cache，Linear Attention（Gated DeltaNet）路径复用 conv state 与 ssm state。由于 `chunk_gated_delta_rule` 只输出 final state、不输出中间 state，Linear State 只能在 chunk 边界缓存，因此 checkpoint interval 与调度切片边界对齐。

设计要点：

1. **稀疏 Checkpoint**：Linear State 按 `checkpoint_interval`（默认 1024）间隔缓存，复用 `long_prefill_token_threshold` 作为 interval 参数，不引入独立配置。
2. **三池分离**：NPU HBM 分为 Full KV Pool、Conv State Pool、SSM State Pool。每个 State Pool 内分 live slot（运行态）和 checkpoint slot（缓存态），所有数据常驻显存。
3. **调度对齐**：Chunked Prefill 单步 token 数必须对齐到 `checkpoint_interval` 边界，保证 final state 可直接作为 checkpoint。
4. **共享 Hash 链**：Full KV 与 Linear State 共用同一条 cumulative prefix hash 链，match 结果取 `min(H_full, H_linear)` 作为有效命中长度。
5. **联动驱逐**：Full KV blocks 和 Linear State checkpoints 联动 LRU 驱逐，淘汰 Full KV 时级联清理失效的 Linear checkpoint，淘汰 Linear checkpoint 时保留 Full KV。

## 2. 线性注意力状态缓存大小对比

Full Attention KV Cache 按 token 粒度缓存，每 token 每层大小为 `2 × KV_Heads × Head_Dim × sizeof(bf16)`。Linear Attention State 按 checkpoint 粒度缓存（默认 1024 token 间隔），包含所有 Linear 层的 conv state 和 ssm state。

**模型关键参数：**

| 模型 | 层数 | Full 层数 | Linear 层数 | Hidden | KV Heads | Linear V Heads | MoE |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5-27B | 64 | 16 | 48 | 5120 | 4 | 48 | N/A |
| Qwen3.5-35B-A3B | 40 | 10 | 30 | 2048 | 2 | 32 | 256 experts, top-8 |
| Qwen3.5-397B-A17B | 60 | 15 | 45 | 4096 | 2 | 64 | 512 experts, top-10 |

通用参数：Head Dim = 256，Linear Dim = 128，conv_kernel_dim = 4，dtype = bf16，ssm_dtype = float32。

**KV Cache 与 Linear State 大小对比：**

| 模型 | Full KV / token (全 Full 层) | Conv / checkpoint (全 Linear 层) | SSM / checkpoint (全 Linear 层) | 单 checkpoint 合计 | 摊销 / token |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-27B | 64 KiB | 2.8 MiB | 144 MiB | 146.8 MiB | 146.8 KiB |
| Qwen3.5-35B-A3B | 20 KiB | 1.4 MiB | 60 MiB | 61.4 MiB | 61.4 KiB |
| Qwen3.5-397B-A17B | 30 KiB | 3.2 MiB | 180 MiB | 183.2 MiB | 183.2 KiB |

结论：Linear State 单 checkpoint 体积远大于 Full KV 单 token，必须采用稀疏 checkpoint。SSM state 占比最高（>95%），是显存规划的主要约束。

## 3. 主要设计点

### 3.1 间隔缓存

Linear State 只在 `checkpoint_interval` 对齐边界缓存：

```text
checkpoint_interval = long_prefill_token_threshold > 0 ? long_prefill_token_threshold : 1024
checkpoint position = k × checkpoint_interval
```

- 间隔越小，命中率越高但显存占用越大；间隔越大则相反。1024 是默认平衡点。
- checkpoint 包含所有 Linear 层的 conv state + ssm state + cumulative prefix hash。
- 调度 token 数向下对齐到 `checkpoint_interval`，保证 chunk 结束位置恰好可缓存。

### 3.2 独立 Buffer

NPU HBM 采用三池分离布局，所有数据常驻显存：

```text
HBM: | Full KV Pool | Conv State Pool | SSM State Pool |
```

每个 State Pool 内部分为 live 区域和 checkpoint 区域：

```text
Conv State Pool:  | live_slot_0 | live_slot_1 | ... | ckpt_slot_0 | ckpt_slot_1 | ... |
SSM State Pool:   | live_slot_0 | live_slot_1 | ... | ckpt_slot_0 | ckpt_slot_1 | ... |
```

- **Full KV Pool**：复用现有 block manager，按 token block 粒度管理。
- **Live Slot**：每个运行中的请求占用一个 live slot，保存当前 prefill/decode 的 Linear Attention 状态。请求结束后释放。
- **Checkpoint Slot**：缓存 `checkpoint_interval` 边界的 Linear State 快照，供后续请求 prefix hit 时复用。cache hit 时将 checkpoint slot 数据拷贝到请求的 live slot；prefill 到 checkpoint 边界时将 live slot 数据拷贝到一个 checkpoint slot。

### 3.3 Buffer 大小分配策略

按 Full Attention 和 Linear Attention 的缓存成本比例切分可用 HBM：

```text
buffer_ratio = full_kv_bytes_per_interval : conv_bytes_per_checkpoint : ssm_bytes_per_checkpoint
```

默认 1024 interval 时近似比例（Qwen3.5-27B）：64 : 2.8 : 144。

分配流程：

1. profile 模型加载后的可用显存，扣除 graph/workspace/safety margin。
2. 按 `buffer_ratio` 比例切分为 Full KV、Conv、SSM 三份。
3. Conv/SSM Pool 内先满足 live slot 下限：`live_slots = 1 + max_running_requests`，剩余空间分配给 checkpoint slot。
4. 校验 `effective_prefix_capacity = min(full_kv_token_capacity, linear_state_token_capacity)`，若瓶颈明显则调整 interval 或分配权重。

### 3.4 淘汰策略

Full KV 和 Linear State 必须同时命中才能复用 prefix，因此淘汰策略需联动决策，而非完全独立。

**联动驱逐原则**：以 Full KV 的有效范围为锚点。Full KV block 是逐 token 粒度的，Linear checkpoint 是 `checkpoint_interval` 粒度的。一个 Linear checkpoint 能被命中，前提是对应位置及之前的 Full KV blocks 都存在。

**驱逐流程**：

1. **Full KV Block 驱逐**：沿用 ref-count + LRU。当某个 Full KV block 被驱逐后，所有晚于该 position 的 Linear checkpoint 变为孤立 entry（Full KV 不连续导致无法从该位置开始 prefill），应级联淘汰以释放 checkpoint slot。
2. **Linear State Checkpoint 驱逐**：以 checkpoint slot 为单位 LRU 淘汰。active request 正在使用的 checkpoint 不可淘汰（ref-count > 0）。Linear checkpoint 被淘汰后，对应的 Full KV blocks 仍然有效——其他请求可通过更早的 Linear checkpoint 或 zero state 从该位置复用 Full KV，因此不级联淘汰 Full KV。
3. **驱逐优先级**：孤立 Linear checkpoint（对应 Full KV 已不连续）> 普通 Linear checkpoint（ref-count=0，最久未命中）> Full KV block（ref-count=0）> request preemption。

**核心逻辑**：淘汰 Full KV 时级联清理失效的 Linear checkpoint；淘汰 Linear checkpoint 时保留 Full KV（Full KV 可被其他 prefix 复用）。`min(H_full, H_linear)` 作为运行时兜底，确保 match 结果始终有效。

### 3.5 命中策略

Prefix match 流程：

1. 按 token block 构建 cumulative hash。
2. Full KV Cache 查询最长 block hit → `H_full`。
3. Linear State Cache 查询 ≤ `H_full` 的最大 checkpoint_interval 对齐 checkpoint → `H_linear`。
4. `H_effective = min(H_full, H_linear)`，模型从 `H_effective` 开始 prefill。

Fallback 场景：

| 场景 | 策略 |
|---|---|
| Full KV miss | 从 0 prefill，Linear State 使用 zero state |
| Full KV hit + Linear State miss | 从 0 prefill（丢弃已命中 KV blocks，Linear State 不可从 KV 重建） |
| 全命中 + 需 logits | 从最近的 checkpoint 恢复，重算 suffix 到最后 token |
| Chunked continuation | 使用 request live state，不重复 match |

## 4. 涉及的算子

| 算子 | 改动 |
|---|---|
| `chunk_gated_delta_rule` | 支持非零 initial state（修复 `fill_(0.0)`），保持只输出 final state，写入 live state slot |
| `linear_state_copy`（新增/扩展） | checkpoint slot ↔ live slot 之间的 NPU 显存拷贝（cache hit 时 checkpoint→live，边界时 live→checkpoint） |

要求：conv state 与 ssm state 均支持批量拷贝；NPU tensor 地址稳定适配 ACL Graph replay；热路径避免 CPU/NPU 同步（如 `tensor.item()`）；forward 中不动态分配大 tensor。
