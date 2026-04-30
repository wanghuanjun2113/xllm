# xLLM Qwen3.5 Prefix Caching on NPU 技术设计

## 1. 背景介绍

目标是在 xLLM 中支持 Qwen3.5 混合注意力架构在 NPU 上的 Prefix Caching：

- Full Attention/MLA 路径复用 KV Cache。
- Linear Attention/Gated DeltaNet 路径复用 conv state 与 ssm state。
- Chunked Prefill 与 Linear State checkpoint 对齐。
- NPU 显存采用稳定地址、固定 slot、低碎片的缓存池设计。

相关 xLLM 实现位置：

| 模块 | 文件 |
|---|---|
| Qwen3.5 模型参数加载 | `xllm/xllm/models/llm/qwen3_5.h` |
| 混合层选择 | `xllm/xllm/core/layers/npu_torch/qwen3_next_hybrid_decoder_layer_base.cpp` |
| Gated DeltaNet | `xllm/xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp` |
| Linear KV/State shape | `xllm/xllm/core/framework/kv_cache/kv_cache_shape.cpp` |
| Prefix Cache | `xllm/xllm/core/framework/prefix_cache/prefix_cache.cpp` |
| Block Manager | `xllm/xllm/core/framework/block/block_manager_impl.cpp` |
| Chunked Prefill Scheduler | `xllm/xllm/core/scheduler/chunked_prefill_scheduler.cpp` |

当前 `qwen3_gated_delta_net_base.cpp` prefill 路径中存在：

```cpp
// Todo: chunked-prefill/prefix-cache use initial_state.
initial_state_tensor.fill_(0.0);
```

该逻辑会破坏 Linear Attention Prefix Cache，需要改为从缓存 state slot 读取 initial state。

## 2. 总体方案

### 2.1 架构数据基准

模型参数以 Hugging Face `config.json` 为准：

- [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json)
- [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)
- [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)

| 模型 | 类型 | 层数 | Full 间隔 | Full 层 | Linear 层 | Hidden | Attn Heads | KV Heads | Head Dim | Linear K Heads | Linear V Heads | Linear Dim | MoE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5-27B | Dense | 64 | 4 | 16 | 48 | 5120 | 24 | 4 | 256 | 16 | 48 | 128 | N/A |
| Qwen3.5-35B-A3B | MoE | 40 | 4 | 10 | 30 | 2048 | 16 | 2 | 256 | 16 | 32 | 128 | 256 experts, top-8 |
| Qwen3.5-397B-A17B | MoE | 60 | 4 | 15 | 45 | 4096 | 32 | 2 | 256 | 16 | 64 | 128 | 512 experts, top-10 |

通用参数：

| 参数 | 值 |
|---|---|
| `linear_conv_kernel_dim` | 4 |
| `max_position_embeddings` | 262144 |
| `mamba_ssm_dtype` | float32 |
| `dtype` | bfloat16 |
| `mrope_interleaved` | true |
| `rope_theta` | 10000000 |
| `partial_rotary_factor` | 0.25 |

### 2.2 Full Attention KV Cache 容量

BF16 下单 token 单 Full Attention 层 KV Cache：

```text
KV/token/layer = 2 * num_key_value_heads * head_dim * sizeof(bf16)
```

| 模型 | KV Heads | Head Dim | 单 Full 层 KV/token | Full 层数 | 全 Full 层 KV/token |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-27B | 4 | 256 | 4 KiB | 16 | 64 KiB |
| Qwen3.5-35B-A3B | 2 | 256 | 2 KiB | 10 | 20 KiB |
| Qwen3.5-397B-A17B | 2 | 256 | 2 KiB | 15 | 30 KiB |

TP 场景按 local KV heads 缩放；若 TP 大于 KV heads，需结合 xLLM `get_local_head_count` 的 `max(1, heads / tp)` 语义处理。

### 2.3 Linear Attention State 容量

xLLM 当前 Linear State shape：

```text
conv_cache:
[num_slots, conv_kernel_dim - 1,
 linear_key_dim * linear_k_heads * 2 + linear_key_dim * linear_v_heads]

ssm_cache:
[num_slots, linear_v_heads, linear_key_dim, linear_value_dim]
```

容量计算：

```text
conv_state/layer =
  (conv_kernel_dim - 1)
  * (linear_key_dim * linear_k_heads * 2
     + linear_key_dim * linear_v_heads)
  * sizeof(bf16)

ssm_state/layer =
  linear_v_heads * linear_key_dim * linear_value_dim * sizeof(float32)
```

| 模型 | Linear V Heads | Conv/层 | SSM/层 | 单 Linear 层 State | Linear 层数 | 单 checkpoint | 默认 1024-token 摊销 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-27B | 48 | 60 KiB | 3 MiB | 3.059 MiB | 48 | 146.8 MiB | 146.8 KiB/token |
| Qwen3.5-35B-A3B | 32 | 48 KiB | 2 MiB | 2.047 MiB | 30 | 61.4 MiB | 61.4 KiB/token |
| Qwen3.5-397B-A17B | 64 | 72 KiB | 4 MiB | 4.070 MiB | 45 | 183.2 MiB | 183.2 KiB/token |

结论：Linear State 单 checkpoint 体积远大于 Full KV 单 token，必须采用稀疏 checkpoint。

## 3. 详细设计

### 3.1 Linear Attention 状态缓存

#### 3.1.1 Sparse Caching

Linear State 只在固定间隔缓存。由于当前 `chunk_gated_delta_rule`
只输出本次 prefill chunk 的 final state，不输出中间 state，
因此 checkpoint 间隔必须和调度切片边界一致：

```text
checkpoint_interval =
  long_prefill_token_threshold > 0 ? long_prefill_token_threshold : 1024

checkpoint position = k * checkpoint_interval
```

本设计不新增独立的 `linear_state_block_size` 用户参数。
`long_prefill_token_threshold` 在 Qwen3.5 Linear Prefix Cache 中同时承担两个职责：

- 调度层长 prefill 单步 token 上限。
- Linear State Prefix Cache 的 checkpoint interval。

选择依据是命中粒度与主机内存占用的平衡：

- **粒度越小**（如 256）：checkpoint 更密集，prefix 命中率更高，但主机内存占用更大（以 Qwen3.5-27B 为例，每 256 token 一个 checkpoint 约 36.7 MiB，1024 token 的 prefix 需 146.8 MiB 主机内存）。
- **粒度越大**（如 4096）：主机内存占用低，但 prefix hit 只能对齐到稀疏位置，浪费已计算的 Full KV tokens。
- **1024 是默认平衡点**：当 `long_prefill_token_threshold == 0`
  时使用 1024。Full KV block size 通常为 16，1024 是其整数倍，保证 hash chain 对齐；同时以 Qwen3.5-27B 为例，每 checkpoint 约 146.8 MiB 主机内存，128K prefix 约需 18 GiB，在典型服务器上可接受。

```text
配置方式：
  long_prefill_token_threshold > 0 时，作为 checkpoint_interval
  long_prefill_token_threshold == 0 时，checkpoint_interval = 1024

约束：
  checkpoint_interval % kv_cache_block_size == 0
  checkpoint_interval <= max_num_batched_tokens
```

注意：vLLM 中 `mamba_block_size` 和 `long_prefill_token_threshold`
是两个独立概念；但在本 xLLM NPU 方案里，Gated DeltaNet prefill
只输出 final state，实际 checkpoint 只能落在 scheduler chunk 结束位置。
因此采用 `long_prefill_token_threshold` 作为 checkpoint interval，
避免出现“配置了更小 state block，但算子无法生成中间 state”的误导。

checkpoint 内容：

- 所有 Linear 层的 `ssm_state`。
- 所有 Linear 层的 `conv_state`。
- 对应该 token 边界的 cumulative prefix hash。

Full Attention KV Cache 和 Linear State Cache 共用同一条 prefix hash 链：

```text
hash_i = Hash(hash_{i-1}, tokens_in_block_i)
```

不同的是 value 类型不同：

| Cache | Key | Value |
|---|---|---|
| Full Attention KV | cumulative prefix hash | KV block ids |
| Linear Attention State | `checkpoint_interval` 对齐边界的同一 prefix hash | host checkpoint entry id |

可选的 `model_id`、adapter、rope、layout version 等信息应作为全局 cache namespace，而不是 Linear Attention 单独 hash。

#### 3.1.2 与 Chunked Prefill 协同

Qwen3.5 参考 vLLM `mamba_cache_mode=align`，不采用 `all` 模式。
原因是 `chunk_gated_delta_rule` 只输出 final state，不输出中间 state。
这意味着 Linear State 只能在每个 chunk 的结束位置缓存；
如果希望每 1024 token 缓存一次 state，就必须让调度 chunk 在 1024 token 边界结束。

核心策略：

```text
Linear State 只在 chunk 结束位置对齐到 checkpoint_interval 边界时缓存。
```

执行流程：

1. Prefix match 得到可复用长度 `H`，其中 `H` 必须命中 `checkpoint_interval` 对齐的 Linear State checkpoint。
2. 本次 prefill 从 `H` 开始。
3. Linear Attention 从 `prev_state_slot` 读取 initial state：
   - cache hit：先将 host checkpoint restore 到 live slot，再读取 live slot；
   - chunked continuation：读取上一 chunk 的 live slot；
   - cache miss：使用 zero state。
4. `chunk_gated_delta_rule` 计算当前 chunk，只输出 final state。
5. final state 写入当前 request 的 `live_state_slot`。
6. 若 chunk 结束位置满足 `pos % checkpoint_interval == 0`，将 `live_state_slot` 注册为 Linear State checkpoint。
7. 后续 decode 或下一 chunk 继续从 `live_state_slot` 读取 state。

最终语义：

```text
read state:  prev_state_slot
write state: live_state_slot
cache state: only when chunk_end % checkpoint_interval == 0
```

`qwen3_gated_delta_net_base.cpp` 需要删除：

```cpp
initial_state_tensor.fill_(0.0);
```

改为：

```text
has_initial_state ? state[prev_state_slot] : zero_state
```

Conv state 必须和 SSM state 一起缓存和恢复，因为卷积分支依赖最近 `conv_kernel_dim - 1` 个 token。

#### 3.1.3 调度 token 数约束

参考 vLLM scheduler，Qwen3.5 Linear Attention 开启 `align` 后，单步调度 token 数按两步确定。
这里 `checkpoint_interval` 由 `long_prefill_token_threshold` 解析得到：

1. 先用 `long_prefill_token_threshold` 限制长 prefill 单步 token 数。
2. 再按 `checkpoint_interval` 对齐，保证 chunk final state 正好可作为 checkpoint。

```text
num_new_tokens = remaining_tokens

checkpoint_interval =
  long_prefill_token_threshold > 0 ? long_prefill_token_threshold : 1024

num_new_tokens = min(num_new_tokens, checkpoint_interval)

num_new_tokens = min(num_new_tokens, token_budget)
num_new_tokens = align_to_checkpoint_interval(request, num_new_tokens)
```

Block 对齐逻辑：

```text
block_size = checkpoint_interval
computed = request.num_computed_tokens
after = computed + num_new_tokens
last_cache_position = prompt_len - prompt_len % block_size

if after < last_cache_position:
  num_new_tokens = floor(num_new_tokens / block_size) * block_size
elif computed < last_cache_position < after:
  num_new_tokens = last_cache_position - computed
else:
  keep num_new_tokens
```

语义：

| 场景 | 行为 |
|---|---|
| 未到最后可缓存边界 | 向下对齐到 `block_size` 倍数 |
| 会跨过最后可缓存边界 | 截断到 `last_cache_position` |
| prompt 尾部不足一个 block | 允许执行，但不写 Linear checkpoint |
| 中间阶段 `num_new_tokens < block_size` | 不调度，等待足够 token budget，避免产生不可缓存的中间 chunk |

配置约束：

```text
checkpoint_interval <= max_num_batched_tokens
```

如果后续 NPU 算子支持输出中间 state，可再引入独立的 state checkpoint
粒度参数；当前版本不这样设计。

若 Linear State 与 Full KV 共用 prefix hash，建议：

```text
checkpoint_interval % kv_cache_block_size == 0
```

### 3.2 NPU Buffer 内存池设计

#### 3.2.1 vLLM Unified Buffer 对比

vLLM Unified Buffer 适合所有层 KV shape 一致的 Attention Cache：

```text
HBM low                                                     HBM high
|------------------------------------------------------------------|
| B0-L0 KV | B0-L1 KV | ... | B0-LN KV | B1-L0 KV | B1-L1 KV | ... |
|------------------------------------------------------------------|
```

Qwen3.5 同时存在 Full KV 和 Linear State：

```text
Full KV Block:
| token0 KV | token1 KV | ... | tokenN KV |

Linear State Checkpoint:
| all linear layers conv_state | all linear layers ssm_state |
```

二者大小、生命周期、释放粒度均不同，不适合混在一个 Unified Buffer 中。

#### 3.2.2 SGLang Separated Buffer 对比

SGLang 对 Mamba/Linear State 使用独立缓存池，显存形态更适合 Qwen3.5：

```text
HBM low                                                     HBM high
|------------------|-------------------|-------------------|
| Attention KV Pool| Conv State Pool   | SSM State Pool    |
|------------------|-------------------|-------------------|
```

其中：

```text
Attention KV Pool:
| KV Block0 | KV Block1 | KV Block2 | ... |

Conv State Pool:
| Slot0 Conv | Slot1 Conv | Slot2 Conv | ... |

SSM State Pool:
| Slot0 SSM  | Slot1 SSM  | Slot2 SSM  | ... |
```

#### 3.2.3 xLLM 推荐布局

xLLM 建议采用三个 NPU HBM Pool。Linear Prefix checkpoint 不常驻显存，放在主机内存中。

```text
HBM low                                                     HBM high
|------------------|------------------|------------------|
| Full KV Pool     | Conv Live Pool   | SSM Live Pool    |
|------------------|------------------|------------------|
```

`Full KV Pool` 继续复用现有 block manager。

`Conv/SSM Live Pool` 只保存当前运行请求的 Linear State：

```text
| Slot0(pad) | Slot1(req0) | Slot2(req1) | ... |
```

`checkpoint_interval` 对齐的 Linear Prefix checkpoint 放主机内存：

```text
Host Prefix State Cache:
| hash_{1 * checkpoint_interval} -> conv/ssm checkpoint |
| hash_{2 * checkpoint_interval} -> conv/ssm checkpoint |
| ...                              |
```

访问方式：

```text
cache hit:
  async copy Host Prefix State -> Live State Slot
  write Live State Slot

chunked prefill:
  read  Previous Live State Slot
  write Current Live State Slot

checkpoint_interval boundary:
  async copy Live State Slot -> Host Prefix State Cache
```

推荐物理 shape：

```text
conv_pool:
[num_slots, num_linear_layers, conv_kernel_dim - 1, conv_dim]

ssm_pool:
[num_slots, num_linear_layers, local_v_heads, key_dim, value_dim]
```

#### 3.2.4 Pool 空间分配逻辑

核心思路：不要让 Full KV Pool 单独吃满显存，而是按 Full Attention
和 Linear Attention 的缓存成本比例切分空间。这个比例由模型参数和
`checkpoint_interval` 决定。

以一段 `checkpoint_interval` tokens 的 prefix 为单位：

```text
full_kv_bytes_per_interval =
  checkpoint_interval * full_kv_bytes_per_token

conv_bytes_per_checkpoint =
  sum(conv_state bytes of all linear layers)

ssm_bytes_per_checkpoint =
  sum(ssm_state bytes of all linear layers)

buffer_ratio =
  full_kv_bytes_per_interval
  : conv_bytes_per_checkpoint
  : ssm_bytes_per_checkpoint
```

默认 `checkpoint_interval = 1024` 时，三类 Buffer 的近似比例为：

| 模型 | Full KV / interval | Conv checkpoint | SSM checkpoint | 近似比例 |
|---|---:|---:|---:|---|
| Qwen3.5-27B | 64 MiB | 2.8 MiB | 144 MiB | 64 : 2.8 : 144 |
| Qwen3.5-35B-A3B | 20 MiB | 1.4 MiB | 60 MiB | 20 : 1.4 : 60 |
| Qwen3.5-397B-A17B | 30 MiB | 3.2 MiB | 180 MiB | 30 : 3.2 : 180 |

当 `long_prefill_token_threshold` 调大时，`full_kv_bytes_per_interval`
会随 `checkpoint_interval` 线性增大，而单个 Conv/SSM checkpoint
大小不变，因此 Full KV Pool 的比例会提高。

分配流程：

1. profile 得到模型加载、graph/workspace、安全水位之后的可用显存。
2. 用上面的 `buffer_ratio` 将可用显存切成 Full KV、Conv、SSM 三份。
3. 按各自分配粒度向下对齐：
   - Full KV 按 `kv_cache_block_size` 对齐。
   - Conv/SSM 按完整 checkpoint slot 对齐。
4. Conv/SSM Pool 还必须满足运行态 live slot 下限。

```text
live_slots = 1 + max_running_requests + extra_live_slots

available_hbm = profile_after_model_load()
reserve_hbm   = graph + workspace + safety_margin
pool_hbm      = available_hbm - reserve_hbm

conv_live_min = live_slots * conv_bytes_per_checkpoint
ssm_live_min  = live_slots * ssm_bytes_per_checkpoint

cache_hbm = pool_hbm - conv_live_min - ssm_live_min

ratio_sum =
  full_kv_bytes_per_interval
  + conv_bytes_per_checkpoint
  + ssm_bytes_per_checkpoint

full_kv_pool_bytes =
  cache_hbm * full_kv_bytes_per_interval / ratio_sum

conv_pool_bytes =
  conv_live_min
  + cache_hbm * conv_bytes_per_checkpoint / ratio_sum

ssm_pool_bytes =
  ssm_live_min
  + cache_hbm * ssm_bytes_per_checkpoint / ratio_sum
```

分配完成后做一次有效容量校验：

```text
full_kv_token_capacity =
  floor(full_kv_pool_bytes / full_kv_bytes_per_token)

linear_checkpoint_capacity =
  min(floor(conv_pool_bytes / conv_bytes_per_checkpoint),
      floor(ssm_pool_bytes / ssm_bytes_per_checkpoint))

linear_state_token_capacity =
  linear_checkpoint_capacity * checkpoint_interval

effective_prefix_capacity =
  min(full_kv_token_capacity, linear_state_token_capacity)
```

如果 `effective_prefix_capacity` 明显小于预期，说明比例分配后某一侧成为瓶颈：

- Full KV Pool 不足：增大 `checkpoint_interval` 或提高 Full KV 分配权重。
- Conv/SSM Pool 不足：减小 `checkpoint_interval` 或提高 Linear State 分配权重。
- live slot 下限过高：降低 `max_running_requests` 或增加可用 HBM，否则没有足够空间承载可缓存 prefix。

当前推荐布局中，Linear Prefix checkpoint 可放在 Host 内存中；
此时 Conv/SSM HBM Pool 至少保证 live slot，Host Prefix State Cache
按同样的 Conv/SSM checkpoint 数量比例规划。若后续实现选择让
Linear checkpoint 常驻 HBM，上述比例可直接用于三个 HBM Buffer 的分配。

### 3.3 Prefix Matching 机制

xLLM 当前 `PrefixCache::match` 已按 block hash 做最长前缀匹配。需要扩展返回值：

```cpp
struct PrefixMatchResult {
  std::vector<Block*> kv_blocks;
  int64_t full_kv_hit_tokens;
  int64_t linear_state_hit_tokens;
  int32_t linear_state_host_entry_id;
};
```

匹配流程：

```text
1. CPU 侧按 token block 构建 cumulative hash。
2. Full KV Cache 查询最长 block hit，得到 H_full。
3. Linear State Cache 查询 <= H_full 的最大 `checkpoint_interval` 对齐 checkpoint，得到 H_linear。
4. H_effective = min(H_full, H_linear)。
5. 模型从 H_effective 开始 prefill。
```

Fallback：

| 场景 | 策略 |
|---|---|
| Full KV miss | 从 0 prefill，Linear State 使用 zero state |
| Full KV hit，Linear State miss | 从 0 prefill（丢弃已命中的 Full KV blocks，重新计算） |
| Full KV 与 Linear State partial hit | 从 `H_effective` prefill |
| prompt 全命中且 `prompt_len - 1` 命中 Linear checkpoint | 保留最后 token 重新计算 logits |
| prompt 全命中但 `prompt_len - 1` 未命中 Linear checkpoint | 从最近的 `checkpoint_interval` 对齐 Linear checkpoint 恢复，重算 suffix 到最后 token |
| chunked continuation | 优先使用 request live state，不重复全局 match |

**Full hit logits 重算对齐规则**：

Prefix Cache 全命中时，仍需要重新计算 prompt 最后一个 token 的 logits。
但 Linear State 只在 `checkpoint_interval` 对齐边界保存 checkpoint，
因此不能无条件只重算最后一个 token。

```text
logits_token_pos = prompt_len - 1
H_recompute =
  floor(logits_token_pos / checkpoint_interval) * checkpoint_interval
```

执行策略：

1. 若 `H_recompute > 0`，查询并恢复 `H_recompute` 对应的 Linear State checkpoint；否则使用 zero state。
2. 复用前 `H_recompute` 个 token 对应的 Full KV blocks。
3. 从 `H_recompute` 开始重算到 `prompt_len - 1`，产出最后 token logits。
4. 若 `prompt_len - 1` 本身恰好是 checkpoint 边界，则 suffix 长度为 1，只重算最后 token。

该规则避免了 `prompt_len - 1` 非 `checkpoint_interval` 对齐时缺少 Linear initial state 的问题。

**Full KV hit + Linear State miss 的代价分析**：

该场景意味着 Linear State checkpoint 被驱逐或从未生成。由于 Linear Attention 的 state 不可从 KV Cache 重建，必须从 token 0 重新 prefill，已命中的 Full KV blocks 被丢弃。

代价量化（以 Qwen3.5-27B、Full KV 命中 4096 tokens 为例）：

```text
节省的 Full KV prefill: 4096 tokens × 16 Full layers 的 attention 计算
浪费的代价: 丢弃 4096 tokens × 64 KiB/token = 256 MiB Full KV cache blocks（可被其他请求复用，非真正浪费）
实际开销: 从 0 重新 prefill 全部 4096 tokens 的 Linear Attention（48 layers）

结论: 主要是延迟惩罚，Full KV blocks 不会被真正释放（仍可被其他 prefix 复用）
```

缓解措施：

- eviction 策略优先保留最近使用的 Linear State checkpoint（见 3.4）。
- 监控指标中单独统计此场景频率，若频率过高则增大主机内存或减小 `long_prefill_token_threshold` 对应的 `checkpoint_interval`。

### 3.4 Cache Eviction 策略

Full KV blocks 和 Linear State checkpoints 采用独立的 LRU 驱逐，但驱逐决策需要联动：

**Full KV Block 驱逐**：

- 沿用现有 block manager 的 ref-count + LRU 机制。
- 驱逐某 block 时，不主动联动 Linear State checkpoint。Full KV 的驱逐粒度是 token block（如 16 tokens），Linear State 的粒度是 `checkpoint_interval` tokens，两者不一定对齐。

**Linear State Checkpoint 驱逐**：

- 独立 LRU，以 checkpoint entry（`hash → conv/ssm data`）为单位。
- 驱逐条件：主机内存 checkpoint slot 不足时，淘汰最久未被 match 命中的 entry。
- 保护规则：当前有 active request 正在从该 checkpoint restore 的 entry 不可驱逐（ref-count > 0）。
- **不联动驱逐 Full KV blocks**：Linear State checkpoint 被驱逐后，对应的 Full KV blocks 仍然有效，只是后续请求无法从该位置复用 Linear State，需回退到更早的 checkpoint 或从 0 prefill。

**联动约束**：

- 当 Full KV blocks 被驱逐到某个位置 `P` 之后，位于 `P` 之前的 Linear State checkpoint 即使保留也失去意义（Full KV 缺失导致无法从该位置开始 prefill）。
- 可选优化：在 Full KV blocks 驱逐后，主动清理对应位置之前的 Linear State checkpoint。此优化可延后实现（P1），初始版本允许不一致，依赖 prefix match 的 `min(H_full, H_linear)` 逻辑自动处理。

**驱逐优先级**（当需要回收空间时）：

```text
1. ref-count == 0 且最久未命中的 Linear State checkpoint
2. ref-count == 0 且最久未命中的 Full KV block
3. 若以上仍不足，触发 request preemption 释放 live state slot
```

### 3.5 接口与调度改动

Batch 输入从单一 `linear_state_ids` 扩展为：

```cpp
std::vector<int32_t> linear_prev_state_ids;
std::vector<int32_t> linear_live_state_ids;
std::vector<uint8_t> linear_has_initial_state;
std::vector<uint8_t> linear_store_state;
std::vector<int64_t> linear_num_computed_tokens;
```

Scheduler 负责：

- prefix match 后确定 `H_effective`。
- 控制 `num_new_tokens` 不跨 `checkpoint_interval` 边界。
- 分配 live state slot。
- cache hit 时触发 Host Prefix State 到 live state slot 的异步恢复。
- 在 `checkpoint_interval` 边界将 live state 异步写入 Host Prefix State Cache。

Model Runner 负责：

- forward 前准备 `prev_state_slot` 与 `live_state_slot`。
- Linear layer 从 `prev_state_slot` 读 initial state。
- final state 写入 `live_state_slot`。
- 边界处执行 live slot 到主机 checkpoint 的 copy/注册。

### 3.6 模块与代码改动清单

**P0 — 必须实现（核心路径，无此无法运行）：**

新增：

```text
xllm/xllm/core/framework/prefix_cache/linear_state_prefix_cache.h
xllm/xllm/core/framework/prefix_cache/linear_state_prefix_cache.cpp
xllm/xllm/core/framework/block/linear_state_block_manager.h
xllm/xllm/core/framework/block/linear_state_block_manager.cpp
```

修改：

```text
# Linear layer initial state 读取（修复 fill_(0.0) 问题）
xllm/xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp
xllm/xllm/core/layers/npu_torch/qwen3_5_gated_delta_net.cpp

# Batch 输入扩展（新增 linear state 相关字段）
xllm/xllm/core/framework/batch/batch_input_builder.cpp
xllm/xllm/core/framework/model/model_input_params.h
xllm/xllm/proto/worker.proto

# KV cache shape 支持 Linear State
xllm/xllm/core/framework/kv_cache/kv_cache_shape.cpp
xllm/xllm/core/framework/kv_cache/kv_cache_utils.cpp
xllm/xllm/core/framework/kv_cache/linear_attention_kv_cache_impl.{h,cpp}

# Prefix match 返回值扩展
xllm/xllm/core/framework/prefix_cache/prefix_cache.{h,cpp}

# Chunked prefill 调度对齐逻辑
xllm/xllm/core/scheduler/chunked_prefill_scheduler.cpp
```

**P1 — 建议实现（内存池、调度联动、eviction）：**

```text
# 三池分配与 live state slot 管理
xllm/xllm/core/framework/block/block_manager*.{h,cpp}

# Continuous batching 调度适配
xllm/xllm/core/scheduler/continuous_scheduler.cpp

# 运行时初始化与参数解析
xllm/xllm/core/runtime/worker_impl.cpp
xllm/xllm/core/runtime/params_utils.cpp
xllm/xllm/core/distributed_runtime/llm_engine.cpp

# Forward shared memory 扩展
xllm/xllm/core/runtime/forward_shared_memory_manager.*
```

**P2 — 优化（Full KV 与 Linear State 联动驱逐、监控）：**

```text
# 联动驱逐优化（Full KV 驱逐后主动清理对应 Linear State checkpoint）
xllm/xllm/core/framework/prefix_cache/prefix_cache.{h,cpp}
xllm/xllm/core/framework/block/linear_state_block_manager.{h,cpp}
```

### 3.7 NPU 算子改动

`chunk_gated_delta_rule`：

- 支持非零 initial state。
- 保持只输出 final state。
- final state 写入 live state slot。
- 不要求输出中间 checkpoint。

新增或扩展 NPU copy kernel：

```text
linear_state_store:
  live slot -> pinned host checkpoint

linear_state_restore:
  pinned host checkpoint -> live slot
```

优化要求：

- conv state 与 ssm state 均支持批量 H2D/D2H copy。
- NPU tensor 地址稳定，适配 ACL Graph replay。
- 避免热路径 `tensor.item()` 触发 CPU/NPU 同步。
- 避免 forward 中动态分配大 tensor。

### 3.8 错误处理

| 故障场景 | 处理策略 |
|---|---|
| Host Prefix State H2D copy 失败 | 标记该 checkpoint entry 为 corrupted，从最近的有效 checkpoint 重新 restore；若无有效 checkpoint，使用 zero state 从 0 prefill。记录错误日志和指标。 |
| Host Prefix State D2H copy 失败（store） | 不影响当前请求（live state 仍在 NPU），仅放弃本次 checkpoint 注册。下一 chunk 边界重试。 |
| Host 内存不足，无法分配 checkpoint slot | 触发 LRU 驱逐，若仍不足则放弃写入 checkpoint，该请求的 Linear State 不缓存，后续请求无法复用此段 prefix。 |
| NPU Live State Slot 分配失败 | 触发 request preemption：选择最低优先级请求，将其 live state copy 到 host 做临时 checkpoint（若有空间），释放 slot 给高优先级请求。被抢占请求恢复时重新 prefill。 |
| `chunk_gated_delta_rule` 执行失败 | 当前 chunk 标记为 failed，不写入 live state slot。该请求回退到上一个有效 checkpoint 位置重新 prefill。 |
| Full KV Pool OOM | 沿用现有 block manager 的 preemption 逻辑。被 preempt 的 Full KV blocks 释放后，对应的 Linear State checkpoint 在下次驱逐时自然淘汰。 |

通用原则：

- 所有 fallback 路径保证**正确性优先**：宁可重新 prefill，不用错误的 state。
- 错误事件通过 metric 上报（`linear_state_cache_errors{type="h2d_failed"}`），监控告警。
- 不引入重试循环：单次操作失败后走 fallback 路径，不在 forward 路径中重试。

## 4. 测试设计

### 4.1 正确性验证

E2E 对比：

```text
cache disabled vs cache enabled
```

覆盖：

| 场景 | 验证点 |
|---|---|
| no hit | 与原始 prefill 输出一致 |
| full hit | tokens/logits 一致 |
| partial hit | hit 边界后输出一致 |
| prompt < checkpoint_interval | 不写 Linear checkpoint |
| prompt = checkpoint_interval | 写入一个 checkpoint |
| prompt > checkpoint_interval | 多 checkpoint 写入与复用 |
| checkpoint_interval 256/1024/1536 | 调度边界正确 |
| 多请求 batch | state slot 不串扰 |
| cache eviction | ref count 与 LRU 正确 |
| preemption/resume | fallback 到最近 checkpoint |
| TP/DP | local heads 与 slot 索引正确 |

数值标准：

```text
greedy output tokens 完全一致
logits max/mean error 满足 BF16 容忍阈值
```

### 4.2 单元测试

- prefix hash 与 block match。
- Full KV hit 与 Linear State hit 合并。
- `checkpoint_interval` 边界调度切分。
- live slot 分配释放。
- Host Prefix State Cache 分配、LRU 驱逐与 H2D/D2H restore/store。
- conv state 与 ssm state restore/store。
- LRU eviction 不释放 active slot。

### 4.3 NPU 性能评估

指标：

| 指标 | 说明 |
|---|---|
| TTFT | 不同命中率下首 token 延迟 |
| Throughput | requests/s 与 tokens/s |
| HBM peak | NPU 显存峰值 |
| HBM fragmentation | 最大连续空闲块 / 总空闲块 |
| State pool utilization | live slot 使用率 |
| Host prefix utilization | 主机 checkpoint slot 使用率 |
| Prefix hit rate | Full KV 与 Linear State 分开统计 |
| Copy overhead | state H2D/D2H restore/store 时间 |
| Kernel time | `chunk_gated_delta_rule` 时间 |
| CPU sync count | `.item()` 等同步点数量 |
| ACL Graph replay rate | graph replay 命中率 |

测试矩阵：

```text
prompt length: 1K / 4K / 32K / 128K
hit rate: 0% / 25% / 50% / 75% / 100%
checkpoint_interval: 256 / 1024 / 1536 / 4096
batch size: 1 / 8 / 32 / max_seqs_per_batch
```

预期：

- Full KV 与 Linear State 同时命中时，TTFT 随命中长度明显下降。
- 稀疏 checkpoint 控制 Linear State 显存增长。
- state copy 开销显著小于从 0 recompute 的 Linear Attention 成本。
- 长时间压测下 HBM 碎片率稳定。
