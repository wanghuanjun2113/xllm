# RFC: Qwen3.5 27B Flash Communication 1 (FC1) 支持

| 字段 | 值 |
|------|-----|
| 作者 | wanghuanjun |
| 日期 | 2026-05-08 |
| 状态 | Draft |
| 目标模型 | Qwen3.5-27B (Dense, 混合全注意力 + Gated Delta Net) |
| 目标平台 | Ascend NPU |
| TP 配置 | TP=2, TP=4, TP=8 |

## 1. 概述

本 RFC 提出在 xllm 推理框架中为 Qwen3.5-27B 模型增加 Flash Communication 1 (FC1) 能力的设计方案。

FC1 是面向张量并行 (Tensor Parallel, TP) 推理的通信优化，核心思想是将层间 hidden states 从每个 TP rank 保存完整 sequence `[S, H]` 改为每个 TP rank 只保存 sequence 维的一段 `[S/TP, H]`。通过将 row-parallel 层输出后的 `all_reduce` 替换为 `reduce_scatter`，并在 column-parallel 层输入前按需执行 `all_gather`，减少冗余计算和访存。

Qwen3.5-27B 是混合架构模型，包含全注意力（Full Attention）层和线性注意力（Gated Delta Net, GDN）层。两类层的 FC1 适配策略不同，本方案完整覆盖两种层类型。

## 2. 背景与动机

### 2.1 当前问题

xllm 中 RowParallelLinear（`o_proj`、`down_proj`）的 forward 在 TP 场景下使用 `parallel_state::reduce()`（即 `all_reduce`），每个 rank 都得到完整的 `[S, H]` 输出。这意味着：

- 每个 TP rank 对完整 sequence 重复执行 RMSNorm、residual add 等操作
- 层间 hidden states 和 residual tensor 占用 `[S, H]` 内存（每个 rank 相同）
- 通信数据量为 `S * H * sizeof(dtype)` 字节的 all_reduce

### 2.2 FC1 收益

1. **RMSNorm 和 residual 计算量减少**：TP=4 时，每 rank 从处理 `[S, H]` 降为 `[S/4, H]`
2. **层间 activation 访存降低**：hidden states 和 residual 按 sequence 维切分，大 batch prefill 场景下内存带宽压力降低
3. **量化场景通信量降低**：`reduce_scatter → RMSNorm → Quant → all_gather` 路径中，all_gather 前可完成量化，通信数据量从 BF16 降至 INT8
4. **Matmul + ReduceScatter 融合**：支持的 NPU 算子可将 matmul 和 reduce_scatter 融合为单次 kernel 调用，减少中间 tensor 产生和访存

### 2.3 Qwen3.5-27B 模型架构特点

Qwen3.5-27B 采用混合注意力架构，每个 decoder layer 要么是全注意力层，要么是 Gated Delta Net 层：

- **全注意力层**：RMSNorm → Attention（QKVParallelLinear + Attention + RowParallelLinear）→ RMSNorm → DenseMLP（ColumnParallelLinear + Activation + RowParallelLinear）
- **GDN 层**：RMSNorm → GDN（4 个 ColumnParallelLinear 投影 + conv1d + delta rule + GatedRMSNorm + RowParallelLinear）→ RMSNorm → DenseMLP

GDN 层有 4 个独立的 column-parallel 投影（`in_proj_qkv`、`in_proj_z`、`in_proj_b`、`in_proj_a`），其内部计算（causal conv1d、delta rule、A_log gating）需要在完整 sequence 上执行，FC1 适配需确保 AllGather 在 projection 前完成。

## 3. 总体方案

### 3.1 设计原则

采用**标志位动态分发**方案：在 `ParallelArgs` 中增加 FC1 启用标志，线性层的 `forward()` 根据标志选择 `reduce_scatter` / `all_gather` 路径。该方案与 xllm 现有的 `enable_result_reduction` 标志位模式一致，实现简洁，风格统一。

### 3.2 通信变换规则

| 位置 | 无 FC1 | 有 FC1 |
|------|--------|--------|
| RowParallel 输出（`o_proj`, `down_proj`） | `all_reduce` → `[S, H]` | `reduce_scatter` → `[S/TP, H]` |
| ColumnParallel 输入（`qkv_proj`, `gate_up_proj` 等） | 直接使用 `[S, H]` | `all_gather` → 恢复 `[S, H]` |
| 层间 hidden states | `[S, H]` | `[S/TP, H]` |
| RMSNorm 输入 | `[S, H]` | `[S/TP, H]` |
| Residual add | `[S, H]` + `[S, H]` | `[S/TP, H]` + `[S/TP, H]` |

### 3.3 层间数据形态

```text
不开 FC1:
  o_proj / down_proj → all_reduce → 每个 rank 得到 [S, H]

开启 FC1:
  o_proj / down_proj → reduce_scatter → 每个 rank 得到 [S/TP, H]
  qkv_proj / gate_up_proj 前 → all_gather → 每个 rank 临时恢复 [S, H]
```

## 4. 数据流设计

### 4.1 全注意力 + Dense MLP 层

以 Qwen3.5-27B TP=4、S=4096 为例：

```text
不开 FC1:
  hidden [4096,H] → RMSNorm [4096,H] → qkv_proj(ColP) → Attention → o_proj(RowP) → AllReduce → [4096,H]
  + residual [4096,H]
  → RMSNorm [4096,H] → gate_up_proj(ColP) [4096,2*17408/4] → SwiGLU [4096,17408/4] → down_proj(RowP) → AllReduce → [4096,H]
  + residual [4096,H]

开启 FC1:
  hidden [1024,H] → RMSNorm [1024,H] → AllGather [4096,H] → qkv_proj(ColP) → Attention → o_proj(RowP) → ReduceScatter → [1024,H]
  + residual [1024,H]
  → RMSNorm [1024,H] → AllGather [4096,H] → gate_up_proj(ColP) [4096,2*17408/4] → SwiGLU [4096,17408/4] → down_proj(RowP) → ReduceScatter → [1024,H]
  + residual [1024,H]
```

### 4.2 Gated Delta Net 层

GDN 层的 FC1 数据流：

```text
不开 FC1:
  hidden [4096,H] → RMSNorm [4096,H]
  → in_proj_qkv(ColP) + in_proj_z(ColP) + in_proj_b(ColP) + in_proj_a(ColP)
  → merge projections → conv1d → GDN计算 → GatedRMSNorm → o_proj(RowP) → AllReduce → [4096,H]
  + residual [4096,H]
  → RMSNorm [4096,H] → DenseMLP → AllReduce → [4096,H]
  + residual [4096,H]

开启 FC1:
  hidden [1024,H] → RMSNorm [1024,H] → AllGather [4096,H]
  → in_proj_qkv(ColP) + in_proj_z(ColP) + in_proj_b(ColP) + in_proj_a(ColP)
  → merge projections → conv1d → GDN计算(全sequence) → GatedRMSNorm → o_proj(RowP) → ReduceScatter → [1024,H]
  + residual [1024,H]
  → RMSNorm [1024,H] → AllGather [4096,H] → DenseMLP → ReduceScatter → [1024,H]
  + residual [1024,H]
```

**关键点**：GDN 内部的 causal conv1d、delta rule、A_log gating 等计算仍需完整 sequence `[S, H]`，通过 AllGather 在 4 个 column-parallel 投影前统一恢复。这些计算发生在投影之后、o_proj 之前，此时数据已经是完整 sequence 状态。

### 4.3 Residual 处理

所有 residual add 操作均在 sequence-sharded 状态下执行：

```text
residual:   [S/TP, H]  (来自上一层输出或上半个 block 的 ReduceScatter 输出)
模块输出:  [S/TP, H]  (来自当前模块的 ReduceScatter 输出)
result:     [S/TP, H]  (element-wise add，无需额外通信)
```

无需修改 residual add 逻辑，因为输入输出维度的 element-wise 操作天然兼容 sequence 维切分。

## 5. 关键接口设计

### 5.1 ParallelArgs 扩展

```cpp
// parallel_args.h
struct ParallelArgs {
    // 现有字段 ...
    int rank_;
    int world_size_;
    int tp_size_;
    // ...

    // FC1 新增字段
    bool enable_flash_comm_v1_ = false;  // 模型构建时设置，全局开关
    bool enable_mmrs_fusion_ = false;    // Matmul+ReduceScatter 融合开关
};
```

`enable_flash_comm_v1_` 在模型初始化阶段根据配置或环境变量设置，在模型生命周期内不变。

`enable_mmrs_fusion_` 在 TP ≤ 8 时自动启用（受 NPU 融合算子限制），TP > 8 时关闭。

### 5.2 运行时前向上下文

FC1 的实际生效还需结合运行时 token 数动态判断。建议在 BatchParams 或等效的 forward 上下文中增加：

```cpp
struct FlashComm1Context {
    bool enabled = false;
    int32_t tp_rank = 0;
    int32_t tp_world_size = 1;
    int32_t original_num_tokens = 0;   // 本次 forward 的真实 token 数
    int32_t padded_num_tokens = 0;     // pad 到 TP size 倍数后的 token 数
    int32_t local_num_tokens = 0;      // 本 rank reduce_scatter 后持有的 token 数
    std::vector<int32_t> token_num_list;  // 用于最后恢复真实 token 顺序
};
```

**生效条件**：

```text
全注意力层 / Dense MLP / GDN 层:
  flash_comm_v1_enabled =
    parallel_args.enable_flash_comm_v1_
    && num_tokens > THRESHOLD

THRESHOLD 建议值: 1000
```

当 `flash_comm_v1_enabled = false` 时，所有层走原有的 `all_reduce` 路径，FC1 不产生任何额外开销。

### 5.3 FC1 上下文构造

建议在模型入口或 runtime 层构造 FC1 context：

```cpp
FlashComm1Context build_flash_comm1_context(
    const ModelInputParams& input_params,
    const ParallelArgs& parallel_args,
    const ModelArgs& model_args);
```

职责：

- 判断是否启用 FC1（检查开关、设备类型、TP size、token 数阈值）
- 根据 `tokens.size(0)`、TP size 和 rank 计算 padding 与本地 shard 范围
- 为最终输出恢复准备 `token_num_list`

### 5.4 Sequence Shard Helper

建议增加一组工具函数，避免把 slice/pad/gather 逻辑散落到模型层：

```cpp
// Embedding 后：将完整 hidden 切成本 rank 的 sequence shard
torch::Tensor shard_sequence(
    const torch::Tensor& input,
    const FlashComm1Context& fc1_context);

// Column-parallel 前：将 sequence shard 恢复为完整 sequence
torch::Tensor gather_sequence(
    const torch::Tensor& input,
    const FlashComm1Context& fc1_context,
    ProcessGroup* tp_group);

// 模型输出前：恢复完整 sequence 并 unpad
torch::Tensor gather_and_unpad_sequence(
    const torch::Tensor& input,
    const FlashComm1Context& fc1_context,
    ProcessGroup* tp_group);
```

### 5.5 RowParallelLinear 通信模式

当前 `RowParallelLinearImpl` 用 `enable_result_reduction_` 控制是否执行 `parallel_state::reduce()`。FC1 需要把这个布尔语义扩展为显式通信模式：

```cpp
enum class RowParallelReduceMode : int8_t {
    NONE = 0,
    ALL_REDUCE = 1,
    REDUCE_SCATTER = 2,
    MATMUL_REDUCE_SCATTER = 3,
};
```

建议给 `RowParallelLinearImpl` 增加可运行时切换的接口：

```cpp
torch::Tensor forward(
    torch::Tensor input,
    RowParallelReduceMode reduce_mode,
    const FlashComm1Context* fc1_context = nullptr);
```

兼容要求：

- 现有 `forward(input)` 保持 `ALL_REDUCE` 或 `NONE` 语义不变
- FC1 enabled 时，`o_proj`、GDN `o_proj`、FFN `down_proj` 传入 `REDUCE_SCATTER` 或 `MATMUL_REDUCE_SCATTER`
- 不支持融合算子时，自动退化为 `matmul + parallel_state::reduce_scatter()`

### 5.6 ColumnParallelLinear 输入模式

首期不强行把 FC1 逻辑塞进 ColumnParallelLinear 类内部，而是在调用点显式处理：

```cpp
torch::Tensor full_hidden =
    fc1_context.enabled ? gather_sequence(hidden, fc1_context, tp_group)
                        : hidden;
auto qkv = qkv_proj_->forward(full_hidden);
```

这样风险更小，也更容易逐层验证。后续如果多个模型共享 FC1，可以再把 gather 逻辑沉淀到 `ColumnParallelLinearImpl` 的输入模式中。

## 6. 当前框架主要适配点

### 6.1 配置与开关

涉及文件：

- `xllm/core/common/global_flags.h`
- `xllm/core/common/global_flags.cpp`
- `xllm/core/common/options.h`
- `xllm/core/common/options.cpp`

建议增加：

- `enable_flashcomm1` 配置项
- `flashcomm1_min_prefill_tokens` 可配置阈值（默认 1000）
- 环境变量 `XLLM_ENABLE_FLASHCOMM1=1`

### 6.2 ModelInputParams / Forward Context

涉及文件：

- `xllm/core/framework/model/model_input_params.h`
- `xllm/core/framework/batch/batch_input_builder.cpp`
- `xllm/core/runtime/worker_impl.cpp`

需要补充：

- FC1 context 字段
- `to(device)` 的设备迁移逻辑
- batch 构造阶段计算 `original_num_tokens` 与 FC1 生效条件

如果希望减少 `ModelInputParams` 的字段膨胀，可以单独定义 `FlashComm1Context`，再由 Qwen3.5 模型入口根据 `ModelInputParams` 和 `ParallelArgs` 临时构造。

### 6.3 Qwen3.5 模型入口

涉及文件：

- `xllm/models/llm/qwen3_5.h`
- `xllm/models/llm/qwen3_next_hybrid_base.h`

当前 `Qwen3HybridModelImplBase::forward()` 的主流程：

```text
tokens -> embed_tokens -> layers -> final norm -> ModelOutput
```

FC1 适配后建议变为：

```text
tokens -> embed_tokens
       -> shard_sequence if FC1
       -> layers with fc1_context
       -> final norm on shard
       -> gather_and_unpad if logits path needs full selected tokens
       -> ModelOutput
```

注意事项：

- `residual` 必须与 `h` 保持同样的 sequence-sharded 形态
- final norm 可以在 shard 上执行，避免重新放大全序列 norm 开销
- logits 前是否 gather 取决于 selected token 所在 rank 的处理策略。首期建议模型输出恢复完整 token 顺序，保证上层 sampler 逻辑不变

### 6.4 Decoder Layer Forward

涉及文件：

- `xllm/core/layers/npu_torch/qwen3_next_hybrid_decoder_layer_base.h`
- `xllm/core/layers/npu_torch/qwen3_next_hybrid_decoder_layer_base.cpp`

`Qwen3HybridDecoderLayerImplBase::forward()` 的流程：

```text
现有:
  hidden = input_norm(hidden_states)          ← [S, H] → [S, H]
  attn_out = attention/gdn(hidden)            ← [S, H] → [S, H]
  hidden_states = residual + attn_out         ← [S, H] + [S, H]
  mlp_input = post_norm(hidden_states)        ← [S, H] → [S, H]
  mlp_out = mlp(mlp_input)                    ← [S, H] → [S, H]
  output = hidden_states + mlp_out            ← [S, H] + [S, H]

FC1 适配后:
  hidden = input_norm(hidden_states)          ← [S/TP, H] → [S/TP, H]
  attn_out = attention/gdn(hidden)            ← 内部 AllGather → 计算后 ReduceScatter → [S/TP, H]
  hidden_states = residual + attn_out         ← [S/TP, H] + [S/TP, H] (无需修改)
  mlp_input = post_norm(hidden_states)        ← [S/TP, H] → [S/TP, H]
  mlp_out = mlp(mlp_input)                    ← 内部 AllGather → 计算后 ReduceScatter → [S/TP, H]
  output = hidden_states + mlp_out            ← [S/TP, H] + [S/TP, H] (无需修改)
```

**关键结论**: Decoder layer 的 forward 流程无需改动。RMSNorm 和 residual add 天然支持 `[S/TP, H]` 输入。AllGather 和 ReduceScatter 被封装在注意力/GDN/MLP 模块内部，对 decoder layer 透明。

建议签名演进为：

```cpp
torch::Tensor forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const torch::Tensor& mrope_cos_sin,
    const FlashComm1Context* fc1_context);
```

### 6.5 Attention 层的 FC1 行为

涉及文件：

- `xllm/core/layers/npu_torch/qwen3_next_attention.h`
- `xllm/core/layers/npu_torch/qwen3_next_attention.cpp`

Attention 的 FC1 适配：

```text
当前:
  hidden → qkv_proj → q/k norm + mRoPE → attention → o_proj → all_reduce

FC1:
  h_shard → gather_sequence → qkv_proj → q/k norm + mRoPE → attention → o_proj(reduce_scatter) → h_shard
```

- `qkv_proj` 前执行 `gather_sequence` 恢复完整 hidden
- `o_proj` 使用 `REDUCE_SCATTER` 模式
- Attention 计算在 AllGather 后的完整 token 上执行，与无 FC1 时行为一致
- `mrope_cos_sin` 按完整 positions 构造
- `AttentionMetadata` 继续使用原始完整 batch 的 `q_seq_lens`、`q_cu_seq_lens`、`block_tables`

### 6.6 DenseMLP 适配

涉及文件：

- `xllm/core/layers/common/dense_mlp.h`
- `xllm/core/layers/common/dense_mlp.cpp`

```text
当前:
  gate_up_proj(ColumnParallel) → activation → down_proj(RowParallel) → reduce()

FC1:
  gather_sequence → gate_up_proj(ColumnParallel) → activation → down_proj(RowParallel, reduce_scatter)
```

- `gate_up_proj` 前执行 `gather_sequence`
- `down_proj` 使用 `REDUCE_SCATTER` 模式

### 6.7 Gated Delta Net 适配

涉及文件：

- `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.h`
- `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp`
- `xllm/core/layers/npu_torch/qwen3_5_gated_delta_net.h`
- `xllm/core/layers/npu_torch/qwen3_5_gated_delta_net.cpp`

GDN forward 流程中涉及 FC1 的关键改动点：

```text
当前:
  project_padded_inputs(input)         ← 4 个 ColumnParallelLinear 投影
  → merge projections
  → causal_conv1d
  → GDN compute (delta rule, A_log gating)
  → gated_rms_norm
  → o_proj(RowParallelLinear) → all_reduce

FC1:
  gather_sequence → project_padded_inputs(input)  ← 4 个 ColumnParallelLinear 在 gathered hidden 上执行
  → merge projections
  → causal_conv1d                               ← 在完整 [S, H] 上计算
  → GDN compute                                 ← 在完整 [S, H] 上计算
  → gated_rms_norm
  → o_proj(RowParallelLinear, reduce_scatter)   ← ReduceScatter 返回 [S/TP, H]
```

GDN 的特殊之处：

1. **4 个独立 ColumnParallel 投影**（`in_proj_qkv`、`in_proj_z`、`in_proj_b`、`in_proj_a`）：共享同一份 AllGather 后的 `[S, H]` 输入，在 GDN forward 入口处统一 gather 一次
2. **conv1d**（也是 ColumnParallelLinear）：同样在 gathered hidden 上执行
3. **中间计算**（merge、conv1d 计算、delta rule、gating）：在完整 sequence 上执行，无需修改
4. **o_proj**（RowParallelLinear）：使用 `REDUCE_SCATTER` 模式返回 `[S/TP, H]`

GDN 中间计算的 FC1 兼容性分析：

| 计算步骤 | 输入形状 | FC1 影响 |
|----------|----------|----------|
| `merge_qkvz_from_split_activations` | `[S, ...]` | 无影响 |
| `merge_ba_from_split_activations` | `[S, ...]` | 无影响 |
| causal conv1d 计算 | `[S, H]` | 无影响 |
| A_log gating (delta rule) | `[S, ...]` | 无影响 |
| chunk/recurrent gated delta rule | `[S, ...]` | 无影响 |
| GatedRMSNorm | `[S, ...]` | 无影响 |

这些计算步骤按 sequence 顺序处理 token（因果依赖），需要完整 sequence 输入。FC1 通过 AllGather 在投影前恢复完整 sequence，确保计算正确性不受影响。

### 6.8 Token Padding

涉及文件：scheduler 或 model runner 层

在模型 forward 前将 token 数 pad 到 TP size 的整数倍：

```text
pad_size = (tp_size - (num_tokens % tp_size)) % tp_size
padded_num_tokens = num_tokens + pad_size
```

padding 的原因：`reduce_scatter` 要求输入在 sequence 维上能被 TP size 整除。pad 操作需要在调度层完成，确保进入模型的所有 tensor 在 sequence 维上对齐。

FC1 关闭时或 token 数已经整除 TP size 时，无需 pad。

**注意事项**：

- `positions`、`AttentionMetadata`、`KVCache` 更新仍以真实 token 为准，不能把 padding token 写入有效 KV
- padding token 只用于通信 shape 对齐

### 6.9 首层和末层处理

**首层（Embedding → 第一个 Decoder Layer）**：

- Embedding 输出的 hidden states 是 `[S, H]`（完整 sequence）
- 第一个 Decoder Layer 需要在进入前将 hidden states 切分为 `[S/TP, H]`
- 实现：在进入 decoder layers 循环前调用 `shard_sequence()`

**末层（最后一个 Decoder Layer → LM Head）**：

- 最后一个 Decoder Layer 输出是 `[S/TP, H]`
- LM Head 需要完整 sequence，需要在进入前执行 `all_gather` 恢复 `[S, H]`
- 实现：在 decoder layers 循环后、LM Head 前调用 `gather_and_unpad_sequence()`

### 6.10 适配点总结

| 适配点 | 文件/模块 | 改动类型 | 改动说明 |
|--------|-----------|----------|----------|
| ParallelArgs | `parallel_args.h` | 新增字段 | `enable_flash_comm_v1_`、`enable_mmrs_fusion_` |
| Forward Context | `model_input_params.h` | 新增字段 | `FlashComm1Context` |
| RowParallelLinear | `linear.cpp` | 修改 forward | 增加 reduce_scatter 路径和 MMRS 融合路径 |
| ColumnParallelLinear 调用点 | attention / GDN / MLP | 新增逻辑 | projection 前 gather_sequence |
| Decoder Layer | `qwen3_next_hybrid_decoder_layer_base.cpp` | 签名演进 | 增加 fc1_context 参数 |
| Attention | `qwen3_next_attention.cpp` | 新增逻辑 | qkv_proj 前 gather，o_proj 用 reduce_scatter |
| DenseMLP | `dense_mlp.cpp` | 新增逻辑 | gate_up_proj 前 gather，down_proj 用 reduce_scatter |
| GDN | `qwen3_gated_delta_net_base.cpp` | 新增逻辑 | 入口 gather，o_proj 用 reduce_scatter |
| Model 入口 | `qwen3_next_hybrid_base.h` | 新增逻辑 | 首层 shard、末层 gather_and_unpad |
| 配置开关 | `global_flags.h`, `options.h` | 新增配置 | `enable_flashcomm1`、`flashcomm1_min_prefill_tokens` |
| Token Padding | scheduler / model runner | 新增逻辑 | pad 到 TP 对齐 |

## 7. 关键算子改动

### 7.1 Matmul + ReduceScatter 融合算子

对于支持的 NPU 平台，FC1 可以使用融合算子将 local matmul 和 reduce_scatter 合并为单次 kernel 调用。

**适用条件**：

- `enable_mmrs_fusion_ = true`（TP ≤ 8 时自动启用）
- 量化方式为非量化（BF16）或 W8A8 量化
- 其他量化方式退化为分步执行

**融合算子接口**：

```cpp
// xllm/core/kernels/ops_api.h
struct MatmulReduceScatterParams {
    torch::Tensor a;           // 输入 [S, H/TP]
    torch::Tensor b;           // 权重 [out, H/TP] 或量化权重
    std::optional<torch::Tensor> bias;
    ProcessGroup* process_group = nullptr;
    int64_t original_num_tokens = 0;
    // W8A8 量化参数
    std::optional<torch::Tensor> deq_scale;
    std::optional<torch::Tensor> output_dtype;
};

torch::Tensor matmul_reduce_scatter(MatmulReduceScatterParams& params);
```

**接入位置**：

- `xllm/core/kernels/ops_api.h` / `ops_api.cpp`
- `xllm/core/kernels/npu/npu_ops_api.h`
- `xllm/core/kernels/npu/matmul_reduce_scatter.cpp`（新增）

**执行策略**：

| 量化方式 | 融合路径 | 退化路径 |
|----------|----------|----------|
| BF16/FP16 未量化 | `npu_mm_reduce_scatter_base` | `matmul() → reduce_scatter()` |
| SmoothQuant W8A8 | 量化后 `npu_mm_reduce_scatter_base` | `quantize → matmul → reduce_scatter` |
| FP8 | 首期不融合 | `matmul → reduce_scatter` |
| 其他量化 | 不融合 | `matmul → reduce_scatter` |

**适用层**：

- `o_proj`（全注意力层和 GDN 层）
- `down_proj`（DenseMLP）

### 7.2 ReduceScatter 基础能力

当前 xllm 已有：

```cpp
torch::Tensor parallel_state::reduce_scatter(
    const torch::Tensor& input,
    ProcessGroup* process_group);
```

它支持 dim0 padding 后执行 reduce-scatter，并在 rank 尾部切掉无效 padding。FC1 首期可以复用这条路径作为保底实现。

需要补充的能力：

- 显式返回 padding 信息，便于后续 gather/unpad 和 graph mode 固化 shape
- 对 FC1 场景增加一致性检查：输入必须是二维 `[tokens, hidden]` 或最后一维为 hidden 的连续 tensor
- 统一 token shard 规则，避免 `shard_sequence()` 和 `reduce_scatter()` 对 padding 的处理不一致

### 7.3 Token Padding 算子

```cpp
// Pad hidden states 到 TP 对齐
torch::Tensor pad_for_fc1(const torch::Tensor& hidden_states, int pad_size) {
    if (pad_size == 0) return hidden_states;
    auto padding = torch::zeros({pad_size, hidden_states.size(-1)},
                                hidden_states.options());
    return torch::cat({hidden_states, padding}, 0);
}

// Unpad hidden states (最后一层输出后)
torch::Tensor unpad_after_fc1(const torch::Tensor& hidden_states, int pad_size) {
    if (pad_size == 0) return hidden_states;
    return hidden_states.slice(0, 0, hidden_states.size(0) - pad_size);
}
```

### 7.4 算子改动总结

| 算子 | 类型 | 适用层 | 说明 |
|------|------|--------|------|
| `matmul_reduce_scatter` | 新增（融合） | `o_proj`, `down_proj` | Matmul + ReduceScatter 融合 kernel |
| `reduce_scatter` | 使用现有 | `o_proj`, `down_proj` | 分步路径退化 |
| `gather` (AllGather) | 使用现有 | `qkv_proj`, `gate_up_proj`, GDN 投影 | 恢复完整 sequence |
| `shard_sequence` | 新增 | 模型入口 | 首层切分 hidden states |
| `gather_sequence` | 新增 | 各 layer 调用点 | Column-parallel 前 AllGather |
| `gather_and_unpad_sequence` | 新增 | 模型出口 | 末层恢复完整 sequence 并 unpad |
| `pad_for_fc1` | 新增 | scheduler/runner | Token padding |

## 8. 配置与启用

### 8.1 启用方式

```bash
# 环境变量
export XLLM_ENABLE_FLASHCOMM1=1

# 或通过模型配置 JSON
{
    "flash_comm_v1": true
}
```

### 8.2 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_flash_comm_v1` | `false` | FC1 全局开关 |
| `fc1_token_threshold` | `1000` | FC1 生效的最小 token 数阈值 |
| `enable_mmrs_fusion` | 自动 (TP≤8: true) | Matmul+ReduceScatter 融合开关 |

### 8.3 生效逻辑

```text
运行时 flash_comm_v1_enabled 的判断:

if (!parallel_args.enable_flash_comm_v1_) {
    return false;  // 全局开关关闭
}

if (device != NPU) {
    return false;  // 仅 NPU 支持
}

if (tp_world_size <= 1) {
    return false;  // 单卡无需 FC1
}

if (num_tokens <= fc1_token_threshold) {
    return false;  // token 数不足，FC1 收益 < 开销
}

return true;
```

小 batch 或 decode 阶段 token 数很少时，FC1 引入的额外 AllGather/ReduceScatter 通信开销可能超过节省的计算和访存收益，因此设置阈值。

### 8.4 兼容性

- FC1 关闭时，所有层走原有 `all_reduce` 路径，零额外开销
- FC1 可与现有量化方案（SmoothQuant W8A8、FP8）兼容
- FC1 可与现有 prefix caching、chunked prefill 等特性兼容
- TP=1 时 FC1 不生效（`reduce_scatter` 等价于 identity）
- 首期要求 `dp_size == 1` 且 `cp_size == 1`，混合并行后续逐步放开

## 9. 分阶段实施

### 阶段一：保底语义路径

- 增加 FC1 开关与 `FlashComm1Context`
- 在 Qwen3.5 模型入口完成 embedding 后 sequence shard
- 在 full attention、GDN、DenseMLP 的 column-parallel 前显式 all_gather
- row-parallel 后使用 `matmul + reduce_scatter`
- final norm 后 gather/unpad，保持上层 logits/sampler 契约不变
- 验证：FC1 off vs FC1 on 数值一致性

### 阶段二：融合算子

- 增加 NPU `matmul_reduce_scatter` kernel wrapper
- `RowParallelLinearImpl` 接入 `MATMUL_REDUCE_SCATTER` 模式
- 对 `o_proj`、GDN `o_proj`、`down_proj` 分别验证融合收益
- 性能 benchmark：prefill latency、tokens/s、HBM 峰值

### 阶段三：范围扩展

- 支持 chunked prefill
- 评估 DP/CP 组合下的 FC1 context
- 把 gather/reduce-scatter 模式从 Qwen3.5 下沉为通用 sequence-parallel helper
- 为 Qwen3.5-MoE 预留与 MC2 互补的 FC1 方案

## 10. 风险与注意事项

### 10.1 性能风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 小 batch 下 FC1 反而降低性能 | 吞吐下降 | token 数阈值判断，小 batch 自动关闭 |
| MMRS 融合算子不稳定 | 正确性问题 | 退化为分步路径，增加 fallback 检测 |
| GDN 4 个投影共享 AllGather 开销 | 延迟增加 | 入口处统一 gather 一次，后续投影复用 |

### 10.2 正确性风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 首层/末层 scatter/gather 遗漏 | 计算结果错误 | 模型 forward 入口/出口增加显式处理 |
| Token padding 不对齐 | ReduceScatter 报错 | 严格检查 pad 逻辑，增加断言 |
| padding token 被写入 KV cache | logits 错误或 cache 污染 | attention metadata 保持真实 token，padding 只用于通信 shape |
| final logits token 顺序不一致 | sampler 取错 token | 首期 final norm 后 gather_and_unpad，保持原契约 |
| GDN conv_cache / ssm_cache 状态更新与 shard 不一致 | recurrent state 错误 | 首期 projection 前 gather full hidden，保持现有 cache 语义 |

### 10.3 验证计划

**单元测试**：

- `shard_sequence()` / `gather_and_unpad_sequence()` 的 padding 和不等长 token 恢复
- `RowParallelLinearImpl` 在 `ALL_REDUCE` 与 `REDUCE_SCATTER + gather` 后结果等价
- `parallel_state::reduce_scatter()` 在 `S % TP != 0` 时输出 shape 和数据正确

**模型级正确性**：

- Qwen3.5-27B TP2/TP4，FC1 off vs FC1 on，对同一批 prompt 比较 prefill logits
- 浮点误差阈值按 BF16 路径设定
- 覆盖 `S <= 1000` 自动关闭、`S > 1000` 自动开启
- 覆盖 `S % TP != 0` 的 padding 场景

**性能验证**：

- prefill `S = 1024 / 2048 / 4096 / 8192`
- TP2 / TP4 / TP8
- FC1 off / FC1 on without mm+RS fusion / FC1 on with mm+RS fusion
- 观察指标：prefill latency、tokens/s、HBM 峰值占用、HCCL 通信耗时
