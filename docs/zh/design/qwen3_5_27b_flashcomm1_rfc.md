# RFC: Qwen3.5-27B FlashComm1 适配方案

## 状态

- 状态：Draft
- 目标模型：Qwen3.5-27B Dense
- 目标后端：NPU torch layer backend
- 参考资料：`/Users/echo/code/vllm_design/vllm-ascend/docs/flashcomm1.md`

## 1. 背景

FlashComm1（下文简称 FC1）是一种面向 Ascend NPU 张量并行推理的 sequence-parallel 通信优化。它把 transformer 层与层之间的 hidden states 从“每个 TP rank 都保存完整 token 序列”改成“每个 TP rank 只保存 sequence 维的一段”。

```text
不开 FC1:
  每个 TP rank 保存 [S, H]

开启 FC1，TP = N:
  层间每个 TP rank 保存 [ceil(S / N), H] 或对应的有效 shard
```

对 Dense 模型来说，FC1 的核心变化是把 row-parallel 边界上的 `all_reduce` 改成 `reduce_scatter`，并在 column-parallel 层真正需要完整 sequence 输入之前再执行 `all_gather`。这样可以减少层间 RMSNorm、residual、量化和 activation 访存的重复工作，并为 `Matmul + ReduceScatter` 融合算子创造条件。

Qwen3.5-27B 在 xLLM 中复用 Qwen3-Next hybrid 框架，完整模型入口是 `Qwen3_5ForCausalLMImpl`，主干层由 `Qwen3_5DecoderLayerImpl` 构造。首期 FC1 适配聚焦 Qwen3.5-27B Dense prefill 场景：

- full attention layer：`Qwen3NextAttentionImpl`
- linear attention layer：`Qwen3GatedDeltaNetBaseImpl`
- Dense FFN：`DenseMLPImpl`
- TP 通信：`parallel_state::{gather, reduce, reduce_scatter}`
- row/column parallel 线性层：`RowParallelLinearImpl`、`ColumnParallelLinearImpl`、`QKVParallelLinearImpl`

## 2. 目标和非目标

### 2.1 目标

1. 为 Qwen3.5-27B Dense 增加 FC1 执行路径，首期覆盖 NPU torch layer backend。
2. 在大 token prefill 场景下，让层间 hidden states 保持 sequence-sharded 形态。
3. 将支持的 row-parallel 输出通信从 `all_reduce` 切换为 `reduce_scatter`。
4. 在 `qkv_proj`、`conv1d`、`gate_up_proj` 等 column-parallel 输入前按需 `all_gather`。
5. 为 `o_proj`、linear attention `out_proj`、FFN `down_proj` 接入 `Matmul + ReduceScatter` 融合算子预留统一接口。
6. 保持不开 FC1 时现有路径不变。

### 2.2 非目标

1. 首期不覆盖 Qwen3.5-MoE 的 MC2、fused MC2 或 expert dispatch/combine 适配。
2. 首期不要求 decode 阶段启用 FC1。decode token 数小，固定通信开销通常高于收益。
3. 首期不改 ATB backend 的 `NpuQwen3DecoderLayerImpl` 系列。
4. 首期不把 FC1 与 graph mode 强绑定，但接口设计需要避免阻碍后续图捕获。

## 3. 总体方案

### 3.1 执行语义

FC1 开启后，Qwen3.5-27B 单层的数据流变为：

```text
层间输入:
  h_shard: [S / TP, H]
  residual_shard: [S / TP, H]

Pre-attention norm:
  只处理本 rank 的 h_shard / residual_shard

Full attention layer:
  qkv_proj 前 all_gather: [S / TP, H] -> [S, H]
  attention 计算完整 S 上的本地 head shard
  o_proj 后 reduce_scatter: [S, H] -> [S / TP, H]

Linear attention layer:
  projection 前 all_gather: [S / TP, H] -> [S, H]
  gated delta net 按现有 padded batch 逻辑计算
  out_proj 后 reduce_scatter: [S, H] -> [S / TP, H]

Post-attention norm:
  只处理本 rank 的 sequence shard

Dense FFN:
  gate_up_proj 前 all_gather: [S / TP, H] -> [S, H]
  activation 保持现有 column-parallel 输出 shard
  down_proj 后 reduce_scatter: [S, H] -> [S / TP, H]

最终 norm / logits:
  如果只需要 selected token，先处理 selected token 的 rank 归属；
  若 lm_head 仍要求完整 token 顺序，则在模型输出前 gather_and_unpad。
```

Qwen3.5-27B TP4 的典型形态如下：

```text
S = 4096, TP = 4

层间 hidden:
  每 rank [1024, H]

qkv_proj / gate_up_proj 输入:
  all_gather 后临时恢复 [4096, H]

o_proj / down_proj 输出:
  reduce_scatter 后回到每 rank [1024, H]
```

### 3.2 首期启用条件

建议新增一个运行时开关，兼容 vLLM Ascend 的命名：

```bash
export XLLM_ENABLE_FLASHCOMM1=1
# 可选兼容：
export XLLM_ENABLE_FLASHCOMM=1
```

首期实际生效条件建议为：

```text
fc1_enabled =
  enable_flashcomm1
  && device == NPU
  && model_type in {"qwen3_5", "qwen3_5_text"}
  && tp_group != nullptr
  && tp_group.world_size() > 1
  && batch_forward_type is prefill-like
  && num_tokens > 1000
```

其中 `num_tokens > 1000` 延续参考实现里的 Dense 阈值，用于规避小 batch 或 decode 阶段额外集合通信带来的负收益。该阈值后续应沉淀为可配置参数。

### 3.3 上下文状态

建议在 `ModelInputParams` 或单独的 forward context 中增加 FC1 状态。考虑 xLLM 当前模型 forward 普遍透传 `ModelInputParams`，首期可以先在其中增加轻量字段：

```cpp
struct FlashComm1Context {
  bool enabled = false;
  int32_t tp_rank = 0;
  int32_t tp_world_size = 1;
  int32_t original_num_tokens = 0;
  int32_t padded_num_tokens = 0;
  int32_t local_num_tokens = 0;
  std::vector<int32_t> token_num_list;
};
```

关键约束：

- `original_num_tokens` 表示本次 forward 的真实 token 数。
- `padded_num_tokens` 表示 pad 到 TP size 倍数后的 token 数。
- `local_num_tokens` 表示本 rank reduce-scatter 后持有的 token 数。
- `token_num_list` 用于最后恢复真实 token 顺序，也可复用现有 `parallel_state::gather(input, pg, token_num_list)` 的不等长 gather 逻辑。

## 4. 关键接口设计

### 4.1 FC1 上下文构造

建议在模型入口或 runtime 层构造 FC1 context：

```cpp
FlashComm1Context build_flash_comm1_context(
    const ModelInputParams& input_params,
    const ParallelArgs& parallel_args,
    const ModelArgs& model_args);
```

职责：

- 判断是否启用 FC1。
- 根据 `tokens.size(0)`、TP size 和 rank 计算 padding 与本地 shard 范围。
- 在 embedding 后把完整 hidden 切成本 rank 的 sequence shard。
- 为最终输出恢复准备 `token_num_list`。

### 4.2 Sequence shard helper

建议增加一组工具函数，避免把 slice/pad/gather 逻辑散落到模型层：

```cpp
torch::Tensor shard_sequence(
    const torch::Tensor& input,
    const FlashComm1Context& fc1_context);

torch::Tensor gather_sequence(
    const torch::Tensor& input,
    const FlashComm1Context& fc1_context,
    ProcessGroup* tp_group);

torch::Tensor gather_and_unpad_sequence(
    const torch::Tensor& input,
    const FlashComm1Context& fc1_context,
    ProcessGroup* tp_group);
```

`shard_sequence` 的行为应与 `reduce_scatter` 的分片规则一致，保证 row-parallel 层输出和下一层输入的 token 顺序稳定。

### 4.3 RowParallelLinear 通信模式

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

- 现有 `forward(input)` 保持 `ALL_REDUCE` 或 `NONE` 语义不变。
- FC1 enabled 时，`o_proj`、linear attention `out_proj`、FFN `down_proj` 传入 `REDUCE_SCATTER` 或 `MATMUL_REDUCE_SCATTER`。
- 不支持融合算子时，自动退化为 `matmul + parallel_state::reduce_scatter()`。

### 4.4 ColumnParallelLinear 输入模式

当前 `ColumnParallelLinearImpl` 和 `QKVParallelLinearImpl` 默认认为输入已经是本次 layer 所需完整 token 范围。FC1 需要在 column-parallel projection 前按需 gather：

```cpp
torch::Tensor forward(
    torch::Tensor input,
    const FlashComm1Context* fc1_context,
    bool input_is_sequence_sharded);
```

首期可以不直接改所有 column-parallel 类，而是在调用点显式处理：

```cpp
torch::Tensor full_hidden =
    fc1_context.enabled ? gather_sequence(hidden, fc1_context, tp_group)
                        : hidden;
auto qkv = qkv_proj_->forward(full_hidden);
```

这样风险更小，也更容易逐层验证。

## 5. 当前框架主要适配点

### 5.1 配置与开关

涉及文件：

- `xllm/core/common/global_flags.h`
- `xllm/core/common/global_flags.cpp`
- `xllm/core/common/options.h`
- `xllm/core/common/options.cpp`

建议增加：

- `enable_flashcomm1`
- `flashcomm1_min_prefill_tokens`
- 可选兼容解析 `XLLM_ENABLE_FLASHCOMM`。

### 5.2 ModelInputParams / forward context

涉及文件：

- `xllm/core/framework/model/model_input_params.h`
- `xllm/core/framework/batch/batch_input_builder.cpp`
- `xllm/core/runtime/worker_impl.cpp`

需要补充：

- FC1 context 字段。
- `to(device)` 的设备迁移逻辑。
- batch 构造阶段计算 `original_num_tokens` 与 FC1 生效条件。

如果希望减少 `ModelInputParams` 的字段膨胀，可以单独定义 `FlashComm1Context`，再由 Qwen3.5 模型入口根据 `ModelInputParams` 和 `ParallelArgs` 临时构造。

### 5.3 Qwen3.5 模型入口

涉及文件：

- `xllm/models/llm/qwen3_5.h`
- `xllm/models/llm/qwen3_next_hybrid_base.h`

当前 `Qwen3HybridModelImplBase::forward()` 的主流程是：

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

- `residual` 必须与 `h` 保持同样的 sequence-sharded 形态。
- final norm 可以在 shard 上执行，避免重新放大全序列 norm 开销。
- logits 前是否 gather 取决于 selected token 所在 rank 的处理策略。首期建议模型输出恢复完整 token 顺序，保证上层 sampler 逻辑不变。

### 5.4 Decoder layer

涉及文件：

- `xllm/core/layers/npu_torch/qwen3_next_hybrid_decoder_layer_base.h`
- `xllm/core/layers/npu_torch/qwen3_next_hybrid_decoder_layer_base.cpp`

`Qwen3HybridDecoderLayerImplBase::forward()` 需要额外接收 FC1 context，并保证：

- `input_norm_` 在 sequence shard 上执行。
- full attention / linear attention 返回 sequence shard。
- `post_norm_` 在 sequence shard 上执行。
- DenseMLP 返回 sequence shard。

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

### 5.5 AttentionMetadata 与 position

FC1 的层间 hidden 是 sequence shard，但 attention 计算仍然需要与完整 token 序列语义对齐。首期建议：

- `positions` 在 attention projection 前随 hidden 一起 gather，或者直接保留完整 `positions` 给 attention。
- `AttentionMetadata` 继续使用原始完整 batch 的 `q_seq_lens`、`q_cu_seq_lens`、`block_tables`。
- full attention 的 `qkv_proj` 输入是 gathered hidden，因此现有 paged attention metadata 不需要按 shard 重写。
- linear attention 当前依赖 padded `[batch, max_len, hidden]` 逻辑，首期也在 projection 前 gather，保持现有 `reshape_qkvz_with_pad()`、KV cache 更新和 `linear_state_indices` 语义不变。

这样做不会把 attention 本身变成 sequence-parallel attention；FC1 首期优化的是层间 norm/residual/FFN 边界和 row-parallel 通信。

## 6. 关键算子改动

### 6.1 ReduceScatter 基础能力

当前 xLLM 已有：

```cpp
torch::Tensor parallel_state::reduce_scatter(
    const torch::Tensor& input,
    ProcessGroup* process_group);
```

它支持 dim0 padding 后执行 reduce-scatter，并在 rank 尾部切掉无效 padding。FC1 首期可以复用这条路径作为保底实现。

需要补充的能力：

- 显式返回 padding 信息，便于后续 gather/unpad 和 graph mode 固化 shape。
- 对 FC1 场景增加一致性检查：输入必须是二维 `[tokens, hidden]` 或最后一维为 hidden 的连续 tensor。
- 统一 token shard 规则，避免 `shard_sequence()` 和 `reduce_scatter()` 对 padding 的处理不一致。

### 6.2 Matmul + ReduceScatter 融合

参考实现中 Dense row-parallel 层会优先使用：

```python
torch_npu.npu_mm_reduce_scatter_base(...)
```

xLLM 建议在 NPU kernel API 中新增 wrapper：

```cpp
struct MatmulReduceScatterParams {
  torch::Tensor a;
  torch::Tensor b;
  std::optional<torch::Tensor> bias;
  ProcessGroup* process_group = nullptr;
  int64_t original_num_tokens = 0;
};

torch::Tensor matmul_reduce_scatter(MatmulReduceScatterParams& params);
```

接入位置：

- `xllm/core/kernels/param.h`
- `xllm/core/kernels/ops_api.h`
- `xllm/core/kernels/ops_api.cpp`
- `xllm/core/kernels/npu/npu_ops_api.h`
- `xllm/core/kernels/npu/matmul.cpp` 或新增 `matmul_reduce_scatter.cpp`

执行策略：

1. BF16/FP16 未量化权重：优先调用 NPU fused op。
2. SmoothQuant W8A8：先保持现有 `scaled_quantize`，再评估是否接入 quantized mm+RS 融合；首期可退化。
3. FP8：当前 xLLM FP8 matmul 主要是 CUDA 路径，NPU 首期不作为融合目标。
4. 不满足融合条件：执行 `matmul()` 后调用 `parallel_state::reduce_scatter()`。

### 6.3 RowParallelLinear 改动

`RowParallelLinearImpl::forward()` 当前末尾是：

```cpp
if (enable_result_reduction_ && world_size_ > 1) {
  output = xllm::parallel_state::reduce(output, process_group_);
}
```

FC1 后需要按模式切换：

```text
ALL_REDUCE:
  matmul -> reduce

REDUCE_SCATTER:
  matmul -> reduce_scatter

MATMUL_REDUCE_SCATTER:
  matmul_reduce_scatter
```

首期重点层：

- `Qwen3NextAttentionImpl::o_proj_`
- `Qwen3GatedDeltaNetBaseImpl::o_proj_`
- `DenseMLPImpl::down_proj_`

### 6.4 ColumnParallelLinear / QKVParallelLinear 改动

首期不强行把 FC1 逻辑塞进 linear 类内部，而是在 layer 调用点增加 gather：

- `Qwen3NextAttentionImpl::forward()`：`qkv_proj_` 前 gather hidden。
- `Qwen3GatedDeltaNetBaseImpl::forward()`：`project_padded_inputs()` 前 gather hidden。
- `DenseMLPImpl::forward()`：`gate_up_proj_` 前 gather hidden。

后续如果多个模型共享 FC1，可以再把 gather 逻辑沉淀到 `ColumnParallelLinearImpl` 的输入模式中。

### 6.5 Norm / residual

FC1 的收益之一来自 norm 和 residual 只处理本地 sequence shard。因此：

- `Qwen3NextRMSNorm` 不需要算子语义改动。
- `Qwen3HybridDecoderLayerImplBase` 需要保证 norm 输入已经是 shard。
- `residual` 的生命周期与 `h` 一样保持 shard，不允许某个子层返回 full hidden 后直接进入下一次 norm。

## 7. Qwen3.5-27B 分层适配方案

### 7.1 Full attention layer

当前流程：

```text
hidden -> qkv_proj -> q/k norm + mRoPE -> attention -> o_proj -> all_reduce
```

FC1 流程：

```text
h_shard
  -> gather_sequence
  -> qkv_proj
  -> q/k norm + mRoPE
  -> attention
  -> o_proj with reduce_scatter
  -> h_shard
```

注意：

- fused `split_qkv_rmsnorm_mrope` 仍然处理 gathered 后的完整 token。
- `mrope_cos_sin` 应按完整 positions 构造。
- `o_proj` 是 FC1 的主要通信边界。

### 7.2 Linear attention layer

当前流程：

```text
hidden -> project_padded_inputs -> conv/gating/chunk_gdn -> out_proj -> all_reduce
```

FC1 首期流程：

```text
h_shard
  -> gather_sequence
  -> project_padded_inputs
  -> conv/gating/chunk_gdn
  -> out_proj with reduce_scatter
  -> h_shard
```

这样做的原因：

- 现有 linear attention 的 prefill 路径按 batch 和 max query length 构造 padded tensor。
- `conv_cache`、`ssm_cache` 和 `linear_state_indices` 都是按完整 sequence 语义更新。
- 首期保持完整 token 输入可以最小化正确性风险。

后续优化可以探索只 gather linear attention 必要窗口，或者将 gated delta net 改造成 sequence-parallel 原生路径。

### 7.3 Dense FFN

当前流程：

```text
hidden -> gate_up_proj -> activation -> down_proj -> all_reduce
```

FC1 流程：

```text
h_shard
  -> gather_sequence
  -> gate_up_proj
  -> activation
  -> down_proj with reduce_scatter
  -> h_shard
```

对 Qwen3.5-27B TP4，`gate_up_proj` 输出为本 rank intermediate shard，`down_proj` 输出 `[S, H]` 的局部贡献，最终通过 reduce-scatter 得到下一层的 `[S / 4, H]`。

## 8. 正确性和边界条件

1. `tokens` 数不能被 TP size 整除时，FC1 context 必须记录 padding，并保证最后输出不包含 padding token。
2. `positions`、`AttentionMetadata`、`KVCache` 更新仍以真实 token 为准，不能把 padding token 写入有效 KV。
3. chunked prefill 需要单独验证。首期可以只允许普通 prefill，或在 chunked prefill 下保守关闭 FC1。
4. DP、CP、EP 混合并行需要逐步放开。首期建议要求 `dp_size == 1` 且 `cp_size == 1`。
5. final logits 如果仍由所有 rank 参与，需要在 lm_head 前恢复完整 token 顺序；如果后续优化为只在 owning rank 计算 selected token logits，需要同步改 sampler 输入契约。

## 9. 验证计划

### 9.1 单元测试

建议增加覆盖：

- `shard_sequence()` / `gather_and_unpad_sequence()` 的 padding 和不等长 token 恢复。
- `RowParallelLinearImpl` 在 `ALL_REDUCE` 与 `REDUCE_SCATTER + gather` 后结果等价。
- `parallel_state::reduce_scatter()` 在 `S % TP != 0` 时输出 shape 和数据正确。

### 9.2 模型级正确性

建议选择 Qwen3.5-27B TP2/TP4：

- FC1 off vs FC1 on，对同一批 prompt 比较 prefill logits。
- 浮点误差阈值按 BF16 路径设定。
- 覆盖 `S <= 1000` 自动关闭、`S > 1000` 自动开启。
- 覆盖 `S % TP != 0` 的 padding 场景。

### 9.3 性能验证

建议 benchmark：

- prefill `S = 1024 / 2048 / 4096 / 8192`
- TP2 / TP4 / TP8
- FC1 off
- FC1 on without mm+RS fusion
- FC1 on with mm+RS fusion

观察指标：

- prefill latency
- tokens/s
- HBM 峰值占用
- HCCL 通信耗时
- norm、activation、row-parallel projection 的耗时变化

## 10. 分阶段实施

### 阶段一：保底语义路径

- 增加 FC1 开关与 context。
- 在 Qwen3.5 模型入口完成 embedding 后 sequence shard。
- 在 full attention、linear attention、DenseMLP 的 column-parallel 前显式 all_gather。
- row-parallel 后使用 `matmul + reduce_scatter`。
- final norm 后 gather/unpad，保持上层 logits/sampler 契约不变。

### 阶段二：融合算子

- 增加 NPU `matmul_reduce_scatter` kernel wrapper。
- `RowParallelLinearImpl` 接入 `MATMUL_REDUCE_SCATTER` 模式。
- 对 `o_proj`、`out_proj`、`down_proj` 分别验证融合收益。

### 阶段三：范围扩展

- 支持 chunked prefill。
- 评估 DP/CP 组合下的 FC1 context。
- 把 gather/reduce-scatter 模式从 Qwen3.5 下沉为通用 sequence-parallel helper。
- 为 Qwen3.5-MoE 预留与 MC2 互补的 FC1 方案，但不混淆 FC1 与 expert dispatch/combine。

## 11. 风险

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| padding token 被写入 KV cache | logits 错误或 cache 污染 | attention metadata 保持真实 token，padding 只用于通信 shape |
| final logits token 顺序不一致 | sampler 取错 token | 首期 final norm 后 gather_and_unpad，保持原契约 |
| linear attention 状态更新与 shard 不一致 | recurrent state 错误 | 首期 projection 前 gather full hidden |
| 小 token 场景通信变慢 | decode 或小 prefill 退化 | 使用 token 阈值，decode 默认关闭 |
| 融合算子支持面不足 | 性能收益有限 | 保留 matmul + reduce_scatter fallback |

## 12. 结论

Qwen3.5-27B 的 FC1 适配可以先以“层间 sequence shard + column 前 gather + row 后 reduce-scatter”为主线落地。这个方案对现有 Qwen3.5 hybrid 层结构侵入较小，能复用当前 `parallel_state::reduce_scatter()` 作为保底路径，同时为 NPU `Matmul + ReduceScatter` 融合算子预留清晰入口。

首期最重要的是保证 shape、token 顺序、residual 形态和 KV cache 更新语义稳定；性能收益可以分两步释放：先减少 norm/residual/activation 的重复处理，再通过 row-parallel 融合算子降低 matmul 后通信开销。
