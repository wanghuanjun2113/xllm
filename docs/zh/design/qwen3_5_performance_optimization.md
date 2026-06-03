# Qwen3.5 性能优化机会分析

本文档记录 Qwen3.5/Qwen3.5-MoE 模型在 xLLM 中的性能优化机会，按优先级和模块分类。

---

## 优化总览

| 优先级 | 优化项数量 | 预期总体收益 |
|-------|-----------|-------------|
| 🔴 高 | 5项 | >20% 性能提升 |
| 🟡 中 | 8项 | 10-20% 性能提升 |
| 🟢 低 | 5项 | <10% 性能提升 |

---

## 🔴 高优先级优化

### 1. Fuse Gated Delta Net 投影层 (50% 内存带宽减少)

**问题描述**

Qwen3.5 的 Linear Attention (Gated Delta Net) 使用 4 个独立的 `ColumnParallelLinear` 投影：
- `in_proj_qkv_` - QKV 投影
- `in_proj_z_` - Gate 投影
- `in_proj_b_` - Beta 投影
- `in_proj_a_` - Alpha 投影

每个投影独立读取 `hidden_states`，导致 4x 内存读取。而 Qwen3-Next 使用 2 个融合投影（`qkvz_proj_`, `ba_proj_`），仅需 2x 内存读取。

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/core/layers/npu_torch/qwen3_5_gated_delta_net.cpp` | 30-61 | 4个独立投影定义 |
| `xllm/core/layers/npu_torch/qwen3_5_gated_delta_net.cpp` | 129-136 | 4次投影调用 |
| `xllm/core/layers/npu_torch/qwen3_next_gated_delta_net.cpp` | 49-65 | Qwen3-Next 融合实现（参考） |

**优化方案**

1. **方案A：Checkpoint 转换时融合**
   - 在模型加载时将 4 个独立权重合并为 2 个融合权重
   - 修改 `qwen3_5_gated_delta_net.cpp:30-61` 使用融合投影
   - 优点：无需修改推理代码，一次转换永久生效
   - 缺点：需要额外的 checkpoint 转换工具

2. **方案B：运行时融合 Kernel**
   - 创建 TileLang kernel 实现单次 GEMM + 输出通道拆分
   - 输入读取一次，输出直接拆分为 qkv/z/b/a 四路
   - 优点：无需修改 checkpoint
   - 缺点：需要开发新 kernel

**预期收益**

- 内存带宽减少 50%
- Linear Attention 层性能提升 30%
- 模型初始化时间减少 50-100ms（大模型）

**实现难度**

- 方案A：中等
- 方案B：高

---

### 2. TileLang `chunk_gated_delta_rule` Kernel (20-40% Linear Attention 加速)

**问题描述**

当前 `torch_chunk_gated_delta_rule()` 和 `torch_recurrent_gated_delta_rule()` 是纯 PyTorch 参考实现，包含大量低效操作：
- 循环式递归计算
- 多次 `.contiguous()` 和 `.clone()` 调用
- 未利用 NPU UB (Unified Buffer) 优化

虽然有 ACLNN kernel 派发路径（`ops_api.cpp:1047`），但 torch fallback 仍被编译和调用。

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp` | 31-244 | Torch fallback 实现 |
| `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp` | 377-384 | Prefill: conv1d + silu 分离 |
| `xllm/core/kernels/npu/tilelang/` | - | 缺少 GDN kernel |

**优化方案**

创建 TileLang kernel 实现：
1. `chunk_gated_delta_rule` - Prefill 路径
   - 融合 Conv1d + SiLU + Gated Delta Rule 计算
   - 利用 UB 进行状态缓存
   - 批量化 reduce 操作

2. `recurrent_gated_delta_rule` - Decode 路径
   - 融合 L2 Norm + Recurrent Update
   - 单 token 高效状态更新
   - 避免 per-token 循环

**实现步骤**

1. 在 `xllm/compiler/tilelang/targets/ascend/kernels/` 创建 `gated_delta_rule.py`
2. 定义 `DISPATCH_SCHEMA` 和 `SPECIALIZATIONS`（参考 `rope.py`）
3. 实现 `build_chunk_gated_delta_rule_kernel()` 和 `build_recurrent_gated_delta_rule_kernel()`
4. 在 `xllm/core/kernels/npu/tilelang/CMakeLists.txt` 注册 kernel
5. 创建 wrapper 文件 `gated_delta_rule_wrapper.cpp`

**预期收益**

- Prefill Linear Attention: 20-30% 加速
- Decode Linear Attention: 30-40% 加速
- 内存占用减少（消除中间 tensor）

**实现难度**

- 高

---

### 3. NPU Rejection Sampling Kernel

**问题描述**

Qwen3.5 MTP (Multi-Token Prediction) 投机型解码的 rejection sampling 使用多个分离的 tensor 操作：
- `index_select_2d`
- division
- comparison
- scatter

这些操作可融合为单个 kernel。MLU 已有实现（`kernels/mlu/rejection_sample.cpp`），NPU 缺失。

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/core/framework/sampling/rejection_sampler.cpp` | 254-316 | 多个分离操作 |
| `xllm/core/kernels/mlu/rejection_sample.cpp` | 20-47 | MLU 实现（参考） |
| `xllm/core/kernels/npu/` | - | 缺少 NPU 实现 |

**优化方案**

创建 `xllm/core/kernels/npu/rejection_sample.cpp`：
- 参考 MLU 实现逻辑
- 使用 Ascend-C 实现融合 rejection sampling
- 输入：draft_probs, target_probs, draft_tokens, threshold
- 输出：accepted_tokens, acceptance_mask, num_accepted

**预期收益**

- 减少 ~10 个 tensor 操作
- 投机型解码验证延迟减少 15-20%
- 减少中间 tensor 内存分配

**实现难度**

- 中等

---

### 4. Linear Attention Initial State Reuse (Prefix Cache 生效)

**问题描述**

当前 Linear Attention (GDN) 的初始状态在每次 forward 时被置为零，即使存在 prefix cache 匹配。这导致：
- 无法复用 prefix cache 的线性注意力状态
- 重复计算已缓存的序列前缀的 linear attention
- 性能浪费严重（尤其是长 reasoning chains）

代码中有 TODO 注释明确指出此问题。

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp` | 437-440 | TODO: "chunked-prefill/prefix-cache use initial_state" |
| `xllm/core/framework/kv_cache/kv_cache_shape.cpp` | 212-243 | Prefix cache 仅支持 DeepSeek |
| `xllm/core/framework/prefix_cache/prefix_cache.cpp` | - | 无 linear state hash |

**优化方案**

1. **实现 Linear State Hashing**
   - 为 linear attention 状态创建 hash 函数
   - 基于 (layer_id, input_hash) 查询 cached state

2. **State Serialization**
   - 设计 SSM + Conv state 的序列化格式
   - 存储到 prefix cache storage

3. **State Retrieval**
   - 在 `qwen3_gated_delta_net_base.cpp:forward()` 中
   - 检查 prefix cache，如有匹配则加载 initial_state
   - 而非 zero 初始化

**实现步骤**

1. 在 `kv_cache_utils.h` 添加 `linear_state_hash()` 函数
2. 扩展 `prefix_cache.h` 支持 linear state 类型
3. 修改 `qwen3_gated_delta_net_base.cpp:437-440` 使用 cached state
4. 添加 linear state serialization/deserialization

**预期收益**

- 长前缀序列复用时，linear attention 计算完全跳过
- First-token latency 减少（对于有 prefix 的请求）
- 内存占用优化（状态共享）

**实现难度**

- 中等

---

### 5. Per-Layer-Type Graph Bucketing

**问题描述**

Qwen3.5 使用混合架构（Hybrid），交替使用：
- Full Attention 层（O(n²) 复杂度）
- Linear Attention 层（O(n) 复杂度）

当前 graph executor 对所有层使用相同的 graph pool 策略：
- 基于 token count 的 bucket（16-multiple）
- 未区分 layer type

Linear Attention 层的特性：
- 状态更新是 per-sequence 固定大小
- 不需要基于 token count 的 bucket
- 可以使用更大的 batch size

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/core/runtime/cuda_graph_executor_impl.cpp` | 1264-1435 | 统一 graph bucket |
| `xllm/core/runtime/acl_graph_executor_impl.cpp` | 1112-1129 | ACL 同样问题 |
| `xllm/core/framework/model/model_args.h` | 452-465 | `is_full_attention_layer()` 判断 |

**优化方案**

1. **添加 Layer-Type Bucket**
   ```cpp
   enum class LayerTypeBucket {
     kFullAttention,   // 基于 token count bucket
     kLinearAttention, // 基于 num_seqs bucket（固定状态大小）
   };
   ```

2. **Separate Graph Pools**
   - Full Attention: 现有 bucket 策略（16-multiple tokens）
   - Linear Attention: 基于 num_seqs bucket（无 padding）

3. **Memory Pool Separation**
   - Linear state 使用独立 memory pool（`kPhysicalPoolIdLinearAttn=2`）
   - 生命周期与 token count 无关

**实现步骤**

1. 修改 `cuda_graph_executor_impl.cpp` 添加 `layer_type_bucket` 参数
2. 在 graph capture 时根据 `is_full_attention_layer()` 选择 bucket
3. 为 linear attention 创建单独的 graph pool
4. 修改 memory pool 分配逻辑

**预期收益**

- Graph memory pressure 减少 30-40%
- Linear attention batch size 可更大
- Decode 阶段 padding overhead 减少

**实现难度**

- 中等

---

## 🟡 中优先级优化

### 6. Remove Merge Operations Tensor Copies (10-15% Forward Overhead)

**问题描述**

`merge_qkvz_from_split_activations()` 和 `merge_ba_from_split_activations()` 使用：
- `torch::split` → 创建 view tensors
- `torch::cat` + `.contiguous()` → 创建新 tensor（内存拷贝）

这些 merge 操作产生不必要的 tensor 拷贝。

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/core/layers/npu_torch/qwen3_5_gated_delta_net.cpp` | 86-97 | merge_qkvz: cat + contiguous |
| `xllm/core/layers/npu_torch/qwen3_5_gated_delta_net.cpp` | 120-122 | merge_ba: cat + contiguous |

**优化方案**

使用 stride manipulation 替代 `cat`：
- 直接传递 split indices 到下游 kernel
- 避免物理内存拷贝
- 保持 stride view

**预期收益**

- 消除 2 次 tensor 拷贝
- Forward pass overhead 减少 10-15%

**实现难度**

- 中等

---

### 7. Fuse Conv1d + SiLU for Prefill

**问题描述**

Linear Attention prefill 路径使用分离的 `torch::conv1d` 和 `torch::silu`：
- Conv1d → 写入 intermediate `conv_output`
- SiLU on sliced result → 另一个 kernel launch

Decode 路径已使用融合的 `causal_conv1d_update`。

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp` | 377-384 | Prefill 分离 conv + silu |
| `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp` | 363 | Decode 已融合 |

**优化方案**

为 prefill 创建融合 kernel：
- Conv1d + SiLU 融合
- 直接写入 gating 输入 buffer

**预期收益**

- 减少 1 个 kernel launch
- 内存带宽优化（避免 intermediate write/read）

**实现难度**

- 中等

---

### 8. SSM State Quantization (INT8/FP8)

**问题描述**

SSM (State Space Model) cache 存储为 full precision（BF16/FP32）：
- `[batch_size, num_heads, head_k_dim, head_v_dim]`
- 占用大量内存（尤其是大 batch）

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/core/framework/kv_cache/kv_cache_utils.cpp` | 136-148 | SSM zeros full precision |
| `xllm/core/framework/kv_cache/kv_cache_shape.cpp` | 298-301 | SSM shape 定义 |

**优化方案**

实现量化 SSM state storage：
- INT8 或 FP8 存储（带 scaling factor）
- 状态更新时 dequantize
- 输出时保持 precision

**预期收益**

- Linear state 内存占用减少 50-75%
- 支持更大 batch size

**实现难度**

- 中等

---

### 9. MoE Expert Prefetch + Compute-Communication Overlap

**问题描述**

Qwen3.5-MoE 的专家权重未预取，且计算通信未重叠：
- `enableCVOverlap=false`（有 TODO 注释）
- 专家权重未预取
- All2All 与专家计算串行

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/core/layers/npu/npu_qwen3_moe_decoder_layer_impl.cpp` | 204 | enableCVOverlap=false |
| `xllm/core/layers/npu/loader/qwen3_moe_decoder_loader.cpp` | 269-329 | 专家权重后加载 |

**优化方案**

1. **Expert Prefetch**
   - 在路由计算时预取下一层专家权重
   - 利用 `enablePreFetchWeight` flag（参考 Qwen3）

2. **Compute-Communication Overlap**
   - Shared experts 计算 || Routed experts All2All dispatch
   - Expert computation || Combine All2All

**预期收益**

- MoE 层延迟减少 20-30%
- 通信隐藏计算

**实现难度**

- 中等

---

### 10. Model-Aware Speculative Token Count

**问题描述**

投机型解码的 speculative tokens 计算固定为 `num_speculative_tokens * 2`，未考虑模型复杂度差异。

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/core/scheduler/continuous_scheduler.cpp` | 149-151 | 固定 speculative tokens (`num_speculative_tokens * 2`) |

**优化方案**

基于 draft/target 模型大小和验证成本动态计算：
- Draft model 越大 → 更少 speculative tokens
- Target verification 成本高 → 更多 speculative tokens
- Hybrid 模型考虑 linear/full attention 分布

**预期收益**

- 投机型解码 acceptance rate 提升 10-15%
- 减少无效 draft 计算

**实现难度**

- 低

---

### 11. MTP Pre-FC Fusion

**问题描述**

MTP draft model 的 pre-forward 包含多个分离操作：
- `pre_fc_norm_embedding`
- `pre_fc_norm_hidden`
- `torch::cat`
- `fc_`

每次 draft forward 调用 4 个 kernel。

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/models/llm/qwen3_5_mtp.h` | 82-95 | Pre-FC 定义 |
| `xllm/models/llm/qwen3_5_mtp.h` | 124-134 | Pre-FC forward |

**优化方案**

创建融合 kernel：
- Norm + Cat + FC 融合为单个 Ascend-C kernel
- 输入：embedding, hidden_states
- 输出：fc_output 直接到 draft model layers

**预期收益**

- Draft forward kernel calls: 4 → 1
- Draft generation latency 减少 15-20%

**实现难度**

- 中等

---

### 12. Fuse Gate-Up GEMM + SiLU for MoE

**问题描述**

MoE 专家计算使用分离的 GroupGEMM 和 activation：
- GroupGEMM1 (gate + up projection)
- SiLU activation on sliced result
- GroupGEMM2 (down projection)

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/core/layers/npu_torch/fused_moe.cpp` | 332-351 | GroupGEMM1 分离 |
| `xllm/core/layers/npu_torch/fused_moe.cpp` | 363-382 | GroupGEMM2 分离 |

**优化方案**

融合 gate-up GEMM + SiLU：
- 单 kernel 完成 gate * silu(up)
- 避免 intermediate tensor

**预期收益**

- MoE expert forward 减少 1 kernel launch
- 内存带宽优化

**实现难度**

- 中等

---

### 13. Layer-Type-Aware Token Chunking

**问题描述**

Chunked prefill scheduler 对所有序列使用固定 chunk size，未区分 linear vs full attention。

Linear attention (GDN) 的特性：
- O(n) 复杂度，可处理任意长度
- 不需要 chunking
- 可使用更大 batch size

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/core/scheduler/chunked_prefill_scheduler.cpp` | 574-575 | 固定 chunk size |
| `xllm/core/scheduler/chunked_prefill_scheduler.cpp` | 718-750 | 统一 allocate_blocks |

**优化方案**

基于 layer_types 动态 chunk size：
- Full attention layers: 小 chunk（如 512 tokens）
- Linear attention layers: 全量处理（无 chunking）
- Hybrid 模型: 混合策略

**预期收益**

- Linear attention batch size 更大
- Prefill throughput 提升 15-20%

**实现难度**

- 中等

---

## 🟢 低优先级优化

### 14. Cache Attention Mask for Decode

**问题描述**

每层 forward 时 `build_attention_mask()` 重新计算 mask，即使 `max_seq_len_` 未变化。

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/models/llm/qwen3_next_hybrid_base.h` | 94-96 | build() 每次 forward |

**优化方案**

Decode phase cache attention mask：
- 仅当 `max_seq_len_` 变化时重建
- 避免重复 mask 计算

**预期收益**

- Decode speedup: 2-5%

**实现难度**

- 低

---

### 15. Fuse Residual + Norm

**问题描述**

Decoder layer 的 residual addition 包含 dtype conversion overhead：
- `x = x.to(torch::kFloat32)` → alloc
- `residual = residual.to(torch::kFloat32)` → alloc
- `x = x + residual` → alloc

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/core/layers/npu_torch/qwen3_next_hybrid_decoder_layer_base.cpp` | 127-131, 145-151 | Dtype conversion |

**优化方案**

- Fuse residual add with post_norm
- 如果 bf16 precision 可接受，消除 fp32 upcast
- 参考 Qwen2 `apply_norm` FP8 fusion

**预期收益**

- Memory bandwidth: 5-10% 减少

**实现难度**

- 低

---

### 16. Single-Pass MTP Weight Loading

**问题描述**

MTP model 加载使用多次 prefix search：
- `kEmbeddingPrefixes`
- `kMtpPrefixes`
- Separate loading for embed_tokens, lm_head, MTP weights

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/models/llm/qwen3_5_mtp.h` | 217-251 | Multiple prefix searches |

**优化方案**

Single-pass weight loading with prefix registry：
- 一次遍历 state_dict
- 注册所有 prefix 匹配

**预期收益**

- MTP initialization: 100-200ms 减少

**实现难度**

- 低

---

### 17. Dynamic UB Detection for TileLang Kernels

**问题描述**

TileLang kernels 使用 hardcoded `UB_BUDGET_BYTES = 64 * 1024`：
- 未适配不同 Ascend 芯片
- 910B: 128KB, 910C: 256KB

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/compiler/tilelang/targets/ascend/kernels/fused_gdn_gating.py` | 72 | Fixed 64KB |

**优化方案**

Extend `detect_vec_core_num` to return UB size per chip：
- 动态检测实际 UB capacity
- kernel 编译时使用检测值

**预期收益**

- 更大 batch/row capacity
- UB utilization 优化

**实现难度**

- 低

---

### 18. Batch mRoPE Apply in TileLang Kernel

**问题描述**

mRoPE apply 使用 `T.serial` 循环 per-head：
```python
for head_idx in T.serial(num_heads):
    T.copy(heads_fp32_ub[head_idx, 0], rope_orig_ub[head_idx, :])
```

**相关文件**

| 文件 | 行号 | 问题 |
|-----|-----|-----|
| `xllm/compiler/tilelang/targets/ascend/kernels/split_qkv_rmsnorm_mrope.py` | 289-303 | T.serial loop |

**优化方案**

Replace `T.serial` loop with `T.tile` batched swap：
- 所有 heads 并行处理
- 消除 sequential V-pipe stalls

**预期收益**

- mRoPE kernel throughput 提升 5-10%

**实现难度**

- 低

---

## 实现优先级排序

建议按以下顺序实现：

### Phase 1 (Kernel 层面，预期收益最大)

| 序号 | 优化项 | 预期收益 | 周期估算 |
|-----|-------|---------|---------|
| 1 | Fuse GDN projections | 30% | 2周 |
| 2 | TileLang chunk GDN kernel | 20-40% | 3周 |
| 3 | NPU rejection sampling | 15-20% | 1周 |

### Phase 2 (Cache/调度层面)

| 序号 | 优化项 | 预期收益 | 周期估算 |
|-----|-------|---------|---------|
| 4 | Linear state reuse | Prefix cache生效 | 2周 |
| 5 | Per-layer-type graph bucketing | 30-40% memory | 2周 |
| 6 | MoE expert prefetch + overlap | 20-30% | 2周 |
| 7 | Model-aware speculative token count | 10-15% acceptance | 1周 |
| 8 | Layer-type-aware chunking | 15-20% | 1周 |

### Phase 3 (内存/融合层面)

| 序号 | 优化项 | 预期收益 | 周期估算 |
|-----|-------|---------|---------|
| 9 | Remove merge tensor copies | 10-15% | 1周 |
| 10 | Fuse conv1d + silu | 1 kernel | 1周 |
| 11 | SSM state quantization | 50-75% memory | 2周 |
| 12 | MTP pre-fc fusion | 15-20% | 1周 |
| 13 | Fuse gate-up GEMM + SiLU | 1 kernel | 1周 |

### Phase 4 (微优化)

| 序号 | 优化项 | 预期收益 | 周期估算 |
|-----|-------|---------|---------|
| 14 | Cache attention mask | 2-5% | 0.5周 |
| 15 | Fuse residual + norm | 5-10% | 0.5周 |
| 16 | Single-pass MTP loading | 100-200ms | 0.5周 |
| 17 | Dynamic UB detection | batch capacity | 0.5周 |
| 18 | Batch mRoPE apply | 5-10% | 0.5周 |

---

## 附录：关键文件索引

### Linear Attention (Gated Delta Net)

| 文件 | 说明 |
|-----|-----|
| `xllm/models/llm/qwen3_5.h` | Qwen3.5 模型定义 |
| `xllm/core/layers/npu_torch/qwen3_5_gated_delta_net.cpp/.h` | GDN 实现 |
| `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp/.h` | GDN 基类 |
| `docs/zh/design/qwen3_5_linear_attention_design.md` | 设计文档 |

### MoE

| 文件 | 说明 |
|-----|-----|
| `xllm/models/llm/qwen3_moe.h` | MoE 模型定义 |
| `xllm/core/layers/qwen3_moe_decoder_layer.cpp/.h` | MoE decoder layer |
| `xllm/core/layers/npu/npu_qwen3_moe_decoder_layer_impl.cpp` | NPU MoE 实现 |
| `xllm/core/layers/npu_torch/fused_moe.cpp` | Fused MoE kernel |

### MTP (Speculative Decoding)

| 文件 | 说明 |
|-----|-----|
| `xllm/models/llm/qwen3_5_mtp.h` | MTP draft model |
| `xllm/core/framework/sampling/rejection_sampler.cpp` | Rejection sampling |
| `xllm/core/scheduler/fixed_steps_scheduler.cpp` | Speculative scheduler |

### KV Cache

| 文件 | 说明 |
|-----|-----|
| `xllm/core/framework/kv_cache/kv_cache_utils.cpp/.h` | KV cache 工具 |
| `xllm/core/framework/kv_cache/kv_cache_shape.cpp` | Cache shape 计算 |
| `xllm/core/framework/prefix_cache/prefix_cache.cpp/.h` | Prefix cache |

### Scheduler

| 文件 | 说明 |
|-----|-----|
| `xllm/core/scheduler/continuous_scheduler.cpp` | Continuous batching |
| `xllm/core/scheduler/chunked_prefill_scheduler.cpp` | Chunked prefill |
| `xllm/core/scheduler/fixed_steps_scheduler.cpp` | Speculative |

### Graph Execution

| 文件 | 说明 |
|-----|-----|
| `xllm/core/runtime/cuda_graph_executor_impl.cpp` | CUDA graph |
| `xllm/core/runtime/acl_graph_executor_impl.cpp` | ACL graph (NPU) |

### TileLang Kernels

| 文件 | 说明 |
|-----|-----|
| `xllm/compiler/tilelang/targets/ascend/kernels/split_qkv_rmsnorm_mrope.py` | MRoPE kernel |
| `xllm/compiler/tilelang/targets/ascend/kernels/fused_gdn_gating.py` | GDN gating |
| `xllm/core/kernels/npu/tilelang/*.cpp` | Kernel wrappers |

---

## 参考文档

- [Qwen3.5 Linear Attention 设计说明](qwen3_5_linear_attention_design.md)
- [TileLang Ascend Kernel 开发指南](../dev_guide/tilelang_ascend_kernel_dev.md)
- [Graph Mode 设计文档](graph_mode_design.md)
- [Chunked Scheduler 文档](../features/chunked_scheduler.md)
- [MTP 文档](../features/mtp.md)