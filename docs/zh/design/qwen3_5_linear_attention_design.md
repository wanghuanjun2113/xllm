# Qwen3.5 Linear Attention 设计说明

## 概述

本文详细列出 Qwen3.5 Hybrid Decoder 中 **Gated Delta Net 线性注意力子层**（`linear_attn` 分支）的数学公式、张量形状、权重参数与 xLLM 算子的对应关系。不展开外层 residual、post-attention layernorm 和 MLP。

### 相关代码文件

| 文件 | 说明 |
| --- | --- |
| `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.h` / `.cpp` | 基类，包含完整 forward 计算图和 Torch 参考实现 |
| `xllm/core/layers/npu_torch/qwen3_5_gated_delta_net.h` / `.cpp` | Qwen3.5 专有：四路分离投影 + merge 逻辑 |
| `xllm/core/layers/npu_torch/qwen3_next_gated_delta_net.h` / `.cpp` | Qwen3-Next：融合投影（与 Qwen3.5 的区别在于权重存储格式） |
| `xllm/core/layers/common/rms_norm_gated.h` / `.cpp` | 输出端 Gated RMSNorm |
| `xllm/core/kernels/ops_api.h` / `.cpp` | 顶层 kernel API 声明与后端分发 |
| `xllm/core/kernels/param.h` | 所有 kernel 参数结构体定义 |
| `xllm/core/kernels/npu/npu_recurrent_gated_delta_rule.cpp` | NPU recurrent_gated_delta_rule 实现（调用 AscendC ACL 算子） |
| `xllm/compiler/tilelang/targets/ascend/kernels/fused_gdn_gating.py` | fused_gdn_gating 的 TileLang kernel（生成 AscendC 源码） |
| `xllm/core/kernels/npu/tilelang/fused_gdn_gating_wrapper.cpp` | fused_gdn_gating TileLang kernel 的 C++ 包装器 |
| `xllm/core/kernels/npu/tilelang/dispatch_registry.h` | TileLang kernel 分发注册机制 |
| `xllm/models/llm/qwen3_5.h` | 模型注册与配置参数加载 |

---

## 配置参数与维度定义

从 `LOAD_QWEN3_5_NEXT_COMPAT_ARGS` 加载，以下使用符号表示：

| 符号 | 配置项 | 默认值 | 说明 |
| --- | --- | --- | --- |
| $d$ | `hidden_size` | — | 模型隐藏层维度 |
| $d_k$ | `linear_key_head_dim` | 128 | 每个 Key head 的维度 |
| $d_v$ | `linear_value_head_dim` | 128 | 每个 Value head 的维度 |
| $H_k$ | `linear_num_key_heads` | 16 | Key/Query head 数量 |
| $H_v$ | `linear_num_value_heads` | 32 | Value head 数量 |
| $K$ | `linear_conv_kernel_dim` | 4 | Causal 深度卷积核大小 |
| $tp$ | — | — | Tensor Parallel 分片数 |

**约束关系**：$H_v$ 必须能被 $H_k$ 整除，即 $H_v / H_k \in \mathbb{N}$（默认比例为 2:1）。

**TP 分片后的局部 head 数**：

$$H_k^{(tp)} = H_k / tp, \quad H_v^{(tp)} = H_v / tp$$

**总维度**：

$$K_{size} = H_k \cdot d_k \quad \text{(Q/K 总维度)}, \qquad V_{size} = H_v \cdot d_v \quad \text{(V 总维度)}$$

---

## 计算对象

### 1. 输入

- $x_t \in \mathbb{R}^{d}$：第 $t$ 个 token 进入 `linear_attn` 子层的 hidden state（已经过外层 `input_layernorm`）。
- Prefill 阶段：$X \in \mathbb{R}^{B \times T \times d}$，其中 $B$ 为 batch size，$T$ 为序列长度。
- Decode 阶段：$X \in \mathbb{R}^{B \times 1 \times d}$。

### 2. 运行时状态（KV Cache 中的线性注意力状态）

| 状态 | 符号 | 形状 | 说明 |
| --- | --- | --- | --- |
| Conv 缓存 | $C$ | `[num_linear_layers, max_batch, K-1, 2K_{size}+V_{size}]` | causal conv 的历史输入窗口 |
| SSM 缓存 | $S_{t-1}^{(h)}$ | `[num_linear_layers, max_batch, H_v^{(tp)}, d_v, d_k]` | 第 $h$ 个 head 的递推状态矩阵 |

代码中通过 `kv_cache.get_conv_cache()` 和 `kv_cache.get_ssm_cache()` 获取。

> **注意**：SSM 缓存的存储布局为 $[d_v, d_k]$（即 $S$ 的转置），与数学公式中的 $S \in \mathbb{R}^{d_k \times d_v}$ 互为转置。在读写时做了 transpose 处理。

### 3. 可学习权重

| 权重 | 符号 | 形状（TP 分片后） | 数据类型 | 对应代码 |
| --- | --- | --- | --- | --- |
| QKV 投影 | $W_{qkv}$ | $[(2K_{size} + V_{size}) / tp, \; d]$ | bf16 | `in_proj_qkv_`（ColumnParallelLinear） |
| Z 投影 | $W_z$ | $[V_{size} / tp, \; d]$ | bf16 | `in_proj_z_`（ColumnParallelLinear） |
| B 投影 | $W_b$ | $[H_v / tp, \; d]$ | bf16 | `in_proj_b_`（ColumnParallelLinear） |
| A 投影 | $W_a$ | $[H_v / tp, \; d]$ | bf16 | `in_proj_a_`（ColumnParallelLinear） |
| 因果卷积权重 | $W_{conv}$ | $[(2K_{size} + V_{size}) / tp, \; K]$ | bf16 | `conv1d_`（ColumnParallelLinear，weight 被 squeeze） |
| 时间衰减参数 | $A_{\log}$ | $[H_v^{(tp)}]$ | float32 | `A_log_` |
| 时间偏置 | $dt\_bias$ | $[H_v^{(tp)}]$ | float32 | `dt_bias_` |
| Gated RMSNorm 权重 | $\gamma$ | $[d_v]$ | bf16 | `norm_`（RmsNormGated） |
| 输出投影 | $W_o$ | $[V_{size}, \; d]$（RowParallelLinear） | bf16 | `o_proj_` |

> **注**：Qwen3.5 使用**四路分离投影**（`in_proj_qkv_`, `in_proj_z_`, `in_proj_b_`, `in_proj_a_`），而 Qwen3-Next 使用**融合投影**（`qkvz_proj_` + `ba_proj_`）。这是因为 Qwen3.5 的官方 checkpoint 将四个投影矩阵存储为独立权重，加载后由 xLLM 的 `merge_qkvz_from_split_activations` 和 `merge_ba_from_split_activations` 完成与 Qwen3-Next 兼容的数据布局转换。

### 4. 最终输出

- $y_t \in \mathbb{R}^{d}$：`linear_attn` 子层对第 $t$ 个 token 的输出。
- Prefill 阶段：$Y \in \mathbb{R}^{B \times T \times d}$。
- Decode 阶段：$Y \in \mathbb{R}^{B \times 1 \times d}$。

---

## 详细计算流程（分 8 步）

### 第 1 步：四路输入投影

对每个 token $x_t \in \mathbb{R}^{d}$，四路线性投影并行计算：

$$
\begin{aligned}
qkv_t^{(0)} &= W_{qkv} x_t \quad \in \mathbb{R}^{(2K_{size} + V_{size}) / tp} \\[4pt]
z_t^{(0)} &= W_z x_t \quad \in \mathbb{R}^{V_{size} / tp} \\[4pt]
b_t^{(0)} &= W_b x_t \quad \in \mathbb{R}^{H_v / tp} \\[4pt]
a_t^{(0)} &= W_a x_t \quad \in \mathbb{R}^{H_v / tp}
\end{aligned}
$$

按 batch 合并（经 `reshape_qkvz_with_pad` 做 padding 对齐到 `max_query_len`）：

$$
\begin{aligned}
QKV^{(0)} &\in \mathbb{R}^{B \times T_{max} \times (2K_{size} + V_{size}) / tp} \\
Z^{(0)} &\in \mathbb{R}^{B \times T_{max} \times V_{size} / tp} \\
B^{(0)} &\in \mathbb{R}^{B \times T_{max} \times H_v / tp} \\
A^{(0)} &\in \mathbb{R}^{B \times T_{max} \times H_v / tp}
\end{aligned}
$$

**对应代码**：`Qwen3_5GatedDeltaNetImpl::project_padded_inputs()`（`qwen3_5_gated_delta_net.cpp:126-139`）

---

### 第 2 步：Merge 与 Layout 重排

这一步将 Qwen3.5 分离的投影结果合并为 Qwen3-Next 兼容的数据布局，然后拆分为独立张量。

#### 2a. merge_qkvz_from_split_activations

将 `in_proj_qkv_` 的输出按 Q、K、V 拆分，并与 z 按 key head 对齐拼接。

1. 拆分：从 $QKV^{(0)}$ 的最后一维切分出 Q、K、V：

$$
\begin{aligned}
Q^{(0)} &: \text{取前 } K_{size}/tp \text{ 维} \quad \to \mathbb{R}^{B \times T \times H_k^{(tp)} \times d_k} \\
K^{(0)} &: \text{取中间 } K_{size}/tp \text{ 维} \quad \to \mathbb{R}^{B \times T \times H_k^{(tp)} \times d_k} \\
V^{(0)} &: \text{取后 } V_{size}/tp \text{ 维} \quad \to \mathbb{R}^{B \times T \times H_v^{(tp)} \times d_v}
\end{aligned}
$$

2. 对 V 和 Z 按 key head 做 repeat/interleave：由于 $H_v$ 是 $H_k$ 的整数倍（$r = H_v / H_k$，默认 $r = 2$），将 Value heads 按 key head 分组：

$$
\begin{aligned}
V^{(0)} &\to \mathbb{R}^{B \times T \times H_k^{(tp)} \times (r \cdot d_v)} \\
Z^{(0)} &\to \mathbb{R}^{B \times T \times H_k^{(tp)} \times (r \cdot d_v)}
\end{aligned}
$$

3. 拼接为 QKVZ 混合张量，最后一维按 $(q, k, v, z)$ 顺序排列：

$$
QKVZ = \operatorname{Concat}(Q^{(0)}, K^{(0)}, V^{(0)}, Z^{(0)}) \quad \in \mathbb{R}^{B \times T \times (2d_k + 2r d_v) \cdot H_k^{(tp)}}
$$

#### 2b. merge_ba_from_split_activations

将 B 和 A 按 key head 对齐后拼接：

$$
\begin{aligned}
B^{(0)} &\to \mathbb{R}^{B \times T \times H_k^{(tp)} \times r} \\
A^{(0)} &\to \mathbb{R}^{B \times T \times H_k^{(tp)} \times r} \\[4pt]
BA &= \operatorname{Concat}(B^{(0)}, A^{(0)}) \quad \in \mathbb{R}^{B \times T \times 2r \cdot H_k^{(tp)}}
\end{aligned}
$$

#### 2c. fused_qkvzba_split_reshape_cat

将 merge 后的 QKVZ 和 BA 张量 flatten 到 2D，分别拆出 `mixed_qkv`、`z`、`b`、`a`：

$$
\begin{aligned}
\text{mixed\_qkv} &\gets QKVZ[...,\ :(2K_{size}+V_{size})/tp] \quad \in \mathbb{R}^{(B \cdot T) \times (2K_{size}+V_{size})/tp} \\
\text{z\_flat} &\gets QKVZ[...,\ (2K_{size}+V_{size})/tp:] \quad \in \mathbb{R}^{(B \cdot T) \times V_{size}/tp} \\
z &\gets \operatorname{Reshape}(\text{z\_flat},\ [B \cdot T,\ H_v^{(tp)},\ d_v]) \\[4pt]
b &\gets BA[...,\ :H_v/tp] \quad \in \mathbb{R}^{(B \cdot T) \times H_v^{(tp)}} \\
a &\gets BA[...,\ H_v/tp:] \quad \in \mathbb{R}^{(B \cdot T) \times H_v^{(tp)}}
\end{aligned}
$$

**对应代码**：
- `merge_qkvz_from_split_activations()`：`qwen3_5_gated_delta_net.cpp:64-98`
- `merge_ba_from_split_activations()`：`qwen3_5_gated_delta_net.cpp:100-123`
- `fused_qkvzba_split_reshape_cat()`：`ops_api.cpp` 分发到 NPU 后端

---

### 第 3 步：Causal Depthwise Conv + SiLU

将 `mixed_qkv` reshape 回 3D 后，对每个通道独立做因果卷积。

设 $D_{qkv} = (2K_{size} + V_{size}) / tp$，$K = 4$ 为卷积核大小。对每个通道 $c = 0, \dots, D_{qkv}-1$：

$$
\tilde{u}_{t,c} = \mathrm{SiLU}\!\left( \sum_{i=0}^{K-1} W_{conv}[c, i] \cdot u_{t-i, c} \right)
$$

其中 $\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$。

**Prefill 路径**（全序列处理，`qwen3_gated_delta_net_base.cpp:363-384`）：

1. 将 `mixed_qkv` transpose 为 `[B, D_qkv, T]`
2. 提取末尾 $K-1$ 个 token 的历史窗口存入 `conv_cache`
3. 调用 `torch::conv1d(x, weight, padding=3, groups=D_qkv)`（padding=3 使输出比输入长，slice `[:, :, :seq_len]` 取回因果长度）
4. 经 `torch::silu` 激活

**Decode 路径**（单 token 增量，`qwen3_gated_delta_net_base.cpp:386-402`）：

调用 `causal_conv1d_update`，仅计算最新 token 的卷积结果，复用 `conv_cache` 中的历史状态；经 SiLU 激活。

**对应算子**：
- Prefill：`torch::conv1d` + `torch::silu`
- Decode：`causal_conv1d_update`（`ops_api.cpp` → `npu_causal_conv1d_update_v2`）

---

### 第 4 步：计算 g 和 beta（fused_gdn_gating）

对每个 token $t$ 和每个 value head $h$（$h = 0, \dots, H_v^{(tp)}-1$），计算两个门控量：

$$
\begin{aligned}
\Delta t_{t,h} &= \operatorname{softplus}(a_{t,h} + dt\_bias_h) \\[4pt]
&= \frac{1}{\beta}\log\!\left(1 + e^{\beta \cdot (a_{t,h} + dt\_bias_h)}\right)
\end{aligned}
$$

其中 $\beta = 1.0$（softplus 平滑参数），$\text{threshold} = 20.0$（数值稳定截断）：

$$
\operatorname{softplus}(x) = \begin{cases}
x, & \text{if } \beta x > 20.0 \\[4pt]
\displaystyle\frac{1}{\beta}\log(1 + e^{\beta x}), & \text{otherwise}
\end{cases}
$$

$$
g_{t,h} = -\exp(A_{\log, h}) \cdot \Delta t_{t,h} \quad \in \mathbb{R}
$$

$$
\beta_{t,h} = \sigma(b_{t,h}) = \frac{1}{1 + e^{-b_{t,h}}} \quad \in (0, 1)
$$

**直觉解释**：

- $dt\_bias_h$：每个 head 的时间偏置（训练参数），控制基础时间步长
- $a_{t,h}$：由输入动态预测的时间调制量
- $\Delta t_{t,h}$：综合时间步长，总是正数（softplus 保证），越大 → 衰减越快
- $g_{t,h}$：控制旧状态的衰减强度（$g_{t,h} \le 0$，因为 $-\exp(A_{\log}) \le 0$ 且 $\Delta t \ge 0$）
- $\beta_{t,h}$：控制当前 token 的写入强度（sigmoid 门控，$\in (0, 1)$）

**输出形状**：

$$
\begin{aligned}
g &\in \mathbb{R}^{B \times T \times H_v^{(tp)}} \quad \text{(float32)} \\
\beta &\in \mathbb{R}^{B \times T \times H_v^{(tp)}} \quad \text{(bf16)}
\end{aligned}
$$

**对应代码**：

- TileLang Kernel：`fused_gdn_gating.py:154-412`
  - 核心计算流程（行 261-308）：cast a → float32 → +dt_bias → softplus → mul(-exp(A_log)) → g; cast b → float32 → sigmoid → cast to bf16 → beta
- C++ Wrapper：`fused_gdn_gating_wrapper.cpp`
- Kernel 分发：`ops_api.cpp` → `npu::tilelang::fused_gdn_gating`

Prefill 和 decode 路径使用相同 kernel，区别仅在于输入的 batch 维度：
- Prefill：a、b view 为 `[-1, H_v/tp]`（$B \cdot T$ 行，`qwen3_gated_delta_net_base.cpp:408`）
- Decode：a、b view 为 `[-1, H_v/tp]`（$B$ 行，`qwen3_gated_delta_net_base.cpp:419`）

---

### 第 5 步：拆分 Q/K/V（process_mixed_qkv）

将卷积后的 `mixed_qkv` 拆分为独立的 Q、K、V 张量：

$$
\text{mixed\_qkv} \in \mathbb{R}^{B \times T \times (2K_{size} + V_{size}) / tp}
$$

按最后一维切分为 $[K_{size}/tp,\ K_{size}/tp,\ V_{size}/tp]$：

$$
\begin{aligned}
Q &\in \mathbb{R}^{B \times T \times H_k^{(tp)} \times d_k} \quad \text{（取前 } K_{size}/tp \text{ 维）} \\
K &\in \mathbb{R}^{B \times T \times H_k^{(tp)} \times d_k} \quad \text{（取中间 } K_{size}/tp \text{ 维）} \\
V &\in \mathbb{R}^{B \times T \times H_v^{(tp)} \times d_v} \quad \text{（取后 } V_{size}/tp \text{ 维）}
\end{aligned}
$$

**对应代码**：`Qwen3GatedDeltaNetBaseImpl::process_mixed_qkv()`（`qwen3_gated_delta_net_base.cpp:555-573`），内部使用 `torch::split` + `view`。

---

### 第 6 步：Q/K L2 归一化

对每个 key head $h$ 的 Q 和 K 向量做 L2 归一化，然后对 Q 侧乘以缩放因子 $1 / \sqrt{d_k}$：

$$
\begin{aligned}
\bar{q}_{t,h} &= \frac{q_{t,h}}{\sqrt{\sum_{j=1}^{d_k} q_{t,h,j}^2 + \varepsilon}} \cdot \frac{1}{\sqrt{d_k}} \\[8pt]
\bar{k}_{t,h} &= \frac{k_{t,h}}{\sqrt{\sum_{j=1}^{d_k} k_{t,h,j}^2 + \varepsilon}}
\end{aligned}
$$

其中 $\varepsilon = 10^{-6}$。

**Prefill 路径**：L2-Norm 在 `chunk_gated_delta_rule` 内核内部完成（参数 `use_qk_l2norm_in_kernel = true`，`qwen3_gated_delta_net_base.cpp:445`）。

**Decode 路径**：在调用 `recurrent_gated_delta_rule` 之前显式调用 `l2_norm`（`qwen3_gated_delta_net_base.cpp:452-453`），缩放因子 `scale = 1.0 / \sqrt{d_k}` 作为参数传入 kernel（行 457）。

**对应算子**：`l2_norm`（`ops_api.cpp` → `npu_l2norm_last_dim`）

---

### 第 7 步：Gated Delta Rule 递推（核心）

这是线性注意力的核心。当前 token 的信息被写入一个固定大小的矩阵状态 $S \in \mathbb{R}^{d_k \times d_v}$，而非随序列长度增长的 KV cache。

#### 7.1 Recurrent 形式（Decode 路径，逐 token 增量）

**符号定义**（对第 $h$ 个 head）：

| 符号 | 形状 | 含义 |
| --- | --- | --- |
| $\bar{q}_{t,h}$ | $[d_k]$ | L2 归一化 + scaled 的 query |
| $\bar{k}_{t,h}$ | $[d_k]$ | L2 归一化的 key |
| $v_{t,h}$ | $[d_v]$ | value |
| $g_{t,h}$ | 标量 | 衰减因子（来自 `fused_gdn_gating`） |
| $\beta_{t,h}$ | 标量 | 写入门控（来自 `fused_gdn_gating`） |
| $S_{t-1}^{(h)}$ | $[d_k, d_v]$ | 上一时刻的递推状态 |
| $S_t^{(h)}$ | $[d_k, d_v]$ | 更新后的递推状态 |
| $o_t^{(h)}$ | $[d_v]$ | 当前 token 的注意力输出 |

**Step 7.1.1 — 旧状态衰减**：

$$
\tilde{S}_{t-1}^{(h)} = e^{g_{t,h}} \cdot S_{t-1}^{(h)} \quad \in \mathbb{R}^{d_k \times d_v}
$$

衰减因子 $e^{g_{t,h}} \in (0, 1]$（因为 $g_{t,h} \le 0$），表示对过去记忆的保留程度。$g_{t,h}$ 越负，遗忘越快。

**Step 7.1.2 — 从旧状态读出已有记忆**：

$$
m_t^{(h)} = \bar{k}_{t,h}^{\top} \, \tilde{S}_{t-1}^{(h)} \quad \in \mathbb{R}^{d_v}
$$

这是以当前 key 为 query，从历史状态中检索到的、与当前 key 最相关的旧记忆向量。本质是 key 与状态矩阵中每个 $d_k$ 维度的加权求和。

**Step 7.1.3 — 计算写入增量（Delta Rule）**：

$$
\delta_t^{(h)} = \beta_{t,h} \cdot \left( v_{t,h} - m_t^{(h)} \right) \quad \in \mathbb{R}^{d_v}
$$

增量 = 经门控调制的 (当前 value − 已有记忆中 key 能解释的部分)。这实现了"只写入新信息"的效果——如果 value 与已有记忆一致，则几乎不写入。

**Step 7.1.4 — 状态更新（Rank-1 更新）**：

$$
S_t^{(h)} = \tilde{S}_{t-1}^{(h)} + \bar{k}_{t,h} \cdot \left( \delta_t^{(h)} \right)^{\top} \quad \in \mathbb{R}^{d_k \times d_v}
$$

用 key 作为写入方向（外积），增量作为写入内容。这是一个 rank-1 更新。

**Step 7.1.5 — 读取输出**：

$$
o_t^{(h)} = \bar{q}_{t,h}^{\top} \, S_t^{(h)} \quad \in \mathbb{R}^{d_v}
$$

用 query 从**更新后**的状态中读出输出。

**展开为完整公式**：

$$
\boxed{
S_t^{(h)} = e^{g_{t,h}} S_{t-1}^{(h)} + \bar{k}_{t,h} \left[ \beta_{t,h} \left( v_{t,h} - \bar{k}_{t,h}^{\top} \, e^{g_{t,h}} S_{t-1}^{(h)} \right) \right]^{\top}
}
$$

$$
\boxed{
o_t^{(h)} = \bar{q}_{t,h}^{\top} \, S_t^{(h)}
}
$$

**Decode 路径输入形状**：

| 参数 | 形状 | 说明 |
| --- | --- | --- |
| `q` | $[B, H_v^{(tp)}, d_k]$ | 已做 L2 norm + scale |
| `k` | $[B, H_v^{(tp)}, d_k]$ | 已做 L2 norm |
| `v` | $[B, H_v^{(tp)}, d_v]$ | — |
| `g` | $[B, H_v^{(tp)}]$ | 用于 $\exp(g)$ 计算衰减 |
| `beta` | $[B, H_v^{(tp)}]$ | sigmoid 门控 |
| `ssm_cache` | $[B, H_v^{(tp)}, d_v, d_k]$ | 即 $S_{t-1}$，NPU kernel 使用 $d_v \times d_k$ 布局 |
| `scale` | $1/\sqrt{d_k}$ | 缩放因子 |

输出：`core_attn_out`: $[B, 1, H_v^{(tp)}, d_v]$

**对应代码**：
- Torch 参考实现：`torch_recurrent_gated_delta_rule()`（`qwen3_gated_delta_net_base.cpp:31-96`）
- NPU 实现：`recurrent_gated_delta_rule`（`ops_api.cpp` → `npu_recurrent_gated_delta_rule` → AscendC `aclnnRecurrentGatedDeltaRule`）
- 调用位置：`qwen3_gated_delta_net_base.cpp:458-474`

#### 7.2 Chunked 形式（Prefill 路径，分块并行）

对于长度为 $T$ 的序列，分为 $\lceil T/C \rceil$ 个 chunk（$C = 64$）：

1. 序列 pad 到 $C$ 的整数倍
2. 每个 chunk 内通过 chunked parallel 形式计算局部注意力（使用因果衰减 mask）
3. Chunk 间通过初始状态 $S_{t-1}$ 传递

Torch 参考实现：`torch_chunk_gated_delta_rule()`（`qwen3_gated_delta_net_base.cpp:98-244`），核心步骤：

1. Pad 所有输入张量
2. 预计算 `v_beta = v * beta`，`k_beta = k * beta`
3. 在 chunk 内计算因果衰减矩阵 `decay_mask`：由 `g.cumsum(-1)` 的差值 + tril 得到
4. 计算 chunk 内 attention：`attn = -(k_beta @ k^T) * decay_mask`，再做逐行修正
5. Chunk 间做增量状态传递，类似并行扫描

**对应算子**：
- `chunk_gated_delta_rule`（`ops_api.cpp` → `npu_chunk_gated_delta_rule`）
- 调用位置：`qwen3_gated_delta_net_base.cpp:429-450`

**ChunkGatedDeltaRuleParams 参数说明**：

| 参数 | 形状 | 说明 |
| --- | --- | --- |
| `q` | $[B, T, H_k^{(tp)}, d_k]$ | — |
| `k` | $[B, T, H_k^{(tp)}, d_k]$ | — |
| `v` | $[B, T, H_v^{(tp)}, d_v]$ | — |
| `g` | $[B, T, H_v^{(tp)}]$ | float32 |
| `beta` | $[B, T, H_v^{(tp)}]$ | bf16 |
| `initial_state` | $[B, H_v^{(tp)}, d_k, d_v]$ | 当前实现填 0 |
| `scale` | auto | $1/\sqrt{d_k}$ |
| `chunk_size` | 64 | — |
| `use_qk_l2norm_in_kernel` | true | — |
| `head_first` | false | — |
| `output_final_state` | true | — |
| `cu_seqlens` | $[B+1]$ | int32 |

输出：
- `core_attn_out`: $[B, T, H_v^{(tp)}, d_v]$
- `last_recurrent_state`: $[B, H_v^{(tp)}, d_k, d_v]$

---

### 第 8 步：Gated RMSNorm + 输出投影

#### 8.1 Gated RMSNorm

对每个 token 的输出 $o_t \in \mathbb{R}^{H_v^{(tp)} \times d_v}$，以 z 作为门控信号：

$$
u_t = \mathrm{GRMSNorm}(o_t, z_t; \gamma)
$$

具体计算（`norm_before_gate = true`, `is_rms_norm = true`）：

1. 先对 $o_t$ 做 RMSNorm：

   $$
   \hat{o}_t = \frac{o_t}{\sqrt{\frac{1}{d_v}\sum_{j=1}^{d_v} o_{t,j}^2 + \varepsilon}} \cdot \gamma
   $$

2. 再用 sigmoid(z_t) 做门控调制：

   $$
   u_t = \hat{o}_t \odot \sigma(z_t)
   $$

其中 $\gamma \in \mathbb{R}^{d_v}$ 是可学习的 RMSNorm 缩放权重，$\varepsilon$ 为 `rms_norm_eps`。

**输入形状**：

| 参数 | 形状 | 说明 |
| --- | --- | --- |
| `core_attn_out` | $[B \cdot T, H_v^{(tp)}, d_v]$ | reshape 为 $[B \cdot T, H_v^{(tp)} \cdot d_v]$ |
| `z` | $[B \cdot T, H_v^{(tp)}, d_v]$ | reshape 为 $[B \cdot T, H_v^{(tp)} \cdot d_v]$ |

**输出形状**：`norm_out`: $[B, T, H_v^{(tp)}, d_v]$

**对应代码**：
- `RmsNormGated::forward()`（`rms_norm_gated.cpp:34-52`）→ `gated_layer_norm(params)`
- `ops_api.cpp` → `npu::layer_norm_fwd`
- 调用位置：`qwen3_gated_delta_net_base.cpp:477-483`

#### 8.2 输出投影

将归一化后的输出展平并投影回 `hidden_size` 维度：

$$
y_t = W_o \, u_t \in \mathbb{R}^{d}
$$

批量形式：

$$
Y = \mathrm{Unpad}(U) \cdot W_o^{\top}
$$

其中 `Unpad` 为 `reshape_qkvz_unpad`（去掉 prefill 的 padding），`o_proj_` 为 `RowParallelLinear`（input_is_parallelized=true, if_reduce_results=true），内部做 all-reduce 后输出 $[total\_tokens, d]$。

**对应代码**：
- `reshape_qkvz_unpad()`（`qwen3_gated_delta_net_base.cpp:493-510`）
- `o_proj_->forward()`（`qwen3_gated_delta_net_base.cpp:489`）

---

## 总公式链

将整个子层压缩为一步到位的公式链：

$$
\begin{aligned}
x_t &\xrightarrow{W_{qkv}, W_z, W_b, W_a}
(q_t^{(0)}, k_t^{(0)}, v_t^{(0)}, z_t, b_t, a_t) \\[4pt]
(q_t^{(0)}, k_t^{(0)}, v_t^{(0)}) &\xrightarrow{\text{causal depthwise conv} + \mathrm{SiLU}}
(q_t, k_t, v_t) \\[4pt]
(a_t, b_t, A_{\log}, dt\_bias) &\xrightarrow{\text{fused\_gdn\_gating}}
(g_t, \beta_t) \\[4pt]
(q_t, k_t) &\xrightarrow{\text{L2Norm} + \text{scale}}
(\bar{q}_t, \bar{k}_t) \\[4pt]
(\bar{q}_t, \bar{k}_t, v_t, g_t, \beta_t, S_{t-1}) &\xrightarrow{\text{Gated Delta Recurrence}}
(o_t, S_t) \\[4pt]
(o_t, z_t) &\xrightarrow{\text{GRMSNorm}}
u_t \xrightarrow{W_o}
y_t
\end{aligned}
$$

---

## 算子对应关系总表

| 步骤 | 数学公式 | xLLM 算子 / 方法 | 实现文件 | 后端 |
| --- | --- | --- | --- | --- |
| 1 | $W_{qkv}, W_z, W_b, W_a$ 四路投影 | `in_proj_qkv_`, `in_proj_z_`, `in_proj_b_`, `in_proj_a_`（ColumnParallelLinear） | `qwen3_5_gated_delta_net.cpp:129-136` | Torch |
| 2a | QKVZ merge | `merge_qkvz_from_split_activations` | `qwen3_5_gated_delta_net.cpp:64-98` | Torch |
| 2b | BA merge | `merge_ba_from_split_activations` | `qwen3_5_gated_delta_net.cpp:100-123` | Torch |
| 2c | Layout 重排与拆包 | `fused_qkvzba_split_reshape_cat` | `ops_api.cpp` | NPU (AscendC) |
| 3 | causal depthwise conv + SiLU | `torch::conv1d` (prefill) / `causal_conv1d_update` (decode) | `qwen3_gated_delta_net_base.cpp:363-402` | Torch / NPU |
| 4 | $g_{t,h}, \beta_{t,h}$ | `fused_gdn_gating` | `fused_gdn_gating.py`, `fused_gdn_gating_wrapper.cpp` | NPU (TileLang→AscendC) |
| 5 | Q/K/V 拆分 | `process_mixed_qkv` | `qwen3_gated_delta_net_base.cpp:555-573` | Torch |
| 6 | Q/K L2 归一化 | `l2_norm` (decode) / kernel 内部 (prefill) | `ops_api.cpp` → `npu_l2norm_last_dim` | NPU |
| 7a | Gated Delta Recurrence (prefill) | `chunk_gated_delta_rule` | `ops_api.cpp` → `npu_chunk_gated_delta_rule` | NPU (AscendC) |
| 7b | Gated Delta Recurrence (decode) | `recurrent_gated_delta_rule` | `ops_api.cpp` → `npu_recurrent_gated_delta_rule` → `aclnnRecurrentGatedDeltaRule` | NPU (AscendC ACL) |
| 8a | Gated RMSNorm | `RmsNormGated::forward` → `gated_layer_norm` | `rms_norm_gated.cpp:34-52` | NPU |
| 8b | 输出投影 $W_o$ | `o_proj_->forward`（RowParallelLinear） | `qwen3_gated_delta_net_base.cpp:489` | Torch |

---

## 补充说明

### A_log 和 dt_bias 的来源与作用

- $A_{\log}$ 和 $dt\_bias$ 是每个 head 的**训练参数**（非运行时计算），形状为 $[H_v / tp]$，数据类型 float32。
- 推理时从模型 checkpoint 加载，通过 `LOAD_SHARDED_WEIGHT` 宏按 TP 分片。
- 加载位置：`qwen3_gated_delta_net_base.cpp:312-313`。

**数学关系回顾**：

$$
g_{t,h} = -\exp(A_{\log, h}) \cdot \operatorname{softplus}(a_{t,h} + dt\_bias_h)
$$

- $A_{\log}$ 控制每个 head 的基础遗忘率：$\exp(A_{\log})$ 越大 → $|g|$ 越大 → 衰减越快 → 更"短视"
- $dt\_bias$ 给每个 head 的时间步长一个固定偏置，与输入动态量 $a_{t,h}$ 叠加

### Prefill 与 Decode 路径对比

| 方面 | Prefill | Decode |
| --- | --- | --- |
| 序列长度 | $T \ge 1$，通常较大 | $T = 1$ |
| Conv | `torch::conv1d` 全序列处理 | `causal_conv1d_update` 单步增量 |
| GDN 门控 | batch 处理（view 为 2D） | 同 prefill，batch 更小 |
| Q/K L2 Norm | kernel 内部（`use_qk_l2norm_in_kernel = true`） | 显式调用 `l2_norm` |
| 递推 | `chunk_gated_delta_rule`（分块并行） | `recurrent_gated_delta_rule`（逐 token 增量） |
| SSM Cache | 覆盖 position 对应的最终状态 | 读取上一时刻状态，就地更新为当前状态 |

两条路径计算的**数学对象完全相同**，只是工程实现根据序列长度做了不同优化。

### Qwen3.5 vs Qwen3-Next 对比

| 方面 | Qwen3-Next | Qwen3.5 |
| --- | --- | --- |
| QKVZ 投影 | 融合为 `qkvz_proj_` | 分离为 `in_proj_qkv_` + `in_proj_z_` |
| BA 投影 | 融合为 `ba_proj_` | 分离为 `in_proj_b_` + `in_proj_a_` |
| 加载后处理 | 无需 merge | `merge_qkvz_from_split_activations` + `merge_ba_from_split_activations` |
| 基类 forward | 相同（`Qwen3GatedDeltaNetBaseImpl::forward`） | 相同 |

分离投影的设计是因为 Qwen3.5 的官方 checkpoint 将四个投影矩阵存储为独立权重。二者在数学上等价。

### Hybrid Decoder 架构

Qwen3.5 采用混合架构：部分层使用 full attention（`Qwen3NextAttention`），部分层使用 linear attention（`Qwen3_5GatedDeltaNet`）。通过 `full_attention_interval` 参数或 `layer_types` 列表控制每层的注意力类型。

判断逻辑（`model_args.h:452-465`）：

```cpp
bool is_full_attention_layer(int layer_id) const {
    if (!layer_types.empty()) {
        return layer_types[layer_id] == "full_attention";
    }
    return full_attention_interval == 0
        || layer_id % full_attention_interval != 0;
}
```

即 `full_attention_interval = 4` 时，layer_id 能被 4 整除的层使用 linear attention，其余层使用 full attention。这个设计在长序列场景下大幅减少了 KV cache 的内存占用（linear attention 的 SSM 状态是固定大小 $O(d_k d_v)$，而 full attention 的 KV cache 随序列长度 $O(T)$ 增长）。

### KV Cache 中的线性注意力缓存

线性注意力引入了两类新的缓存结构（定义于 `kv_cache_utils.h`）：

| 缓存类型 | 形状 | 说明 |
| --- | --- | --- |
| `conv_cache` | $[num\_linear\_layers, max\_batch, K-1, 2K_{size}+V_{size}]$ | 保存 causal conv 最近 $K-1 = 3$ 个历史输入，用于 decode 时的增量卷积 |
| `ssm_cache` | $[num\_linear\_layers, max\_batch, H_v, d_v, d_k]$ | 保存每个 batch 每个 head 的 $S_t$ 递推状态矩阵，存储布局为 $[d_v, d_k]$（方便 NPU kernel 的 ACL 算子访问维度） |
