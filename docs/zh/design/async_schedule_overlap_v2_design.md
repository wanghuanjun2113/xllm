# 异步调度重叠 V2 设计文档

## 概述

本文档设计一种 Split-Phase 异步调度重叠机制，消除 decode 阶段两次 NPU 计算之间的设备空泡。

当前 `enable_schedule_overlap` 实现为**伪异步**模式——虽然内部使用了 `step_async()` 发送异步请求，但 `LLMEngine::step()` 紧接着 `collectAll(futures).get()` 阻塞等待，NPU 计算与 CPU 输出处理仍是严格串行的，无法实现真正的重叠。

本文档面向需要理解异步调度原理和参与实现的开发者，重点介绍：

- 当前实现的根因分析（含精确的代码定位和时序图）
- 两阶段语义模型（Materialize / Commit）
- Split-Phase 接口设计与实现
- 正确性约束与边界情况
- 实施路径与测试策略

### 设计目标

- 在 decode 场景下，把"上一步 Host 侧输出处理"与"下一步 NPU 计算"重叠执行
- 保留当前 fake-token / true-token 两阶段语义，不改动 Worker / RPC 协议
- 最小化对现有 request / sequence 状态机的侵入
- 为 `LLMEngine` 和 `VLMEngine` 提供统一的实现框架

### 非目标

- 不重新设计 request / sequence 的 token 状态机
- 不扩展到多步并行 in-flight（同时间只允许一个 pending step）
- 不覆盖 speculative decoding、`RecEngine` 多轮解码、prefill overlap
- 不修改 `enable_schedule_overlap` 对外的行为语义

## 1. 问题分析

### 1.1 当前实现的两阶段语义

当前 overlap 依赖两个阶段协作完成一步 decode 的输出处理：

**阶段 A（Materialize）**：step 计算完成后，把 batch 推进到"下一步可调度"的中间态。

- 在 `LLMEngine::step()` 内部完成
- 调用 `batch.process_sample_output(result, /*replace_fake_token=*/false)`
- 对每个 sequence 执行 `seq->append_token(sampled_token)`，追加一个占位 token（值可能非真实）
- Beam search 路径调用 `process_beam_search_output(..., false)`

**阶段 B（Commit）**：在下一步已提交后，把上一步的占位 token 回填为真实 token，并触发流式输出等 Host 逻辑。

- 在 `LLMEngine::update_last_step_result()` 内完成
- 调用 `batch.process_sample_output(result, /*replace_fake_token=*/true)`
- 对每个 sequence 执行 `seq->update_last_step_token(real_token, token_idx)`

### 1.2 Worker 侧的输出消费模型

当前 Worker 侧已经依赖两阶段消费同一轮 step 的结果：

1. **阶段 A 消费**：`LLMEngine::step()` 等待 worker future 完成后，从 `ForwardOutput` 直接读取 sample 结果
2. **阶段 B 消费**：`update_last_step_result()` 通过 `get_last_step_result_async()` 读取 worker 存储的 `last_step_output_`

Worker 内部通过 mutex + condition variable 协调这两个消费者的时序：

```
Worker 线程:  compute → store last_step_output_ → notify engine
Engine 线程:  step() 中 collectAll().get() 消费 ForwardOutput（阶段 A）
              update_last_step_result() 中 get_last_step_result_async() 消费 last_step_output_（阶段 B）
```

### 1.3 当前执行时序

`ContinuousScheduler::step_with_schedule_overlap()` 的实际执行顺序：

```
Step N 迭代:

  1. schedule_request()                    CPU: 组装 batch_N
  2. engine_->step(batch_N)                ← 同步阻塞!
     │  内部: 发送异步请求到 workers
     │  内部: collectAll(futures).get()     ← 等待 NPU 完成 batch_N
     │  内部: process_sample_output(..., false)  ← 阶段 A: batch_N materialize
  3. engine_->update_last_step_result(batch_{N-1})  ← 同步阻塞!
     │  内部: get_last_step_result_async()
     │  内部: collectAll(futures).get()     ← 等待获取 batch_{N-1} 输出
     │  内部: process_sample_output(..., true)      ← 阶段 B: batch_{N-1} commit
  4. process_batch_output(true)            CPU: 流式输出、状态更新
```

### 1.4 时序图

```
当前实现（伪异步）:

时间轴 ──────────────────────────────────────────────────────────►

NPU:  [compute_N] ────────────── idle ────────────── [compute_{N+1}] ──── idle ────
                               │                     │                    │
CPU:                          [get N-1 result][process N-1][schedule N+1] │
                               ├──────── 0.7s 空泡 ────────┤             │
                                                          [get N result][process N]
```

空泡由三部分组成：

1. `update_last_step_result()` 的 RPC 等待开销
2. `process_batch_output()` 的 Host 侧处理（token 回填、流式输出、停止检查）
3. `schedule_request()` 的调度开销（batch 组装、KV cache 分配）

**这些全部在 NPU 空闲期间串行执行。**

### 1.5 根因定位

| 文件 | 行号 | 问题 |
|------|------|------|
| `llm_engine.cpp` | L1032 | `collectAll(futures).get()` 在 `step()` 内部阻塞等待 NPU 完成 |
| `continuous_scheduler.cpp` | L1102 | `engine_->step(batch)` 调用后才能处理上一步输出 |
| `continuous_scheduler.cpp` | L1108 | `update_last_step_result()` 再次阻塞等待 |
| `engine.h` | L39 | `step()` 是同步接口，返回 `ForwardOutput` 而非拆分提交与等待 |

核心矛盾：**阶段 A 被绑定在 `step()` 内部，导致 scheduler 必须等 NPU 完成后才能推进 batch 状态。**

## 2. 设计方案

### 2.1 核心思路

不是把 `step()` 改成返回 future，而是把它拆成两个操作：

1. **`step_async(batch)`**：只负责提交计算到 worker，不等待，不修改 batch
2. **`wait_pending_step(batch)`**：等待 pending step 完成，并执行阶段 A

然后重排 scheduler 的执行顺序：

```
当前:  step(N) [阻塞等NPU + 阶段A] → update_result(N-1) [阶段B] → process_output(N-1)

优化:  wait_pending(N-1) [阶段A] → schedule(N) → step_async(N) [不等待] → update_result(N-1) [阶段B] → process_output(N-1)
                                                                              ↑ 与 NPU 计算 N 重叠 ↑
```

### 2.2 设计原则

**原则一：保留两阶段契约**

阶段 A 和阶段 B 分别对应不同的状态迁移，不能合并：

- 阶段 A：materialize scheduled step → batch 进入"下一步可调度"状态
- 阶段 B：commit true output → 占位 token 回填为真实 token，触发流式输出

**原则二：Batch 生命周期归 Scheduler 所有**

`Batch` 内部包含 `Sequence*`、`SequencesGroup*` 等指针语义对象。Engine 只持有 pending futures，不持有 `Batch` 副本。这避免了：

- Engine 内部复制 `Batch` 的开销和复杂度
- future 闭包持有 batch 引用的生命周期问题
- Scheduler 和 Engine 同时维护 batch 所有权

**原则三：系统始终只有一个 in-flight step**

当前 worker 线程池和 overlap 语义都建立在"单 worker 顺序执行 step"的前提下。本设计不改变这一点。

**原则四：不修改 Worker / RPC 协议**

当前 worker 侧已经满足 split-phase 方案需要的两个条件：

1. step 提交本身就是异步的（`step_async()` 在 threadpool 中执行）
2. 真实输出已经保存在 `last_step_output_` 中，可供阶段 B 读取

因此不需要修改 `WorkerImpl::step_async()`、`RemoteWorker`、RPC 协议。

### 2.3 为什么不采用 Future-of-Batch 方案

明确不采用以下接口：

```cpp
folly::SemiFuture<std::vector<Batch>> step_async(std::vector<Batch>& batch);
```

原因：

1. **Scheduler 不需要 future 值**，只需要"何时等待"的控制权
2. **Future 会导致 Engine 持有 `Batch` 副本**，增加生命周期复杂度
3. **Future 容易把阶段 A 和阶段 B 混成一次完成**，破坏两阶段语义
4. **当前只有一个 pending step**，promise/future 编排是不必要的复杂度

### 2.4 目标时序图

```
优化后（真异步 Split-Phase）:

时间轴 ──────────────────────────────────────────────────────────►

NPU:  [compute_N] ─────┐        [compute_{N+1}] ─────┐
                        │ overlap                       │ overlap
CPU:          [wait_N] [sched N+1] [update_N] [proc_N] [wait_{N+1}] ...
                        │  ↑ NPU 计算N+1 与 CPU 处理N 重叠 ↑  │
```

具体到每轮迭代内部：

```
迭代 N:
  wait_pending_step(batch_{N-1})     等待 NPU 完成 N-1，执行阶段 A  ← 短暂等待
  schedule_request(batch_N)          基于 N-1 阶段 A 状态调度 N     ← CPU
  step_async(batch_N)                提交 N 到 worker，不等待        ← 立即返回
  update_last_step_result(N-1)       对 N-1 执行阶段 B               ← 与 NPU 计算 N 重叠
  process_batch_output(N-1)          Host 逻辑                       ← 与 NPU 计算 N 重叠
```

## 3. 接口设计

### 3.1 Engine 接口扩展

```cpp
// engine.h
class Engine {
 public:
  virtual ~Engine() = default;

  // 原接口保持
  virtual ForwardOutput step(std::vector<Batch>& batch) = 0;
  virtual void update_last_step_result(std::vector<Batch>& batch) = 0;

  // Split-Phase API（仅 overlap 路径使用）
  virtual void step_async(std::vector<Batch>& batch) { NOT_IMPLEMENTED(); }
  virtual void wait_pending_step(std::vector<Batch>& batch) { NOT_IMPLEMENTED(); }
  virtual bool has_pending_step() const { return false; }
};
```

### 3.2 同步接口退化为包装器

`step()` 可以退化为 `step_async` + `wait_pending_step` 的组合，保证旧调用方兼容：

```cpp
ForwardOutput LLMEngine::step(std::vector<Batch>& batch) {
  step_async(batch);
  wait_pending_step(batch);
  return {};
}
```

这样 split-phase 路径和同步路径共享同一套阶段 A 逻辑，避免逻辑复制。

### 3.3 接口语义

| 方法 | 职责 | 是否阻塞 | 是否修改 batch |
|------|------|----------|---------------|
| `step_async(batch)` | 提交计算到 workers | 否 | 否 |
| `wait_pending_step(batch)` | 等待完成 + 阶段 A | 是 | 是 |
| `update_last_step_result(batch)` | 阶段 B | 是 | 是 |
| `step(batch)` | step_async + wait_pending_step | 是 | 是 |

## 4. Engine 侧实现

### 4.1 LLMEngine 新增成员

```cpp
class LLMEngine : public Engine {
 private:
  // Split-Phase 状态（仅持有 futures，不持有 batch）
  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>>
      pending_futures_;
  bool has_pending_ = false;
};
```

**不新增 `pending_batch_`**。Batch 生命周期由 Scheduler 持有。

### 4.2 `step_async(batch)` 实现

从当前 `step()` 中提取"提交计算"部分：

```cpp
void LLMEngine::step_async(std::vector<Batch>& batch) {
  CHECK(!has_pending_) << "Cannot step_async while pending step exists";

  auto raw_forward_inputs = prepare_inputs(batch);

  // CP 分区逻辑（与当前 step() 一致）
  // ...

  // 提交异步请求到所有 workers
  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(worker_clients_num_);
  for (auto worker_rank = 0; worker_rank < worker_clients_num_; ++worker_rank) {
    const int32_t dp_rank = worker_rank / dp_local_size_;
    const RawForwardInput* input_to_send = &raw_forward_inputs[dp_rank];
    // CP 路径选择 input_to_send ...
    futures.emplace_back(
        worker_clients_[worker_rank]->step_async(*input_to_send));
  }

  pending_futures_ = std::move(futures);
  has_pending_ = true;
}
```

**不做**以下事情：不等待 `collectAll`，不处理 sample/beam 输出，不修改 batch，不 fulfill promise。

### 4.3 `wait_pending_step(batch)` 实现

从当前 `step()` 中提取"等待完成 + 阶段 A"部分：

```cpp
void LLMEngine::wait_pending_step(std::vector<Batch>& batch) {
  CHECK(has_pending_) << "No pending step to wait for";

  // 等待所有 worker 完成
  auto results = folly::collectAll(pending_futures_).get();

  if (FLAGS_enable_eplb && !options_.enable_schedule_overlap()) {
    process_eplb_data(results);
  }

  // 阶段 A：按当前 step() 的聚合逻辑执行
  size_t dp_rank = 0;
  for (auto worker_rank = 0; worker_rank < worker_clients_num_;
       worker_rank += dp_local_size_) {
    auto result = results[worker_rank].value();
    if (result.has_value()) {
      if (result.value().outputs.empty() && layer_forward_interrupted_) {
        throw ForwardInterruptedException();
      }
      if (result.value().src_seq_idxes.size() == 0) {
        // replace_fake_token=false → 阶段 A
        batch[dp_rank].process_sample_output(result.value(), false);
      } else {
        batch[dp_rank].process_beam_search_output(result.value(), false);
      }
      batch[dp_rank].refresh_sequences_from_groups();
    } else {
      LOG(FATAL) << "Failed to execute model, result has no value";
    }
    ++dp_rank;
  }

  // 清空 pending 状态
  pending_futures_.clear();
  has_pending_ = false;
}
```

关键点：

- `wait_pending_step(batch)` 执行的是**阶段 A**（`replace_fake_token=false`）
- 它**不替代** `update_last_step_result(batch)`（那是**阶段 B**）
- 分布式聚合规则（DP/TP/CP）完全沿用当前 `step()` 的逻辑

### 4.4 `update_last_step_result()` 保持不变

阶段 B 维持当前语义不变：

```cpp
void LLMEngine::update_last_step_result(std::vector<Batch>& last_batch) {
  // 从 worker 侧 get_last_step_result_async()
  // ...
  for (auto i = 0; i < last_batch.size(); i++) {
    // replace_fake_token=true → 阶段 B
    last_batch[i].process_sample_output(raw_forward_outputs[i],
                                        options_.enable_schedule_overlap());
    last_batch[i].refresh_sequences_from_groups();
  }
}
```

### 4.5 VLMEngine

与 `LLMEngine` 一致的 split-phase 结构。第一阶段先保证 `LLMEngine` 路径正确，再复用同一模式。

### 4.6 其他 Engine

| Engine | 处理方式 |
|--------|---------|
| `SpeculativeEngine` | 禁用 overlap，保持原逻辑，不实现新接口 |
| `RecEngine` | 已禁用 overlap，不实现 |
| `DiTEngine` | 不涉及 decode，不实现 |

## 5. Scheduler 侧设计

### 5.1 不新增 batch 双缓冲

保留现有字段：

- `last_batch_`：上一轮已提交的 batch
- `last_running_requests_`：上一轮的 running requests
- `last_running_sequences_`：上一轮的 running sequences
- `is_first_step_`：首次 step 标志

仅新增一个标志：

```cpp
bool has_inflight_step_ = false;
```

`last_batch_` 在一轮迭代内依次经历两种语义：

1. 迭代开始时：等待其 pending step 完成，执行阶段 A
2. 提交新 step 后：对其执行阶段 B + Host 侧请求处理

因此不需要额外引入双缓冲。

### 5.2 新的执行流程

```cpp
void ContinuousScheduler::step_with_schedule_overlap_v2(
    const absl::Duration& timeout) {
  bool last_batch_ready = false;

  // 1. 等待上一轮 pending step 完成，执行阶段 A
  //    只有阶段 A 完成后，last_batch_ 才具备"可调度下一步"的状态
  if (has_inflight_step_) {
    engine_->wait_pending_step(last_batch_);
    has_inflight_step_ = false;
    last_batch_ready = !all_batch_empty(last_batch_);
  }

  // 2. 基于阶段 A 后的请求状态组装当前 batch
  std::vector<Batch> batch = schedule_request(timeout);
  bool cur_batch_empty = all_batch_empty(batch);

  // 3. 提交当前 step，不等待（NPU 开始计算）
  if (!cur_batch_empty) {
    engine_->step_async(batch);
    has_inflight_step_ = true;
    kv_cache_manager_->reset_transfer_infos();
  }

  // 4. 对上一轮 batch 执行阶段 B + Host 输出处理
  //    此时 NPU 正在计算当前 step，实现重叠
  if (!is_first_step_ && last_batch_ready) {
    engine_->update_last_step_result(last_batch_);
    process_batch_output(true);
  }

  // 5. 保存当前 batch，供下一轮使用
  last_batch_ = std::move(batch);
  last_running_sequences_ = running_sequences_;
  last_running_requests_ = running_requests_;
  is_first_step_ = false;
}
```

### 5.3 为什么阶段 A 必须在 `schedule_request()` 之前

当前 `schedule_request()` 依赖 batch 的 materialized 状态来决定：

- 哪些 sequence 可以继续 decode
- 哪些 request 已经完成可以释放
- KV cache 如何分配

如果把阶段 A 放在 `schedule_request()` 之后，scheduler 会基于过期的 batch 状态做调度决策，导致：

- 已结束的 sequence 被当作 running 继续调度
- KV cache 分配与实际需求不匹配
- Batch 组装与实际可用 sequence 不一致

## 6. 状态机

### 6.1 单 Batch 状态机

```
Submitted ──wait_pending_step(batch)──► Materialized ──update_last_step_result(batch)──► Committed
```

| 状态 | 含义 |
|------|------|
| Submitted | step 已提交到 worker，结果未等待，batch 不可调度 |
| Materialized | 阶段 A 完成，batch 已具备"下一轮可调度"状态 |
| Committed | 阶段 B 完成，真实 token 已回填，可安全做流式输出和终止检查 |

### 6.2 Scheduler 全局状态

Scheduler 全局维护两个事实：

1. `has_inflight_step_`：是否存在已提交但未等待的 step
2. `last_batch_` 的 materialized 状态：由本轮是否执行过 `wait_pending_step(last_batch_)` 得到，无需单独字段

### 6.3 迭代周期内的 Batch 流转

```
迭代 N 开始时:
  last_batch_ 状态: Submitted（上一轮 step_async 提交的）
  has_inflight_step_ = true

  wait_pending_step(last_batch_)  →  last_batch_ 变为 Materialized
  schedule_request()              →  基于 Materialized 状态组装 batch_N
  step_async(batch_N)             →  batch_N 进入 Submitted
  update_last_step_result(last_batch_)  →  last_batch_ 变为 Committed
  process_batch_output()          →  处理 Committed 的 last_batch_

  last_batch_ = std::move(batch_N)    →  新的 last_batch_ 处于 Submitted
```

## 7. 正确性约束

实现必须满足以下不变量：

1. **单 in-flight**：任意时刻最多只有一个 pending step
2. **阶段 A 先于调度**：`schedule_request()` 之前，必须先对上一轮 batch 执行 `wait_pending_step()`
3. **阶段 B 后于阶段 A**：`update_last_step_result(batch_k)` 只能在 `wait_pending_step(batch_k)` 之后执行
4. **Batch 生命周期**：Scheduler 必须持有 `batch_k` 直到阶段 B 完成
5. **Worker 串行**：Worker 侧 step 仍串行执行，不允许并发跑多个 step

如果这些约束被破坏，最容易出现的问题：

- 下一轮调度看不到上一轮的阶段 A 状态 → 调度决策错误
- placeholder token 被回填两次或完全没回填 → token 序列错误
- worker 输入修正和 batch 输出修正不对应同一轮 step → 生成结果错误

## 8. 边界情况

### 8.1 首次 step

```
is_first_step_ = true, has_inflight_step_ = false

→ 跳过 wait_pending_step()
→ 跳过 update_last_step_result()
→ 只执行 schedule_request() + step_async()
```

### 8.2 尾批次 drain

没有新请求，但上一轮还有 pending step：

```
1. wait_pending_step(last_batch_)     → 完成 pending step
2. schedule_request()                 → 得到空 batch
3. 不提交新 step（cur_batch_empty）
4. update_last_step_result(last_batch_)  → 阶段 B
5. process_batch_output(true)            → 处理最后一轮输出
```

注意：代码中 `last_batch_ready` 在 `wait_pending_step` 后设置为 true，即使 `cur_batch_empty` 为 true，仍然会执行阶段 B，保证最后一轮输出不丢失。

### 8.3 Request cancel / preempt

当前代码已存在 sequence 在 `schedule_request()` 期间被 preempt、KV cache 被释放、但 sequence 仍在 `last_batch_` 中等待阶段 B 的情况。

Split-phase 方案必须继续保留 `Sequence::update_last_step_token()` 中对 `kv_state_.num_kv_blocks() == 0` 的保护，不能把它当作"旧逻辑遗留"删除。

### 8.4 Batch 为空但仍有 pending step

当 `cur_batch_empty = true` 且 `has_inflight_step_ = true` 时：

```
1. wait_pending_step(last_batch_)  → 完成 pending step
2. schedule_request()              → 得到空 batch
3. 不提交新 step
4. update_last_step_result()       → 仍需处理上一轮的输出
```

新 step 不提交，`has_inflight_step_` 保持 false。下一轮如果仍然没有新请求，`last_batch_ready` 为 true 但 `last_batch_` 可能为空，需要跳过阶段 B。

### 8.5 Chunked prefill

不改变 `Batch::update_sequence_state()` 对 `pre_scheduled_step_prefill_queue()` 的维护。chunked prefill 场景下 overlap 语义保持与当前一致。

### 8.6 Beam search

阶段 A 必须保留 beam 路径分支：

- sample 路径：`process_sample_output(..., false)`
- beam 路径：`process_beam_search_output(..., false)`

不扩大 beam search 的 overlap 语义范围。

### 8.7 异常与中断

`wait_pending_step(batch)` 中的异常处理：

- `collectAll(...).get()` 抛错
- `ForwardInterruptedException`
- Worker 结果缺失

必须保证：

1. `has_pending_` 被清理（用局部 guard 或 RAII）
2. `has_inflight_step_` 不再误认为有 pending step
3. Scheduler 不对这轮 batch 再执行阶段 B

建议使用 scope guard 模式：

```cpp
void LLMEngine::wait_pending_step(std::vector<Batch>& batch) {
  CHECK(has_pending_);
  auto guard = folly::makeGuard([this] {
    pending_futures_.clear();
    has_pending_ = false;
  });
  // ... 可能抛出异常的逻辑 ...
}
```

## 9. 实施路径

### Phase 1：Engine Split-Phase 化

**涉及文件：**

- `xllm/core/distributed_runtime/engine.h`
- `xllm/core/distributed_runtime/llm_engine.h`
- `xllm/core/distributed_runtime/llm_engine.cpp`

**动作：**

1. 在 `engine.h` 新增 `step_async()` / `wait_pending_step()` / `has_pending_step()`
2. 从 `LLMEngine::step()` 中提取提交逻辑到 `step_async()`
3. 从 `LLMEngine::step()` 中提取等待+阶段A逻辑到 `wait_pending_step()`
4. `step()` 退化为 `step_async()` + `wait_pending_step()` 的包装器
5. `update_last_step_result()` 保持不变
6. 新增 `pending_futures_` + `has_pending_` 成员

### Phase 2：Scheduler 重排顺序

**涉及文件：**

- `xllm/core/scheduler/continuous_scheduler.h`
- `xllm/core/scheduler/continuous_scheduler.cpp`

**动作：**

1. 新增 `has_inflight_step_` 标志
2. 新增 `step_with_schedule_overlap_v2()` 方法
3. `step()` 中 overlap 路径切换到 `v2`
4. 保持 `process_batch_output(true)` 路径不变

### Phase 3：VLMEngine 复用模式

**涉及文件：**

- `xllm/core/distributed_runtime/vlm_engine.h`
- `xllm/core/distributed_runtime/vlm_engine.cpp`

**动作：** 按 `LLMEngine` 同样方式拆分。

### Phase 4：测试与回归

1. Engine 单测：`step_async` 不阻塞、`wait_pending_step` 正确清空 pending
2. Scheduler 单测：验证执行顺序（wait → schedule → submit → commit）
3. 边界情况测试：首次 step、尾批次 drain、request cancel/preempt
4. 分布式回归：多 DP / TP / CP 组合、EPLB
5. 性能对比：`overlap=false` vs `当前 overlap` vs `split-phase overlap`

### Phase 5：清理与收口

- 补充 Worker 侧注释，澄清"同一步结果分别服务于阶段 A 和阶段 B"
- 评估是否统一 `LLMEngine` / `VLMEngine` 的公共阶段 A 辅助逻辑
- 评估是否需要更精确的 tracing / metrics

## 10. 测试策略

### 10.1 Engine 单测

覆盖：

1. `step_async(batch)` 不阻塞，设置 pending 状态
2. `wait_pending_step(batch)` 清空 pending，执行阶段 A
3. `step(batch)` 与 `step_async + wait_pending_step` 行为一致
4. `update_last_step_result(batch)` 按阶段 B 语义回填真实 token
5. 异常路径：pending 状态正确清理

### 10.2 Scheduler 单测

**重点验证顺序**，不是只验证"最终能跑完"：

1. `wait_pending_step(last_batch_)` 发生在 `schedule_request()` 之前
2. `step_async(cur_batch)` 发生在 `update_last_step_result(last_batch_)` 之前
3. `process_batch_output(true)` 只在阶段 B 之后执行
4. 尾批次 drain 不漏最后一轮输出

### 10.3 回归测试矩阵

| 场景 | 预期 |
|------|------|
| 单 batch decode | 输出与 non-overlap 一致 |
| 多 batch 连续 decode | 吞吐提升，输出正确 |
| request cancel | 不 crash，不丢其他 request |
| request preempt / KV cache 回收 | token 状态机正确 |
| beam search | 输出与当前 overlap 一致 |
| `FLAGS_enable_eplb=true` | EPLB 数据聚合正确 |
| 多 DP / TP / CP 组合 | 分布式聚合索引正确 |

### 10.4 性能指标

建议指标：

- tokens/s
- inter-token latency
- `engine_latency_seconds` counter
- Host 侧阶段 B 耗时
- wait_pending_step 耗时
- NPU 利用率（空泡占比）

三组对比：

1. `enable_schedule_overlap=false`
2. 当前 overlap 实现
3. Split-phase overlap 实现

## 11. 风险评估

### 11.1 阶段 A / 阶段 B 边界错位

**风险**：token 状态机错乱，生成结果错误。

**缓解**：

- `wait_pending_step` 和 `update_last_step_result` 分别硬编码 `replace_fake_token=false/true`
- 不共享阶段 A/B 之间的中间状态

### 11.2 分布式聚合索引不一致

**风险**：CP / EPLB 场景下结果错误。

**缓解**：

- `wait_pending_step` 完全沿用当前 `step()` 的聚合逻辑
- `update_last_step_result` 完全沿用当前 `stride` 规则
- 不重新发明索引计算

### 11.3 Pending 状态泄漏

**风险**：异常后 scheduler 卡在"永远等待同一个 pending step"的状态。

**缓解**：

- 使用 scope guard 确保 `has_pending_` 和 `has_inflight_step_` 被清理
- 在 `wait_pending_step` 入口设置 guard，无论成功或异常都清理状态

### 11.4 过早抽象

**风险**：把 `LLMEngine` / `VLMEngine` 的差异抹平出错。

**缓解**：Phase 1-2 只改 `LLMEngine`，稳定后再复用到 `VLMEngine`。

## 12. 总结

本方案的核心不是把 `step()` 改成返回 future，而是把当前 overlap 里已经存在的两阶段状态机显式化：

1. **`step_async(batch)`**：提交 step，不等待，不修改 batch
2. **`wait_pending_step(batch)`**：等待完成，执行阶段 A（materialize）
3. **`update_last_step_result(batch)`**：执行阶段 B（commit）— 保持不变

Scheduler 把执行顺序调整为：

```
wait previous → schedule current → submit current → commit previous
```

这样既复用了当前 worker / request / sequence 语义（不改 Worker/RPC 协议），又把 Host 侧输出处理移动到下一轮 NPU 计算窗口中，实现真正有意义的 schedule overlap。
