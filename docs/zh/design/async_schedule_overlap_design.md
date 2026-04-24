# 异步调度重叠设计文档

## 概述

本文档设计一种真正异步的调度重叠机制，用于消除 decode 阶段两次 NPU 计算之间的设备空泡（bubble）。当前 `enable_schedule_overlap` 实现为半异步模式，NPU 计算与 CPU 输出处理仍存在串行等待，无法充分利用硬件流水线。

本文档面向需要理解异步调度原理和关键设计的开发者，重点介绍以下内容：

- 当前实现的问题分析
- 异步调度重叠的核心设计
- 双缓冲流水线机制
- 关键数据结构修改
- 实现路径与风险控制

本文档的设计目标包括：

- 消除 decode 阶段 NPU 空泡，提升吞吐
- 保持调度逻辑清晰，避免复杂的锁竞争
- 兼容现有架构，最小化侵入性修改

本文档的非目标包括：

- 不覆盖 prefill 阶段的异步调度
- 不涉及分布式场景下的跨节点同步
- 不覆盖 speculative decoding 的特殊处理

## 1. 问题分析

### 1.1 当前流程时序

当前 `enable_schedule_overlap=true` 时，调度器在 `ContinuousScheduler::step_with_schedule_overlap()` 中的执行时序如下：

```
Step N:
  │
  ├─► schedule_request()          // CPU: 组装 batch_N
  │
  ├─► engine_->step(batch_N)      // 发送异步请求
  │     │
  │     └─► collectAll(futures).get()  ← 阻塞等待 NPU 完成!
  │
  ├─► update_last_step_result(batch_{N-1})  // CPU: 处理上一步输出
  │     │
  │     └─► collectAll(futures).get()  ← 阻塞等待输出获取!
  │
  └─► process_batch_output(true)  // CPU: 更新序列状态、流式输出
```

关键问题在于 `LLMEngine::step()` 和 `update_last_step_result()` 都使用了 `folly::collectAll(futures).get()`，导致：

1. **Step N 发送请求后立即等待 NPU 完成**，无法提前开始处理 Step N-1 的输出
2. **CPU 处理输出与 NPU 计算串行执行**，无法真正重叠
3. **两次 decode 之间存在空泡**，设备利用率不足

### 1.2 时序图对比

**当前实现（半异步）：**

```
时间轴 ─────────────────────────────────────────────────►

Step N-1:   [NPU 计算] ────────────┤
                                        │ 空泡
Step N:                          [等待]├─► [NPU 计算] ────────┤
                                       ↑                      │
                                       │                      │ 空泡
CPU:                          [处理N-1输出]                   ├─► [处理N输出]
```

**目标实现（真异步）：**

```
时间轴 ─────────────────────────────────────────────────►

Step N-1:   [NPU 计算] ────────────┤
                                        ↓ (overlap)
Step N:                        [NPU 计算] ────────────────┤
                                ↑                          ↓ (overlap)
CPU:                  [处理N-1输出]               [处理N输出]
```

### 1.3 根因定位

问题根源在于接口设计：

| 文件 | 函数 | 问题 |
|------|------|------|
| `engine.h:39` | `step(batch)` | 同步接口，返回 `ForwardOutput` 而非 future |
| `llm_engine.cpp:1032` | `collectAll(futures).get()` | 在 `step()` 内部阻塞等待 |
| `continuous_scheduler.cpp:1102` | `engine_->step(batch)` | 无法提前开始处理上一步输出 |

核心矛盾：**接口是同步的，但内部使用异步调用后又立即等待**。

## 2. 设计方案

### 2.1 核心思路

将同步接口改为**延迟等待模式**：

1. `step()` 发送异步请求，**返回 future 不等待**
2. Scheduler 决定**何时等待**，在等待前先处理上一步输出
3. 使用**双缓冲**保存两步的状态，避免数据竞争

### 2.2 接口修改

**Engine 接口扩展：**

```cpp
// engine.h 新增接口
class Engine {
 public:
  // 原同步接口（保持兼容）
  virtual ForwardOutput step(std::vector<Batch>& batch) = 0;
  
  // 新增：异步 step，返回 future
  virtual folly::SemiFuture<std::vector<Batch>> 
      step_async(std::vector<Batch>& batch) = 0;
  
  // 新增：等待 pending future
  virtual void wait_pending_step() = 0;
  
  // 原接口保持
  virtual void update_last_step_result(std::vector<Batch>& batch) = 0;
};
```

**LLMEngine 实现：**

```cpp
// llm_engine.h 新增成员
class LLMEngine : public Engine {
 private:
  // 保存当前 step 的 pending futures
  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> pending_futures_;
  std::vector<Batch> pending_batch_;
  bool has_pending_ = false;
};

// llm_engine.cpp
folly::SemiFuture<std::vector<Batch>> LLMEngine::step_async(
    std::vector<Batch>& batch) {
  // 发送异步请求，不等待
  pending_futures_ = send_step_requests(batch);
  pending_batch_ = batch;
  has_pending_ = true;
  
  // 返回一个 future，调用者可以决定何时等待
  folly::Promise<std::vector<Batch>> promise;
  auto future = promise.getSemiFuture();
  
  // 存储 promise，在 wait_pending_step 中 fulfill
  pending_promise_ = std::move(promise);
  return future;
}

void LLMEngine::wait_pending_step() {
  if (!has_pending_) return;
  
  // 等待当前 step 完成
  auto results = folly::collectAll(pending_futures_).get();
  
  // 处理输出，更新 batch
  process_step_results(results, pending_batch_);
  
  // fulfill promise
  if (pending_promise_) {
    pending_promise_.setValue(std::move(pending_batch_));
  }
  
  has_pending_ = false;
  pending_futures_.clear();
}
```

### 2.3 Scheduler 流程重构

**新流程：**

```cpp
void ContinuousScheduler::step_with_schedule_overlap(
    const absl::Duration& timeout) {
  // 1. 等待上一步 NPU 完成（如果有）
  if (has_pending_step_) {
    engine_->wait_pending_step();  // 等待 Step N-1 的 NPU 完成
  }
  
  // 2. 组装新 batch
  std::vector<Batch> batch = schedule_request(timeout);
  bool cur_batch_empty = all_batch_empty(batch);
  bool last_batch_empty = all_batch_empty(last_batch_);
  
  if (cur_batch_empty && last_batch_empty) {
    return;
  }
  
  // 3. 发送新 step（异步，不等待）
  if (!cur_batch_empty) {
    engine_->step_async(batch);  // Step N 发送，不等待
    has_pending_step_ = true;
  }
  
  // 4. 处理上一步输出（CPU，与 Step N NPU 重叠）
  if (!is_first_step_ && !last_batch_empty) {
    update_last_step_result(last_batch_);
    process_batch_output(true);
  }
  
  // 5. 保存状态
  last_batch_ = std::move(batch);
  is_first_step_ = false;
}
```

**时序变化：**

```
Step N-1: [NPU 计算] ─────────────┤
                                 ↓ overlap
Step N:            [等待N-1] [NPU 计算] ────────┤
                   ↑          ↓ overlap
CPU:        [处理N-1输出]              [处理N输出]
```

### 2.4 双缓冲设计

为了避免数据竞争，需要双缓冲保存两步的状态：

```cpp
// continuous_scheduler.h 新增成员
class ContinuousScheduler {
 private:
  // Batch 双缓冲
  std::vector<Batch> batch_buffer_[2];  // [0] = current, [1] = last
  int32_t current_buffer_idx_ = 0;
  
  // Running sequences 双缓冲
  std::vector<Sequence*> running_seq_buffer_[2];
  std::vector<std::shared_ptr<Request>> running_req_buffer_[2];
  
  // Pending step 状态
  bool has_pending_step_ = false;
  
  // 获取当前/上一缓冲索引
  int32_t current_idx() const { return current_buffer_idx_; }
  int32_t last_idx() const { return 1 - current_buffer_idx_; }
  
  // 交换缓冲
  void swap_buffer() { current_buffer_idx_ = 1 - current_buffer_idx_; }
};
```

### 2.5 完整流水线

将上述设计整合，得到完整的 decode 流水线：

```
Pipeline 状态机：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────┐    ┌──────────┐    ┌───────────┐              │
│  │ Wait N-1│───►│ Send N   │───►│ Process   │───► 循环     │
│  │ (NPU)   │    │ (Async)  │    │ N-1 Output│              │
│  └─────────┘    └──────────┘    └───────────┘              │
│       ↑                              │                      │
│       │                              ↓                      │
│       └──────────────────────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

时间轴展开：

T0: Wait Step 0 (首次跳过)
T1: Send Step 1 (async)
T2: Process Step 0 output (CPU)    ← 与 Step 1 NPU 重叠
T3: Wait Step 1
T4: Send Step 2 (async)
T5: Process Step 1 output (CPU)    ← 与 Step 2 NPU 重叠
...
```

## 3. 详细设计

### 3.1 Engine 接口定义

```cpp
// engine.h
class Engine {
 public:
  virtual ~Engine() = default;
  
  // 原接口保持
  virtual ForwardOutput step(std::vector<Batch>& batch) = 0;
  virtual void update_last_step_result(std::vector<Batch>& batch) = 0;
  
  // ===== 新增接口 =====
  
  // 异步执行 step，返回 future
  // 调用后立即返回，NPU 计算进行中
  // 返回的 future 可用于后续等待
  virtual folly::SemiFuture<std::vector<Batch>> 
      step_async(std::vector<Batch>& batch) {
    NOT_IMPLEMENTED();
    return {};
  }
  
  // 等待当前 pending 的 step 完成
  // 必须在 step_async 之后调用
  // 完成后处理输出并更新 batch
  virtual void wait_pending_step() { NOT_IMPLEMENTED(); }
  
  // 检查是否有 pending step
  virtual bool has_pending_step() const { return false; }
};
```

### 3.2 LLMEngine 实现

```cpp
// llm_engine.h
class LLMEngine : public Engine {
 private:
  // ===== 新增成员 =====
  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> 
      pending_futures_;
  std::vector<Batch> pending_batch_;
  folly::Promise<std::vector<Batch>> pending_promise_;
  std::mutex pending_mtx_;
  bool has_pending_ = false;
  
 public:
  // 实现 step_async
  folly::SemiFuture<std::vector<Batch>> step_async(
      std::vector<Batch>& batch) override;
  
  // 实现 wait_pending_step
  void wait_pending_step() override;
  
  // 实现 has_pending_step
  bool has_pending_step() const override { return has_pending_; }
};

// llm_engine.cpp
folly::SemiFuture<std::vector<Batch>> LLMEngine::step_async(
    std::vector<Batch>& batch) {
  CHECK(!has_pending_) << "Cannot step_async while pending step exists";
  
  auto raw_forward_inputs = prepare_inputs(batch);
  
  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(worker_clients_num_);
  
  // 发送异步请求（与原 step 相同）
  for (auto worker_rank = 0; worker_rank < worker_clients_num_; ++worker_rank) {
    futures.emplace_back(
        worker_clients_[worker_rank]->step_async(*input_to_send));
  }
  
  // 保存状态，不等待
  {
    std::lock_guard<std::mutex> lock(pending_mtx_);
    pending_futures_ = std::move(futures);
    pending_batch_ = batch;
    has_pending_ = true;
  }
  
  // 创建返回 future
  folly::Promise<std::vector<Batch>> promise;
  auto future = promise.getSemiFuture();
  pending_promise_ = std::move(promise);
  
  return future;
}

void LLMEngine::wait_pending_step() {
  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  std::vector<Batch> batch;
  
  {
    std::lock_guard<std::mutex> lock(pending_mtx_);
    if (!has_pending_) return;
    futures = std::move(pending_futures_);
    batch = std::move(pending_batch_);
    has_pending_ = false;
  }
  
  // 等待 NPU 完成
  auto results = folly::collectAll(futures).get();
  
  // 处理 EPLB（如果启用）
  if (FLAGS_enable_eplb) {
    process_eplb_data(results);
  }
  
  // 处理输出，更新 batch
  for (auto i = 0; i < batch.size(); i++) {
    auto result = results[i * dp_local_tp_size_].value();
    if (result.has_value()) {
      batch[i].process_sample_output(result.value(), 
                                     options_.enable_schedule_overlap());
      batch[i].refresh_sequences_from_groups();
    }
  }
  
  // fulfill promise（如果有）
  if (pending_promise_) {
    pending_promise_.setValue(std::move(batch));
  }
}
```

### 3.3 Scheduler 修改

```cpp
// continuous_scheduler.h
class ContinuousScheduler {
 private:
  // ===== 新增成员 =====
  bool has_pending_step_ = false;
  
 public:
  void step(const absl::Duration& timeout) override {
    if (!options_.enable_schedule_overlap()) {
      // 原逻辑保持不变
      step_sync(timeout);
    } else {
      step_with_schedule_overlap_v2(timeout);
    }
  }
  
 private:
  void step_with_schedule_overlap_v2(const absl::Duration& timeout);
};

// continuous_scheduler.cpp
void ContinuousScheduler::step_with_schedule_overlap_v2(
    const absl::Duration& timeout) {
  // Phase 1: 等待上一步 NPU 完成
  if (has_pending_step_) {
    engine_->wait_pending_step();
  }
  
  // Phase 2: 组装新 batch
  std::vector<Batch> batch = schedule_request(timeout);
  bool cur_batch_empty = all_batch_empty(batch);
  bool last_batch_empty = all_batch_empty(last_batch_);
  
  if (cur_batch_empty && last_batch_empty) {
    return;
  }
  
  // Phase 3: 发送新 step（异步）
  if (!cur_batch_empty) {
    engine_->step_async(batch);
    has_pending_step_ = true;
    kv_cache_manager_->reset_transfer_infos();
  }
  
  // Phase 4: 处理上一步输出（CPU，与 Phase 3 NPU 重叠）
  if (!is_first_step_ && !last_batch_empty) {
    engine_->update_last_step_result(last_batch_);
    process_batch_output(true);
  }
  
  // Phase 5: 更新状态
  last_batch_ = std::move(batch);
  last_running_sequences_ = running_sequences_;
  last_running_requests_ = running_requests_;
  is_first_step_ = false;
}
```

### 3.4 WorkerImpl 异步执行

WorkerImpl 的 `step_async()` 已经是异步实现，但需要确保：

```cpp
// worker_impl.cpp:658
// 当前实现已经正确：在 threadpool 中执行，不阻塞调用者
folly::SemiFuture<std::optional<ForwardOutput>> WorkerImpl::step_async(
    const ForwardInput& input) {
  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  
  threadpool_.schedule([this, input, promise]() mutable {
    // NPU 计算在 worker 线程中执行
    auto output = this->step(input);
    promise.setValue(output);
  });
  
  return future;  // 立即返回，不等待
}
```

需要修改的是 **Worker::step_async()** 的包装层：

```cpp
// worker.cpp
folly::SemiFuture<std::optional<ForwardOutput>> Worker::step_async(
    const ForwardInput& inputs) {
  return impl_->step_async(inputs);  // 已经是异步
}
```

这部分无需修改，已经是正确的异步模式。

### 3.5 其他 Engine 实现

需要同步修改的 Engine 子类：

| Engine | 状态 |
|--------|------|
| `VLMEngine` | 需实现 `step_async` / `wait_pending_step` |
| `RecEngine` | 需实现（如果支持 overlap） |
| `SpeculativeEngine` | 禁用 overlap，保持原逻辑 |
| `DiTEngine` | 不涉及 decode，无需修改 |

## 4. 边界情况处理

### 4.1 首次 Step

首次调用时没有上一步输出需要处理：

```cpp
// is_first_step_ 标志控制
if (!is_first_step_ && !last_batch_empty) {
  // 首次跳过
  engine_->update_last_step_result(last_batch_);
  process_batch_output(true);
}
is_first_step_ = false;
```

### 4.2 Batch 空情况

当 batch 为空时，不发送 step：

```cpp
if (!cur_batch_empty) {
  engine_->step_async(batch);
  has_pending_step_ = true;
} else {
  has_pending_step_ = false;  // 清除 pending 标志
}
```

### 4.3 Request 取消/完成

当 request 在 pending 期间被取消：

```cpp
// process_batch_output 中已有处理
if (options_.enable_schedule_overlap()) {
  if (request->cancelled()) {
    continue;  // 跳过已取消的 request
  }
  if (request->finished() && !request->last_token_handled()) {
    request->handle_last_token();
  }
}
```

保持现有逻辑，无需修改。

### 4.4 Chunked Prefill

Chunked prefill 阶段不启用 overlap：

```cpp
// 保持原有判断逻辑
// chunked prefill 时可能混有 prefill 和 decode
// 当前 overlap 仅优化 decode 部分
```

后续可扩展支持 chunked prefill overlap。

### 4.5 分布式场景

分布式场景下的注意事项：

```cpp
// wait_pending_step 需等待所有 DP worker
for (auto worker_rank = 0; worker_rank < worker_clients_num_;
     worker_rank += dp_local_size_) {
  // 只取 driver worker 的结果
  auto result = results[worker_rank].value();
  ...
}
```

保持现有逻辑，DP 同步由 Engine 负责。

## 5. 性能预期

### 5.1 空泡消除效果

假设：

- NPU decode 计算：10ms
- CPU 输出处理：2ms
- 调度开销：1ms

**当前实现：**

```
Decode NPU: 10ms
空泡: 3ms (等待 + CPU处理)
Decode NPU: 10ms
空泡: 3ms
...
吞吐 = 1 / 13ms = 77 token/s
```

**优化后：**

```
Decode NPU: 10ms
       ↓ overlap
CPU 处理: 2ms (重叠执行)
空泡: 1ms (仅调度)
Decode NPU: 10ms
...
吞吐 = 1 / 11ms = 91 token/s
```

**预期提升：~18% 吞吐提升**

### 5.2 时延影响

单 token 时延不变（仍为 NPU 计算时间），但整体吞吐提升。

### 5.3 显存影响

双缓冲设计不增加显存，仅在 CPU 端保存指针引用：

```cpp
// last_batch_ 和 batch 都是引用/指针
// 不复制 tensor 数据
```

## 6. 实现路径

### 6.1 Phase 1：接口定义（低风险）

1. 在 `engine.h` 中定义新接口 `step_async` / `wait_pending_step`
2. 默认实现为 `NOT_IMPLEMENTED()`
3. 保持原有 `step()` 接口不变

**涉及文件：**
- `xllm/core/distributed_runtime/engine.h`

### 6.2 Phase 2：LLMEngine 实现（中风险）

1. 在 `LLMEngine` 中实现 `step_async` / `wait_pending_step`
2. 添加 `pending_futures_` / `pending_batch_` 成员
3. 单元测试验证异步等待逻辑

**涉及文件：**
- `xllm/core/distributed_runtime/llm_engine.h`
- `xllm/core/distributed_runtime/llm_engine.cpp`

### 6.3 Phase 3：Scheduler 重构（高风险）

1. 重写 `step_with_schedule_overlap_v2()`
2. 添加 `has_pending_step_` 标志
3. 调整时序：wait → send → process
4. 集成测试验证流水线正确性

**涉及文件：**
- `xllm/core/scheduler/continuous_scheduler.h`
- `xllm/core/scheduler/continuous_scheduler.cpp`

### 6.4 Phase 4：其他 Engine 实现（低风险）

1. `VLMEngine` 实现新接口（如果启用 overlap）
2. 其他 Engine 保持 `NOT_IMPLEMENTED`

**涉及文件：**
- `xllm/core/distributed_runtime/vlm_engine.h`
- `xllm/core/distributed_runtime/vlm_engine.cpp`

### 6.5 Phase 5：性能验证

1. 添加性能测试用例
2. 对比 enable/disable overlap 的吞吐
3. 确保无空泡产生

**涉及文件：**
- 新增性能测试文件

## 7. 风险评估

### 7.1 数据竞争风险

**风险：** pending batch 和 last batch 可能同时被访问

**缓解：**
- 使用双缓冲，确保 batch 生命周期不重叠
- `pending_mtx_` 保护 pending 状态

### 7.2 请求取消风险

**风险：** request 在 pending 期间被取消，但 output 仍被处理

**缓解：**
- `process_batch_output` 中已有取消检查
- 保持现有逻辑不变

### 7.3 Worker 线程池饱和风险

**风险：** 两个 step 同时在 worker threadpool 中排队

**缓解：**
- `worker_impl.h:247` 注释说明：threadpool 保证 step 串行执行
- 一个 step 完成 before 下一个开始

### 7.4 兼容性风险

**风险：** 新接口可能影响现有调用者

**缓解：**
- 保持原 `step()` 接口不变
- 仅在 `enable_schedule_overlap=true` 时使用新流程
- 其他 Engine 默认 `NOT_IMPLEMENTED`

## 8. 测试策略

### 8.1 单元测试

**Engine 层测试：**
```cpp
TEST(LLMEngineTest, StepAsyncAndWait) {
  // 创建 mock batch
  std::vector<Batch> batch = create_test_batch();
  
  // 发送异步 step
  auto future = engine->step_async(batch);
  
  // 验证 has_pending_step
  EXPECT_TRUE(engine->has_pending_step());
  
  // 等待完成
  engine->wait_pending_step();
  
  // 验证 batch 已更新
  EXPECT_FALSE(engine->has_pending_step());
}
```

**Scheduler 层测试：**
```cpp
TEST(ContinuousSchedulerTest, AsyncOverlapPipeline) {
  // 添加多个 request
  add_requests(10);
  
  // 执行多次 step
  for (int i = 0; i < 10; i++) {
    scheduler->step(timeout);
  }
  
  // 验证所有 request 完成
  EXPECT_EQ(num_finished_requests(), 10);
}
```

### 8.2 集成测试

**端到端测试：**
```cpp
TEST(IntegrationTest, AsyncOverlapThroughput) {
  // 配置 enable_schedule_overlap=true
  auto engine = create_engine_with_overlap();
  
  // 发送并发请求
  auto requests = send_concurrent_requests(100);
  
  // 测量吞吐
  auto throughput = measure_throughput();
  
  // 验证吞吐提升
  EXPECT_GT(throughput, baseline_throughput * 1.1);
}
```

### 8.3 性能对比测试

| 配置 | 预期吞吐 |
|------|----------|
| `enable_schedule_overlap=false` | baseline |
| `enable_schedule_overlap=true` (当前) | +5% |
| `enable_schedule_overlap=true` (优化) | +18% |

## 9. 后续扩展

### 9.1 Prefill Overlap

当前仅支持 decode overlap，可扩展支持：

- Chunked prefill 与 decode overlap
- Prefill 阶段的 multi-chunk overlap

### 9.2 多级流水线

扩展为三级流水线：

```
[Step N NPU] → [Step N+1 NPU] → [Step N+2 NPU]
              ↓                ↓
        [处理N-1]        [处理N]
```

### 9.3 Speculative Decoding

Speculative decoding 需特殊处理：

- Draft model 和 target model 的异步协调
- Verify 阶段的重叠设计

## 10. 总结

本设计通过以下关键改动消除 NPU 空泡：

1. **接口扩展**：`step_async()` + `wait_pending_step()`
2. **时序重构**：wait → send → process（CPU 与 NPU 重叠）
3. **状态管理**：pending futures + batch 双缓冲
4. **风险控制**：保持原接口兼容，渐进式实现

预期效果：**decode 吞吐提升 ~18%**，单 token 时延不变。

下一步：按 Phase 1-5 路径渐进实现，每阶段充分测试验证。