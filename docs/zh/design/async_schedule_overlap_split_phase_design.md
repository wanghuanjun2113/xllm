# 异步调度重叠 Split-Phase 设计文档（对标 vLLM）

## 概述

本文档重新设计 `enable_schedule_overlap` 的实现路径，目标是在**不改写现有两阶段语义**的前提下，把 decode 阶段的 Host 侧输出处理移动到更合适的时机，与下一步 NPU 计算重叠执行。

与已有方案不同，本文档不把 overlap 问题抽象成“给 `Engine::step()` 加一个 future”这么简单，而是显式保留当前实现已经依赖的两阶段契约：

1. **阶段 A：materialize scheduled step**
   当前 step 完成后，把请求状态推进到“下一步可调度”的形态。
2. **阶段 B：commit true output**
   用真实输出回填上一步请求中的占位 token，并触发流式输出、结束态处理等 Host 逻辑。

本文档的核心结论是：

- 真正需要拆开的不是“同步接口 vs 异步接口”，而是 `step()` 里被捆在一起的“提交计算”和“等待并执行阶段 A”。
- `update_last_step_result()` 对应的阶段 B 不能被吞并掉，因为当前 worker 侧已经依赖 `last_step_output_` 机制在下一步执行前修正输入。
- Scheduler 侧不需要引入新的 batch 双缓冲；复用现有 `last_batch_`、`last_running_requests_`、`last_running_sequences_` 即可。

本文档的设计目标包括：

- 在 decode 场景下，把“上一步真实输出回填 + Host 侧请求处理”与“下一步 NPU 计算”重叠起来
- 保留当前 fake-token / true-token 两阶段语义
- 最小化对现有 worker、RPC、request/sequence 状态机的侵入
- 为 `LLMEngine` 和 `VLMEngine` 提供统一的 split-phase 实现框架

本文档的非目标包括：

- 不重新设计 request / sequence 的 token 状态机
- 不把 overlap 扩展到多步并行 in-flight
- 不在第一阶段覆盖 speculative decoding、`RecEngine` 多轮解码、prefill overlap
- 不修改原有 `enable_schedule_overlap` 对外语义

需要特别说明的是，这份设计并不是孤立地“把 xLLM 现有 overlap 修一修”，
而是显式以 vLLM 当前 async scheduling 的成熟实现为参考系，补齐 xLLM
还缺失的四个闭环：

1. 调度器持有 placeholder 账本，而不是把 fake token 语义扩散到请求输出、
   stopping check、统计口径。
2. Engine 只负责 submit / wait / aggregate，Scheduler 才拥有“何时消费结果、
   何时提交下一轮”的控制权。
3. 能力门禁要集中决策，而不是散落在不同 engine / master 分支里各自强制关掉。
4. 验证必须覆盖控制平面时序与功能组合矩阵，而不只是局部 batch / sequence
   单测。

## 1. 当前实现的真实语义

### 1.1 当前 overlap 不是“半异步 step”，而是“两阶段 step”

当前 `enable_schedule_overlap=true` 时，`ContinuousScheduler::step_with_schedule_overlap()` 的关键流程是：

1. `schedule_request()` 组装 `batch_N`
2. `engine_->step(batch_N)`
3. `engine_->update_last_step_result(batch_{N-1})`
4. `process_batch_output(true)`

其中，`engine_->step(batch_N)` 内部虽然调用了 worker 侧的 `step_async()`，但会立即 `collectAll(...).get()` 等待完成。更重要的是，等待完成后它并不是简单“拿到结果就返回”，而是已经做了**阶段 A**：

- 对 sample 路径执行 `process_sample_output(..., false)`
- 对 beam 路径执行 `process_beam_search_output(..., false)`
- 把 `batch_N` 推进到“下一步可调度”的状态

在当前 overlap 语义下，这一步并不是最终提交真实 token，而是把请求状态推进到一个“可继续发下一步”的中间态。

### 1.2 当前 worker 已经依赖上一轮真实输出来修正下一轮输入

当前 `WorkerImpl::step_async()` 在 overlap 模式下会：

1. 如果存在 `last_step_output_`，先用它修正当前 step 输入中的占位 token
2. 执行当前 step
3. 把当前 step 的真实输出保存到 `last_step_output_`
4. 供后续 `get_last_step_result_async()` 读取

因此，当前链路实际上已经有两个消费者在使用“同一个 step 的真实结果”：

- `step()` 等待 worker future 完成后，用于执行阶段 A
- `update_last_step_result()` 通过 `get_last_step_result_async()` 读取，用于执行阶段 B

这不是偶然实现细节，而是现有 overlap 语义成立的前提。

### 1.3 当前性能问题的根因

当前实现的问题不在于“没有 async API”，而在于调度顺序：

```text
schedule_request(N)
  -> wait step(N) 完成并执行阶段 A
  -> update_last_step_result(N-1) 执行阶段 B
  -> process_batch_output(N-1)
```

也就是说，**阶段 B 被放在了当前 step 的阶段 A 之后**。这样会导致：

- Scheduler 必须先等待 `step(N)` 结束
- 然后才能处理 `step(N-1)` 的真实输出
- Host 侧处理 `N-1` 的时间无法与 `N` 的设备计算重叠

真正应该调整的是等待点，而不是把阶段 A / 阶段 B 混成一个 future。

### 1.4 对照 vLLM 后，xLLM 还缺什么

对照 vLLM 当前实现，可以更清楚地看到 xLLM “异步调度还不彻底”不是单点问题，
而是少了配套抽象：

1. **调度器 placeholder 账本缺失**

   vLLM 的 `AsyncScheduler` 会在 scheduler/request 层维护
   `num_output_placeholders`、`discard_latest_async_tokens`、
   `spec_token_ids` 这些异步状态，异步 token 是否“已提交给业务语义”
   是显式可数的。xLLM 当前则把 fake token `-1` 直接写入
   `Sequence::tokens_`，于是 `Request::generate_output()`、
   `StoppingChecker::check()` 等非调度代码都得知道 overlap 的存在。

2. **Engine 缺少 async-first 的结果句柄**

   vLLM 的 async 路径里，`execute_model(..., non_block=True)`、
   `step_with_batch_queue()`、`AsyncModelRunnerOutput` 共同保证
   “提交”和“消费结果”是显式分离的。xLLM 当前虽然调用了
   `step_async()`，但 `LLMEngine::step()` 立即 `collectAll(...).get()`，
   结果还是在 engine 内部被同步收口。

3. **能力门禁不集中**

   vLLM 会在配置阶段统一检查 executor 能力和 feature 组合兼容性，
   明确决定 async scheduling 是启用、自动关闭还是显式报错。xLLM 当前
   `master.cpp` 中对 embedding、REC、disagg PD prefill、VLM 等路径的
   禁用逻辑是分散的，用户很难知道“为什么某个组合不支持”。

4. **验证矩阵不完整**

   vLLM 已经把 preemption、executor、spec decode、chunked prefill
   等组合放进 e2e 回归。xLLM 目前更多覆盖的是 fake token 相关局部逻辑，
   但缺少控制面时序的证据链，例如：
   `wait previous -> schedule current -> submit current -> commit previous`
   是否真的被严格执行。

## 2. 设计原则

### 2.1 保留两阶段契约

本设计明确保留下面的契约：

- **阶段 A**：step 计算完成后，把 batch 推进到“下一步可调度”的状态
- **阶段 B**：在下一步已提交之后，再把上一步的占位 token 回填为真实 token

阶段 A 和阶段 B 分别对应不同的状态迁移，不能合并。

### 2.2 Split-Phase，而不是 Future-of-Batch

Scheduler 真正需要的是：

1. 提交当前 step，不阻塞
2. 在下一轮开始时，等待该 step 完成并执行阶段 A
3. 在提交新的 step 后，执行上一轮的阶段 B

因此 Engine 层最自然的接口不是 `SemiFuture<std::vector<Batch>>`，而是 split-phase API：

- `step_async(batch)`：只负责提交当前 step
- `wait_pending_step(batch)`：等待 pending step 完成，并对这个 batch 执行阶段 A

### 2.3 batch 生命周期归 Scheduler 所有

`Batch` 内部包含 `Sequence*`、`SequencesGroup*` 等指针语义对象。为了避免：

- engine 内部复制 `Batch`
- future 闭包持有 batch 引用
- scheduler 和 engine 同时维护 batch 所有权

本设计要求 **Scheduler 始终持有 batch 生命周期**。Engine 只持有 pending futures，不持有 `pending_batch_` 副本。

### 2.4 系统里始终只有一个 in-flight step

当前 worker 线程池和 overlap 语义都建立在“单 worker 顺序执行 step”的前提上。本文档不改变这一点：

- 同一时间只允许一个 pending step
- 不追求多级设备流水
- 不引入 batch 级别乱序完成

### 2.5 对齐 vLLM 的责任边界

为了避免继续把“异步”实现成一组零散补丁，本设计把责任边界明确成三层：

1. **Scheduler 负责请求可见状态**

   它拥有 request/sequence 的调度视图，决定“当前 batch 是否已经
   materialized，可不可以拿来调度下一轮”。

2. **Engine 负责 in-flight step**

   它只保存 pending step 的 futures 和聚合元数据，只负责 submit /
   wait / aggregate，不拥有 `Batch` 生命周期。

3. **ResponseProcessor 负责 Host 回调**

   它负责 stream / non-stream 输出串行化与 RPC 发送，但不再承担
   “修正异步 token 账本”的职责。

这个分层基本对应了 vLLM 中 Scheduler、EngineCore/Executor、
OutputProcessor 的职责切分，也是 xLLM 后续继续演进的稳定边界。

## 3. 新的接口设计

### 3.1 Engine 接口

在 `engine.h` 中新增 split-phase 接口：

```cpp
class Engine {
 public:
  virtual ~Engine() = default;

  virtual ForwardOutput step(std::vector<Batch>& batch) = 0;
  virtual void update_last_step_result(std::vector<Batch>& batch) = 0;

  // overlap split-phase API
  virtual void step_async(std::vector<Batch>& batch) {
    NOT_IMPLEMENTED();
  }

  virtual void wait_pending_step(std::vector<Batch>& batch) {
    NOT_IMPLEMENTED();
  }

  virtual bool has_pending_step() const { return false; }
};
```

### 3.2 同步接口作为兼容包装层

为了保持现有调用方兼容，`LLMEngine::step()` / `VLMEngine::step()` 可以退化为：

```cpp
ForwardOutput LLMEngine::step(std::vector<Batch>& batch) {
  step_async(batch);
  wait_pending_step(batch);
  return {};
}
```

这样有两个好处：

- 旧调用方不需要立刻改
- split-phase 路径和同步路径共享同一套阶段 A 逻辑，避免逻辑复制

### 3.3 为什么不返回 future

本设计明确不采用：

```cpp
folly::SemiFuture<std::vector<Batch>> step_async(std::vector<Batch>& batch);
```

原因如下：

1. Scheduler 不需要 future 值，只需要“何时等待”的控制权
2. future 版本会诱导 engine 持有 `Batch` 副本，增加生命周期复杂度
3. future 容易把阶段 A 和阶段 B 混成“一次完成”
4. 当前实现天然只有一个 pending step，不需要额外的 promise/future 编排

## 4. Engine 侧实现

### 4.1 LLMEngine：新增 pending futures，不新增 pending batch

`LLMEngine` 新增成员：

```cpp
class LLMEngine : public Engine {
 private:
  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>>
      pending_futures_;
  bool has_pending_ = false;
};
```

不新增 `pending_batch_`，因为 batch 生命周期由 Scheduler 持有。

### 4.2 `step_async(batch)` 只负责提交计算

`LLMEngine::step_async(batch)` 的职责是把当前 `step()` 里的“提交 worker 异步请求”部分拿出来：

1. 按当前逻辑准备 `RawForwardInput`
2. 调用所有 worker 的 `step_async()`
3. 保存 `pending_futures_`
4. 设置 `has_pending_ = true`

它**不做**以下事情：

- 不等待 `collectAll`
- 不处理 sample / beam 输出
- 不修改 batch
- 不 fulfill 任何 promise

### 4.3 `wait_pending_step(batch)` 负责执行阶段 A

`LLMEngine::wait_pending_step(batch)` 的职责是把当前 `step()` 的后半段抽出来：

1. 等待 `pending_futures_`
2. 按当前 `step()` 的逻辑聚合 DP / TP / CP 结果
3. 在 sample 路径上执行 `batch.process_sample_output(result, false)`
4. 在 beam 路径上执行 `batch.process_beam_search_output(result, false)`
5. 调用 `batch.refresh_sequences_from_groups()`
6. 清空 pending 状态

关键点：

- `wait_pending_step(batch)` 执行的是**阶段 A**
- 它只把 batch 推进到“下一步可调度”状态
- 它**不**替代 `update_last_step_result(batch)`

### 4.4 `update_last_step_result(batch)` 继续负责阶段 B

`update_last_step_result(batch)` 维持当前语义：

1. 从 worker 侧读取 `get_last_step_result_async()`
2. 处理 EPLB 需要的附加结果
3. 对 batch 执行 `process_sample_output(..., true)`
4. 刷新 sequence group 映射

也就是说，`wait_pending_step(batch)` 和 `update_last_step_result(batch)` 分别负责：

- `replace_fake_token = false`
- `replace_fake_token = true`

这两个阶段都保留。

### 4.5 Worker / RPC 层不改协议

本设计不要求修改以下组件的协议语义：

- `WorkerImpl::step_async()`
- `Worker::step_async()`
- `RemoteWorker::step_async()`
- `WorkerClient::get_last_step_result_async()`
- `RemoteWorker::get_last_step_result_async()`

原因是当前 worker 侧已经满足 split-phase 方案需要的两个条件：

1. step 提交本身就是异步的
2. 真实输出已经被保存在 `last_step_output_` 中，可供阶段 B 读取

如需修改，只建议补充注释，澄清“同一步结果分别服务于阶段 A 和阶段 B”。

### 4.6 分布式聚合规则必须沿用现有实现

这里不能重新发明索引规则，必须直接继承现有实现：

- `step_async + wait_pending_step` 使用当前 `step()` 的聚合逻辑
  - `LLMEngine` 按 `dp_local_size_` 选取每个 DP group 的 driver 结果
  - CP 场景沿用现有输入切分逻辑
- `update_last_step_result` 使用当前 `stride` 规则
  - 默认 `stride = dp_local_tp_size_`
  - `FLAGS_enable_eplb` 时 `stride = 1`

这样可以避免 CP / EPLB 场景下出现“本地单卡对、分布式错”的偏差。

### 4.7 VLMEngine 采用同样的 split-phase 结构

`VLMEngine` 的改法与 `LLMEngine` 一致：

- 抽出 `step_async(batch)`
- 抽出 `wait_pending_step(batch)`
- `step(batch)` 退化为包装器
- `update_last_step_result(batch)` 保持为阶段 B

第一阶段先保证 `LLMEngine` 路径正确，再复用同一模式到 `VLMEngine`。

## 5. Scheduler 侧设计

### 5.1 不新增 batch 双缓冲

Scheduler 侧保留现有字段：

- `last_batch_`
- `last_running_requests_`
- `last_running_sequences_`
- `is_first_step_`

仅新增：

```cpp
bool has_inflight_step_ = false;
```

这里的 `last_batch_` 继续表示“上一轮已经提交的 batch”。  
它在一个迭代周期内会依次经历两种语义：

1. 迭代开始时：等待其 pending step 完成，并执行阶段 A
2. 提交新 step 后：对其执行阶段 B，并做 Host 侧请求处理

因此不需要再额外引入 `batch_buffer_[2]`。

### 5.2 新的执行顺序

新的 `step_with_schedule_overlap_v2()` 顺序如下：

```cpp
void ContinuousScheduler::step_with_schedule_overlap_v2(
    const absl::Duration& timeout) {
  bool last_batch_ready = false;

  // 1. 等待上一轮 pending step 完成，并执行阶段 A
  if (has_inflight_step_) {
    engine_->wait_pending_step(last_batch_);
    has_inflight_step_ = false;
    last_batch_ready = !all_batch_empty(last_batch_);
  }

  // 2. 基于阶段 A 后的请求状态组装当前 batch
  std::vector<Batch> batch = schedule_request(timeout);
  bool cur_batch_empty = all_batch_empty(batch);

  // 3. 提交当前 step，不等待
  if (!cur_batch_empty) {
    engine_->step_async(batch);
    has_inflight_step_ = true;
    kv_cache_manager_->reset_transfer_infos();
  }

  // 4. 对上一轮 batch 执行阶段 B，并做 Host 侧输出处理
  if (!is_first_step_ && last_batch_ready) {
    engine_->update_last_step_result(last_batch_);
    process_batch_output(true);
  }

  // 5. 无工作时直接返回
  if (cur_batch_empty && !last_batch_ready) {
    return;
  }

  // 6. 保存当前 batch，供下一轮使用
  last_batch_ = std::move(batch);
  last_running_sequences_ = running_sequences_;
  last_running_requests_ = running_requests_;
  is_first_step_ = false;
}
```

### 5.3 时序说明

新流程的时间轴如下：

```text
迭代 N:

wait_pending_step(batch_{N-1})   // 等待 N-1 完成，执行阶段 A
schedule_request(batch_N)        // 基于 N-1 的阶段 A 状态调度 N
step_async(batch_N)              // 提交 N，不等待
update_last_step_result(N-1)     // 阶段 B，与 N 的 NPU 计算重叠
process_batch_output(N-1)        // Host 逻辑，与 N 的 NPU 计算重叠
```

与当前实现相比，唯一被移动的是：

- 当前：先等待 `step(N)`，再处理 `N-1`
- 新方案：先提交 `N`，再处理 `N-1`

这样既不改变 token 状态机，又能把 Host 侧工作挪到 NPU 计算窗口中。

### 5.4 两层落地：先拿到真重叠，再收敛到 vLLM 式账本

这里需要明确区分两个层次：

1. **L0：控制面真异步**

   先通过 split-phase 把当前“伪异步”改成真正的 submit/wait 重排，
   这是吞吐收益的直接来源。

2. **L1：语义层去 fake-token 泄漏**

   再逐步把 fake token 从对外可见业务语义中收回来，减少它对
   stopping / usage / streaming / metrics 的侵入。

L0 和 L1 不是二选一。L0 解决“没有真正 overlap”，L1 解决
“overlap 做到了，但副作用仍然蔓延到业务逻辑”。

第一版实现允许继续沿用当前 fake-token 两阶段语义，但接口设计要为
L1 预留空间，不能把 `-1` 永久固化成 request/sequence 的公共语义。

### 5.5 引入 `PendingStepHandle`，不要让 future 向外裸奔

对标 vLLM 的 `AsyncModelRunnerOutput`，xLLM 在 engine 内部也应避免让
“一组 `std::vector<folly::SemiFuture<...>>` + 若干聚合规则”散落在多个
调用路径。建议引入一个私有的 `PendingStepHandle`（第一阶段可以只是
`llm_engine.cpp` 内部 struct），统一封装：

- worker futures
- DP / CP / EPLB 聚合所需元数据
- `model_executed` / `layer_forward_interrupted_` 等辅助状态
- 结果清理逻辑

这样 `step_async()` / `wait_pending_step()` 的接口语义会更稳定，
后续 `VLMEngine` 复用时也不必再次复制“收 futures + 聚合 + 清理”的样板。

### 5.6 能力门禁前移到统一校验点

对标 vLLM 的 `supports_async_scheduling()` 和统一 config gating，xLLM 需要
一个集中校验点，负责在进入 scheduler 热路径之前决定 overlap 是否有效。

建议至少统一评估以下维度：

- model family：LLM 支持，embedding / REC 不支持，VLM 待补齐
- instance role：disagg PD prefill 不支持，decode 按场景评估
- topology：单进程 / 远端 worker / DP / TP / CP / EPLB
- request mode：stream / non-stream / batch response
- feature mix：chunked prefill、beam search、speculative、cache transfer

设计要求：

1. 用户显式开启但组合不支持时，必须 fail-closed 或至少强告警并给出原因。
2. 决策结果要有统一日志与指标，而不是分散在 `master.cpp` 的多处分支。
3. 后续如需演进到 `auto/on/off` 三态，可以在这个统一校验点之上扩展，
   而不是再新增一套分支判断。

## 6. 状态机

### 6.1 单 batch 状态机

对某个 `batch_k`，状态转换如下：

```text
Submitted
  -- wait_pending_step(batch_k) -->
Materialized
  -- update_last_step_result(batch_k) -->
Committed
```

语义分别是：

- `Submitted`
  - step 已提交到 worker
  - 结果尚未等待
  - batch 还不能作为下一轮调度依据
- `Materialized`
  - step 已完成
  - 阶段 A 已执行
  - batch 已具备“下一轮可调度”状态
- `Committed`
  - 阶段 B 已执行
  - 真实输出已回填
  - 请求可安全触发流式输出、终止检查等 Host 逻辑

### 6.2 Scheduler 全局状态

Scheduler 全局只需要维护两个事实：

1. 是否存在 `has_inflight_step_`
2. `last_batch_` 是否已经进入 `Materialized` 状态

`Materialized` 状态无需单独字段，可由本轮是否执行过 `wait_pending_step(last_batch_)` 得到。

## 7. 正确性约束

实现时必须满足以下不变量：

1. 任意时刻最多只有一个 pending step
2. `schedule_request()` 之前，必须先对上一轮 batch 执行 `wait_pending_step()`
3. `update_last_step_result(batch_k)` 只能在 `wait_pending_step(batch_k)` 之后执行
4. Scheduler 必须持有 `batch_k` 到阶段 B 完成
5. Worker 侧 step 执行顺序仍然串行，不允许并发跑多个 step

如果这些约束被破坏，最容易出现的问题包括：

- 下一轮调度看不到上一轮的阶段 A 状态
- placeholder token 被回填两次或完全没回填
- worker 输入修正和 batch 输出修正不再对应同一轮 step

## 8. 边界情况

### 8.1 首次 step

第一次进入循环时：

- 没有上一轮 pending step
- 也没有 `last_batch_` 可做阶段 B

因此：

- 跳过 `wait_pending_step()`
- 跳过 `update_last_step_result()`
- 只提交当前 step

### 8.2 尾批次 drain

当没有新请求，但上一轮还有 pending step 时，仍需完成以下流程：

1. `wait_pending_step(last_batch_)`
2. `schedule_request()` 得到空 batch
3. 不再提交新 step
4. 对 `last_batch_` 执行阶段 B
5. 调用 `process_batch_output(true)`

这保证最后一轮不会丢结果。

### 8.3 Request cancel / preempt

这个风险不能只看 `request->cancelled()`。

当前代码已经存在这样一种情况：

- sequence 在 `schedule_request()` 期间被 preempt
- KV cache 被释放
- 但这个 sequence 仍在 `last_batch_` 中等待阶段 B

因此 split-phase 方案必须继续保留当前 `Sequence::update_last_step_token()` 中对
`kv_state_.num_kv_blocks() == 0` 的保护，不能把它当成“旧逻辑遗留”删掉。

### 8.4 Chunked prefill

第一阶段不扩展 chunked prefill 语义，只要求：

- 不破坏 `Batch::update_sequence_state()` 当前对
  `pre_scheduled_step_prefill_queue()` 的维护
- 对已有 overlap 可运行路径保持兼容

如果后续要做 prefill overlap，应单独设计，不与本文档耦合。

### 8.5 Beam search

阶段 A 必须保留当前分支：

- sample 路径：`process_sample_output(..., false)`
- beam 路径：`process_beam_search_output(..., false)`

本设计不尝试扩大 beam search 的 overlap 语义范围，只要求与当前行为一致。

### 8.6 异常与中断

`wait_pending_step(batch)` 中如果出现：

- `collectAll(...).get()` 抛错
- `ForwardInterruptedException`
- worker 结果缺失

则必须保证：

1. `has_pending_` 被清理
2. `has_inflight_step_` 不再误认为仍有 pending step
3. Scheduler 不会对这轮 batch 再执行阶段 B

这里建议用局部 guard 或明确的异常清理路径，避免把 scheduler 卡在“永远等待同一个 pending step”的状态。

### 8.7 长期收敛：把 fake token 从业务语义层移除

当前 fake token 机制已经泄漏到多个非调度逻辑：

- `Request::generate_output()` 需要手动减去一个额外 step
- `StoppingChecker::check()` 需要倒序跳过负 token
- `Sequence::update_last_step_token()` 需要维护额外索引与补写逻辑
- 文档层面还要提醒“输出 token 较少时会影响吞吐”

这正是 xLLM “异步做得不彻底”的第二个根因。

对标 vLLM，建议把这部分作为 **Phase 4** 的语义收敛项，方向是引入
**scheduler-owned placeholder ledger**，而不是继续让 `-1` 充当公共 token：

- `RequestState` 维护 pending/committed decode token 计数
- `Sequence` 区分“调度已预占”与“业务可见已提交”
- `Batch::output_targets_` 继续承担 commit 目标映射，但 fake token 不再进入
  对外可见 `tokens_`

这样 stopping、usage、streaming 只看 committed token，调度器仍然可以保留
两阶段语义，但 overlap 的实现细节不会继续污染业务层代码。

## 9. 验证与测试策略

### 9.1 Engine 单测

新增 split-phase 单测，至少覆盖：

1. `step_async(batch)` 不阻塞，并设置 pending 状态
2. `wait_pending_step(batch)` 能正确清空 pending，并执行阶段 A
3. `step(batch)` 与 `step_async + wait_pending_step` 行为一致
4. `update_last_step_result(batch)` 仍按阶段 B 语义回填真实 token

### 9.2 Scheduler 单测

重点验证顺序，而不是只验证“最终能跑完”：

1. `wait_pending_step(last_batch_)` 发生在 `schedule_request()` 之前
2. `step_async(cur_batch)` 发生在 `update_last_step_result(last_batch_)` 之前
3. `process_batch_output(true)` 只在阶段 B 之后执行
4. 尾批次 drain 场景不会漏掉最后一轮输出

### 9.3 回归测试

至少覆盖以下回归场景：

- 单 batch decode
- 多 batch 连续 decode
- request cancel
- request preempt / KV cache 回收
- beam search 基础路径
- `FLAGS_enable_eplb=true`
- 多 DP / TP / CP 组合

另外建议直接借鉴 vLLM 的 async scheduling 组合测试思路，增加一组
受控笛卡尔积 smoke matrix。第一阶段不必做全量爆炸组合，但至少覆盖：

- overlap `{off, on}`
- executor/topology `{single-process, remote-worker}`
- preemption `{off, on}`
- chunked prefill `{off, on}`
- stream mode `{off, on}`
- distributed features `{plain, CP, EPLB}`

每个组合都至少验证两类证据：

1. 功能正确性：输出 token、finish reason、usage 与非 overlap 一致
2. 时序正确性：`wait_pending_step(last_batch_)` 先于 `schedule_request()`，
   `step_async(cur_batch)` 先于 `update_last_step_result(last_batch_)`

### 9.4 性能测试

不在设计文档中预设固定收益值，统一以实测为准。建议指标包括：

- tokens/s
- inter-token latency
- `engine_latency_seconds`
- Host 侧阶段 B 耗时
- wait pending step 耗时

建议至少做三组对比：

1. `enable_schedule_overlap=false`
2. 当前 overlap 实现
3. split-phase overlap 实现

## 10. 实施路径

### Phase 0：统一能力门禁与观测点

涉及：

- `xllm/core/distributed_runtime/master.cpp`
- `xllm/core/distributed_runtime/engine.h`
- 相关日志与 metrics 埋点

动作：

1. 收敛所有 overlap enable / disable 决策到单点函数
2. 明确每一种禁用路径的 reason code / warning
3. 暴露 overlap 最终生效状态，便于性能与问题排查

### Phase 1：Engine split-phase 化

涉及：

- `xllm/core/distributed_runtime/engine.h`
- `xllm/core/distributed_runtime/llm_engine.h`
- `xllm/core/distributed_runtime/llm_engine.cpp`

动作：

1. 新增 `step_async()` / `wait_pending_step()`
2. 让 `step()` 成为同步包装器
3. 保持 `update_last_step_result()` 不变

### Phase 2：ContinuousScheduler 重排顺序

涉及：

- `xllm/core/scheduler/continuous_scheduler.h`
- `xllm/core/scheduler/continuous_scheduler.cpp`

动作：

1. 新增 `has_inflight_step_`
2. 改写 overlap 路径为 `wait -> schedule -> submit -> commit`
3. 保持现有 `process_batch_output(true)` 路径

### Phase 3：VLMEngine 复用模式

涉及：

- `xllm/core/distributed_runtime/vlm_engine.h`
- `xllm/core/distributed_runtime/vlm_engine.cpp`

动作：

1. 按 `LLMEngine` 同样方式拆分阶段 A
2. 保持阶段 B 路径不变

### Phase 4：placeholder ledger 收敛

涉及：

- `xllm/core/framework/request/request_state.h`
- `xllm/core/framework/request/request.cpp`
- `xllm/core/framework/request/sequence.h`
- `xllm/core/framework/request/sequence.cpp`
- `xllm/core/framework/request/stopping_checker.cpp`
- `xllm/core/framework/batch/batch.cpp`

动作：

1. 把 fake token 对 usage / stopping / streaming 的影响收敛到调度内部
2. 逐步用 placeholder 计数替代“到处跳过 `-1`”的逻辑
3. 确保外部业务语义只看到 committed token

### Phase 5：验证与性能回归

涉及：

- scheduler 单测
- engine 单测
- 分布式回归
- 基准性能测试

### Phase 6：清理与收口

在 split-phase 路径稳定后，再考虑：

- 是否统一 `LLMEngine` / `VLMEngine` 的公共阶段 A 辅助逻辑
- 是否补充更准确的注释和 tracing
- 是否继续扩展到其他 engine

## 11. 预期收益与风险

### 11.1 预期收益

本方案的收益来源是：

- 不再把 `N-1` 的 Host 侧阶段 B 串行放在 `N` 的 NPU 等待之后
- 尽量把 `update_last_step_result()` 和 `process_batch_output(true)` 放到 `step(N)` 的设备计算窗口中

它不会带来：

- 多个 step 同时在设备上并行
- worker 线程池并发执行多个 decode step

因此，收益上限取决于：

- Host 侧阶段 B 的占比
- 当前 NPU 计算时间
- request 规模和调度粒度

### 11.2 主要风险

主要风险包括：

1. 阶段 A / 阶段 B 边界实现错位，导致 token 状态机错乱
2. 分布式聚合索引处理不一致，导致 CP / EPLB 场景行为错误
3. drain / interrupt 路径清理不完整，导致 pending 状态泄漏
4. 过早抽象公共逻辑，反而把 `LLMEngine` / `VLMEngine` 的差异抹平出错

## 12. 总结

这份重构方案的关键，不是把 `step()` 改成返回 future，而是把当前 overlap 里已经存在的两阶段状态机显式化：

1. `step_async(batch)`：提交 step
2. `wait_pending_step(batch)`：等待 step 完成并执行阶段 A
3. `update_last_step_result(batch)`：执行阶段 B

在这个前提下，Scheduler 只需要把顺序调整为：

```text
wait previous -> schedule current -> submit current -> commit previous
```

这样既能复用当前 worker / request / sequence 语义，又能把 Host 侧输出处理移动到下一轮 NPU 计算窗口中，实现真正有意义的 schedule overlap。
