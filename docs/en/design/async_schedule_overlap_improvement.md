# Async Schedule Overlap Improvement Design

> Reference: vLLM v1 async scheduling pattern
> Related Chinese docs: [split-phase design](../zh/design/async_schedule_overlap_split_phase_design.md), [v2 design](../zh/design/async_schedule_overlap_v2_design.md)
> Date: 2026-04-24
> Status: Supplementary Design (English translation + vLLM comparison)

## Note

**This document supplements existing Chinese design documents:**

- **`async_schedule_overlap_split_phase_design.md`**: Comprehensive split-phase design with detailed implementation plan
- **`async_schedule_overlap_v2_design.md`**: V2 design with time sequence diagrams and correctness constraints

The Chinese documents are more detailed and should be the primary reference. This English document:
1. Provides an English summary for international contributors
2. Adds vLLM batch_queue pattern comparison
3. Documents the root cause analysis with exact code locations

## 1. Current Problem Analysis

### 1.1 xLLM Current Implementation

xLLM has partial async scheduling support via `enable_schedule_overlap` flag, but the implementation has critical blocking issues that prevent true overlap.

**Current flow in `step_with_schedule_overlap()` (`continuous_scheduler.cpp:1085-1115`):**

```cpp
// Current implementation (PROBLEM: blocking)
void ContinuousScheduler::step_with_schedule_overlap(const absl::Duration& timeout) {
    // 1. Schedule batch_N (CPU)
    std::vector<Batch> batch = schedule_request(timeout);
    
    // 2. Execute batch_N - BLOCKS on NPU!
    if (!cur_batch_all_empty) {
        engine_->step(batch);  // <-- BLOCKING CALL
        kv_cache_manager_->reset_transfer_infos();
    }
    
    // 3. Update last step result - BLOCKS again!
    if (!is_first_step_ && !last_batch_all_empty) {
        engine_->update_last_step_result(last_batch_);  // <-- BLOCKING CALL
        process_batch_output(true);
    }
    
    last_batch_ = std::move(batch);
}
```

**Root cause: `LLMEngine::step()` blocks immediately (`llm_engine.cpp:1031-1032`):**

```cpp
// Send async requests to workers
futures.emplace_back(worker_clients_[worker_rank]->step_async(*input_to_send));

// BUT then immediately blocks!
auto results = folly::collectAll(futures).get();  // <-- PROBLEM!
```

### 1.2 vLLM Reference Implementation

vLLM v1 achieves true async scheduling via batch_queue mechanism:

**vLLM `step_with_batch_queue()` (`engine/core.py:445-561`):**

```python
def step_with_batch_queue(self):
    # 1. Try to schedule new batch if queue not full (NON-BLOCKING)
    if self.scheduler.has_requests() and len(batch_queue) < max_size:
        scheduler_output = self.scheduler.schedule()
        exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
        batch_queue.appendleft((future, scheduler_output, exec_future))
        
        # Don't block if queue not full - return immediately
        if len(batch_queue) < batch_queue_size and not batch_queue[-1][0].done():
            return None, True  # <-- NON-BLOCKING RETURN
    
    # 2. Only block when queue is full OR no more requests to schedule
    future, scheduler_output = batch_queue.pop()
    model_output = future.result()  # <-- BLOCKING WAIT (but only when necessary)
    engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
```

**Key patterns:**
1. `execute_model(non_block=True)` returns Future, doesn't wait
2. batch_queue stores multiple pending futures
3. Only block when queue is full or no more work
4. CPU processing overlaps with GPU computation

### 1.3 Gap Analysis

| Aspect | vLLM (Reference) | xLLM (Current) | Gap |
|--------|------------------|----------------|-----|
| Engine API | `step()` returns after submitting | `step()` blocks on futures | **Critical** |
| Batch queue | deque of pending futures | single `last_batch_` | **Critical** |
| Execution | `non_block=True` + Future | `step_async()` then `collectAll().get()` | **Critical** |
| Ordering | Schedule → Submit → Wait previous → Process | Schedule → Block → Block → Process | **Critical** |
| Overlap | CPU process overlaps with GPU | No overlap (both block) | **Critical** |

---

## 2. Proposed Design: Split-Phase Async API

### 2.1 Engine Interface Changes

**Add split-phase async methods to `Engine` class (`engine.h`):**

```cpp
class Engine {
 public:
  // Existing synchronous methods (backward compatible)
  virtual ForwardOutput step(std::vector<Batch>& batch) = 0;
  virtual void update_last_step_result(std::vector<Batch>& batch) = 0;
  
  // NEW: Split-phase async methods
  // Phase 1: Submit execution without waiting
  virtual void step_async(std::vector<Batch>& batch) = 0;
  
  // Phase 2: Wait for pending step to complete (Phase A: materialize fake tokens)
  virtual void wait_pending_step(std::vector<Batch>& batch) = 0;
  
  // Check if there's a pending step waiting
  virtual bool has_pending_step() const = 0;
  
  // Get futures for pending step (for batch_queue implementation)
  virtual std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>>& 
      get_pending_futures() = 0;
};
```

### 2.2 LLMEngine Implementation

**Modify `LLMEngine::step()` to support non-blocking execution:**

```cpp
class LLMEngine : public Engine {
 private:
  // Store pending futures from last step_async() call
  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> pending_futures_;
  std::vector<Batch> pending_batch_;  // Batch associated with pending_futures_
  std::mutex pending_mutex_;
  bool has_pending_ = false;
  
 public:
  // NEW: Submit without blocking
  void step_async(std::vector<Batch>& batch) override {
    std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
    futures.reserve(worker_clients_num_);
    
    for (auto worker_rank = 0; worker_rank < worker_clients_num_; ++worker_rank) {
      futures.emplace_back(worker_clients_[worker_rank]->step_async(*input_to_send));
    }
    
    // Store futures WITHOUT waiting - key change!
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      pending_futures_ = std::move(futures);
      pending_batch_ = batch;  // Store batch for later retrieval
      has_pending_ = true;
    }
    
    // Phase A: Materialize fake tokens immediately
    // This allows next step scheduling to proceed
    for (size_t dp_rank = 0; dp_rank < batch.size(); ++dp_rank) {
      batch[dp_rank].process_sample_output(RawForwardOutput{}, false);  // fake tokens
    }
  }
  
  // NEW: Wait for pending step (Phase A complete, Phase B begins)
  void wait_pending_step(std::vector<Batch>& batch) override {
    std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      futures = std::move(pending_futures_);
      batch = std::move(pending_batch_);
      has_pending_ = false;
    }
    
    // NOW we wait - but this happens during next step's NPU computation
    auto results = folly::collectAll(futures).get();
    
    // Phase B results ready (stored in workers for update_last_step_result)
  }
  
  bool has_pending_step() const override {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    return has_pending_;
  }
};
```

### 2.3 Scheduler Implementation: Batch Queue Pattern

**Introduce batch_queue mechanism similar to vLLM:**

```cpp
class ContinuousScheduler {
 private:
  // NEW: Batch queue for async overlap
  struct PendingStep {
    std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
    std::vector<Batch> batch;
    bool submitted;
  };
  
  std::deque<PendingStep> batch_queue_;
  size_t batch_queue_size_ = 2;  // Allow 2 pending batches (configurable)
  
 public:
  void step_with_batch_queue(const absl::Duration& timeout) {
    // 1. Try to schedule new batch if queue not full
    if (batch_queue_.size() < batch_queue_size_) {
      std::vector<Batch> batch = schedule_request(timeout);
      if (!all_empty(batch)) {
        // Submit WITHOUT waiting
        engine_->step_async(batch);
        batch_queue_.push_front({/*futures from engine*/, batch, true});
        
        // If queue still has room, return immediately (don't block)
        if (batch_queue_.size() < batch_queue_size_) {
          return;  // NON-BLOCKING - overlap achieved!
        }
      }
    }
    
    // 2. Queue is full, must wait for oldest batch
    if (!batch_queue_.empty()) {
      PendingStep oldest = batch_queue_.back();
      batch_queue_.pop_back();
      
      // NOW wait for results (during which new batch NPU computation proceeds)
      engine_->wait_pending_step(oldest.batch);
      
      // Phase B: Update with real tokens
      engine_->update_last_step_result(oldest.batch);
      
      // Process outputs (CPU work overlaps with next batch NPU)
      process_batch_output(oldest.batch, true);
    }
  }
};
```

### 2.4 Correct Execution Ordering

**NEW flow achieving true overlap:**

```
Step iteration timeline:
─────────────────────────────────────────────────────────────────

Time  │ CPU (Scheduler)           │ NPU (Workers)
──────────────────────────────────────────────────────────────────
T0    │ wait_pending(batch_{N-1}) │ [NPU computing batch_{N-1}]
      │ → Phase A complete        │
      │                           │
T1    │ schedule_request(batch_N) │ [NPU computing batch_{N-1}]
      │ → Prepare batch_N         │
      │                           │
T2    │ step_async(batch_N)       │ [NPU computing batch_{N-1}]
      │ → Submit, don't wait      │ ←──── NPU starts batch_N
      │ → Fake tokens materialize │
      │                           │
T3    │ update_last_step(N-1)     │ [NPU computing batch_N] ←── OVERLAP!
      │ → Phase B: real tokens    │
      │                           │
T4    │ process_output(N-1)       │ [NPU computing batch_N] ←── OVERLAP!
      │ → Detokenize, response    │
      │                           │
T5    │ wait_pending(batch_N)     │ [NPU batch_N completes]
      │ → Phase A complete        │
─────────────────────────────────────────────────────────────────

Key insight: T3-T4 CPU work overlaps with T2-T5 NPU computation!
```

**vs current blocking flow:**

```
Current xLLM flow (NO overlap):
─────────────────────────────────────────────────────────────────

Time  │ CPU                       │ NPU
──────────────────────────────────────────────────────────────────
T0    │ schedule(batch_N)         │ [idle]
T1    │ step(batch_N)             │ [idle]
      │ → BLOCKS waiting for NPU  │
T2    │ [blocked]                 │ [NPU computing batch_N]
T3    │ [unblocked]               │ [idle]
T4    │ update_last(N-1)          │ [idle]
      │ → BLOCKS waiting          │
T5    │ [blocked]                 │ [NPU computing batch_{N-1}]
T6    │ [unblocked]               │ [idle]
T7    │ process_output            │ [idle]
─────────────────────────────────────────────────────────────────

Gap between T1-T3 and T5-T7: NPU idle, CPU blocked = NO OVERLAP
```

---

## 3. Implementation Plan

### Phase 1: Engine Split-Phase API (Week 1)

1. **Add new methods to `Engine` interface (`engine.h`):**
   - `step_async()`
   - `wait_pending_step()`
   - `has_pending_step()`
   - `get_pending_futures()`

2. **Implement in `LLMEngine`:**
   - Modify `step()` to optionally not block
   - Add `pending_futures_` storage
   - Implement Phase A fake token materialization

3. **Update `VLMEngine` similarly** (inherits from Engine)

### Phase 2: Batch Queue Mechanism (Week 2)

1. **Add `batch_queue_` to `ContinuousScheduler`:**
   - `PendingStep` struct
   - Configurable `batch_queue_size`

2. **Implement `step_with_batch_queue()`:**
   - Non-blocking submit
   - Conditional wait (only when queue full)

3. **Replace `step_with_schedule_overlap()` with new implementation**

### Phase 3: Integration & Testing (Week 3)

1. **Update configuration:**
   - Add `--batch-queue-size` flag
   - Deprecate old `--enable-schedule-overlap` (or map to new impl)

2. **Testing:**
   - Unit tests for split-phase API
   - Integration tests for batch queue
   - Performance benchmarks (compare old vs new)

### Phase 4: Advanced Features (Week 4)

1. **Multiple batch queue support** (like vLLM's PP batch queue)
   - For pipeline parallelism scenarios
   - `batch_queue_size > 2`

2. **Structured output handling:**
   - Deferred grammar bitmask (pending tokens)
   - Similar to vLLM's `pending_structured_output_tokens`

---

## 4. Expected Performance Improvement

### 4.1 Latency Analysis

**Current decode step latency breakdown:**

```
Component         │ Current │ Optimized │ Reduction
───────────────────────────────────────────────────
NPU compute       │ 10ms    │ 10ms      │ 0ms
CPU schedule      │ 1ms     │ 1ms       │ 0ms
CPU-NPU sync wait │ 3ms     │ ~0.2ms    │ 2.8ms (overlap)
CPU output process│ 2ms     │ ~0.2ms    │ 1.8ms (overlap)
───────────────────────────────────────────────────
Total per step    │ 16ms    │ ~11.4ms   │ 4.6ms (28% reduction)
```

### 4.2 Throughput Improvement

**Current throughput (TPOT constrained):**

```
Step time: 16ms → Throughput: 62.5 tokens/sec (per sequence)
```

**Optimized throughput:**

```
Step time: 11.4ms → Throughput: 87.7 tokens/sec (+40% improvement)
```

---

## 5. Key Code Changes Summary

### 5.1 Files to Modify

| File | Changes |
|------|---------|
| `engine.h` | Add split-phase API declarations |
| `llm_engine.h/cpp` | Implement `step_async()`, `wait_pending_step()`, pending storage |
| `continuous_scheduler.h/cpp` | Add batch_queue, implement new `step_with_batch_queue()` |
| `scheduler_base.h` | Add virtual methods for batch queue support |

### 5.2 Backward Compatibility

- Keep existing `step()` for synchronous mode
- New methods optional via feature flag
- Old `enable_schedule_overlap` maps to new implementation

---

## 6. Design Comparison: vLLM Batch Queue vs xLLM Split-Phase

### 6.1 vLLM Approach: Batch Queue (Multiple In-flight)

vLLM uses a **batch_queue** pattern that allows multiple pending batches simultaneously:

```python
# vLLM engine/core.py
self.batch_queue: deque[tuple[Future, SchedulerOutput, Future]] = deque(maxlen=batch_queue_size)

def step_with_batch_queue(self):
    # Schedule new batch WITHOUT blocking (if queue not full)
    if self.scheduler.has_requests() and len(batch_queue) < max_size:
        scheduler_output = self.scheduler.schedule()
        exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
        batch_queue.appendleft((future, scheduler_output, exec_future))
        if len(batch_queue) < batch_queue_size:
            return None, True  # Non-blocking return
    
    # Only block when queue is full
    future, scheduler_output = batch_queue.pop()
    model_output = future.result()
    self.scheduler.update_from_output(scheduler_output, model_output)
```

**Key characteristics:**
- Multiple batches can be pending (`batch_queue_size > 1`)
- Scheduler returns immediately after submit (doesn't wait)
- Only blocks when queue is full or no new requests
- Designed for pipeline parallelism (PP needs multiple in-flight batches)

### 6.2 xLLM Approach: Split-Phase (Single In-flight)

xLLM's design uses **split-phase** pattern with exactly one pending step:

```cpp
// xLLM proposed design
void step_with_schedule_overlap_v2(const absl::Duration& timeout) {
    // 1. Wait for previous step (Phase A: materialize)
    if (has_inflight_step_) {
        engine_->wait_pending_step(last_batch_);
        has_inflight_step_ = false;
    }
    
    // 2. Schedule current batch
    std::vector<Batch> batch = schedule_request(timeout);
    
    // 3. Submit without waiting (Phase 1: submit)
    if (!cur_batch_empty) {
        engine_->step_async(batch);
        has_inflight_step_ = true;
    }
    
    // 4. Commit previous batch (Phase B: commit) - overlaps with NPU!
    if (!is_first_step_ && last_batch_ready) {
        engine_->update_last_step_result(last_batch_);
        process_batch_output(true);
    }
}
```

**Key characteristics:**
- Only one pending step at any time (`has_inflight_step_`)
- Explicit Phase A (wait + materialize) before scheduling
- Explicit Phase B (commit) after submit - overlaps with next NPU
- No batch_queue needed, simpler state management

### 6.3 Trade-offs

| Aspect | vLLM Batch Queue | xLLM Split-Phase |
|--------|------------------|------------------|
| In-flight steps | Multiple (configurable) | Single (always 1) |
| When to block | Only when queue full | Every iteration (wait previous) |
| State management | Deque of futures + outputs | Simple `last_batch_` + flag |
| PP support | Required (multiple stages) | Not primary use case |
| Complexity | Higher (queue management) | Lower (single pending) |
| Memory overhead | Multiple SchedulerOutputs | Single batch |
| Optimal for | Pipeline Parallelism | Pure decode overlap |

### 6.4 Why xLLM Chose Split-Phase

xLLM's design choice is deliberate:

1. **No Pipeline Parallelism**: xLLM decode doesn't need multiple in-flight batches like PP
2. **Simpler Implementation**: Single pending step is easier to debug and maintain
3. **Preserves Existing Contract**: Worker already expects sequential step execution
4. **Two-phase Semantic Clarity**: Phase A (materialize) and Phase B (commit) are explicit

The key insight: **for pure decode overlap, one pending step is sufficient**. Batch queue's complexity is justified for PP but overkill for simple decode overlap.

### 6.5 Correctness Constraints

**vLLM batch_queue correctness:**
- Each SchedulerOutput must match its corresponding model_output
- Futures in queue must complete in FIFO order (vLLM assumes this)
- Queue size must match PP stages

**xLLM split-phase correctness:**
- Phase A must complete before `schedule_request()` (batch must be materialized)
- Phase B must happen after Phase A for the same batch
- Scheduler owns batch lifecycle; Engine only holds futures

---

## 7. References

- vLLM `engine/core.py`: `step_with_batch_queue()` implementation (lines 445-561)
- vLLM `scheduler/async_scheduler.py`: Output placeholder mechanism
- vLLM `scheduler/scheduler.py`: `num_output_placeholders` for async mode
- xLLM `docs/zh/design/async_schedule_overlap_split_phase_design.md`: Complete Chinese design
- xLLM `docs/zh/design/async_schedule_overlap_v2_design.md`: V2 design with time diagrams
- xLLM `llm_engine.cpp:1031-1032`: Blocking `collectAll(futures).get()`
- xLLM `continuous_scheduler.cpp:1085-1115`: Current overlap implementation