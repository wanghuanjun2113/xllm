# xLLM Async Scheduling Improvement: Learning from vLLM Patterns

## Executive Summary

This document analyzes vLLM's async scheduling patterns and proposes specific improvements for xLLM to eliminate CPU-GPU synchronization bottlenecks.

## Problem Statement

### Current xLLM Blocking Points

```
Worker (NPU):
  sampler_->forward(logits) → sample_output.next_tokens (NPU tensor)
  device_.synchronize_default_stream()  ← BLOCKING POINT #1 (line 221)
  return ForwardOutput(sample_output.next_tokens=NPU tensor)

Engine:
  batch.process_sample_output(sample_output)
    build_token(index, next_tokens, ...)
      token_ids[index].item<int64_t>()  ← BLOCKING POINT #2 (line 798)
      ↑ Implicit D2H copy, waits for NPU→CPU transfer
```

### Root Cause

1. **Worker sync**: `synchronize_default_stream()` waits for all kernels
2. **Engine sync**: `.item<int64_t>()` requires CPU values but tensor is on NPU

## vLLM Patterns Analysis

### Pattern 1: Pinned Memory + Event Sync (Synchronous Mode)

**File**: `vllm/vllm/v1/worker/gpu_model_runner.py:6893-6906`

```python
def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
    # Pre-allocated pinned CPU buffer
    pinned = self.sampled_token_ids_pinned_cpu[:sampled_token_ids.shape[0]]
    
    # Async copy from GPU to pinned memory
    pinned.copy_(sampled_token_ids, non_blocking=True)
    
    # Use CUDA event for synchronization (not default stream)
    self.transfer_event.record()
    self.transfer_event.synchronize()
    
    # Convert to Python list (data already in CPU)
    return pinned.tolist()
```

**Key Insight**: Uses pinned memory + CUDA event instead of implicit sync via `.item()`

### Pattern 2: Async D2H Copy on Separate Stream (Async Mode)

**File**: `vllm/vllm/v1/worker/gpu_model_runner.py:230-292`

```python
class AsyncGPUModelRunnerOutput:
    def __init__(self, sampled_token_ids, ...):
        with torch.cuda.stream(async_output_copy_stream):
            # Async copy to CPU
            self.sampled_token_ids_cpu = sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self.async_copy_ready_event.record()
        # Return immediately, don't wait!
    
    def get_output(self) -> ModelRunnerOutput:
        # Only sync when output is needed
        self.async_copy_ready_event.synchronize()
        return ModelRunnerOutput(
            sampled_token_ids=self.sampled_token_ids_cpu.tolist()
        )
```

**Key Insight**: Separate stream for D2H, defer sync until Engine needs values

### Pattern 3: Return Python List to Scheduler

**File**: `vllm/vllm/v1/worker/gpu_model_runner.py:3414-3428`

```python
def sample_tokens(...):
    if self.use_async_scheduling:
        # Async path: return GPU tensor + event
        return AsyncGPUModelRunnerOutput(
            sampled_token_ids=sampled_token_ids,  # GPU tensor
            ...
        )
    else:
        # Sync path: return CPU list
        return ModelRunnerOutput(
            sampled_token_ids=self._to_list(sampled_token_ids)  # Python list
        )
```

**Key Insight**: Worker converts tensor to list before returning to Engine

### Pattern 4: Scheduler Uses List Directly

**File**: `vllm/vllm/v1/engine/scheduler.py:1299-1361`

```python
def update_from_output(self, scheduler_output, model_runner_output):
    # sampled_token_ids is list[list[int]], not tensor!
    sampled_token_ids = model_runner_output.sampled_token_ids
    
    for req_id, ... in num_scheduled_tokens.items():
        req_index = model_runner_output.req_id_to_index[req_id]
        
        # Direct indexing, no .item() needed!
        generated_token_ids = sampled_token_ids[req_index]
        
        # Append to Python list directly
        request.append_output_token_ids(generated_token_ids)
```

**Key Insight**: No tensor operations in Scheduler, all data is Python types

## Proposed Improvements for xLLM

### Improvement 1: Introduce AsyncForwardOutput

**Goal**: Add async D2H copy capability to Worker output

**File**: `xllm/xllm/core/runtime/forward_params.h`

```cpp
// New struct for async D2H copy
struct AsyncSampleOutput {
  torch::Tensor next_tokens_gpu;           // GPU tensor (original)
  torch::Tensor next_tokens_cpu;           // Pinned CPU tensor (copy target)
  at::cuda::CUDAEvent copy_ready_event;    // Event for sync
  bool copy_started = false;
  
  // Blocking call to get CPU values
  std::vector<int64_t> get_next_tokens_cpu(int64_t index) {
    if (!copy_started) {
      start_async_copy();
    }
    copy_ready_event.block();
    return next_tokens_cpu[index].tolist();
  }
  
  void start_async_copy() {
    next_tokens_cpu.copy_(next_tokens_gpu, /*non_blocking=*/true);
    copy_ready_event.record();
    copy_started = true;
  }
};

struct ForwardOutput {
  // Existing fields...
  
  // New field for async mode
  std::optional<AsyncSampleOutput> async_sample_output;
};
```

### Improvement 2: Add Pinned Memory Pool to Worker

**File**: `xllm/xllm/core/runtime/llm_worker_impl.cpp`

```cpp
class LLMWorkerImpl {
  // Pre-allocated pinned CPU tensors for async D2H
  torch::Tensor sampled_tokens_pinned_cpu_;
  at::cuda::CUDAEvent transfer_event_;
  at::cuda::CUDAStream transfer_stream_;
  
  void init_async_resources() {
    sampled_tokens_pinned_cpu_ = torch::empty(
      {max_batch_size_, 1},
      torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(torch::kCPU)
        .pinned_memory(true)
    );
    transfer_event_ = at::cuda::CUDAEvent();
    transfer_stream_ = at::cuda::CUDAStream();
  }
};
```

### Improvement 3: Modify Worker::step_internal for Async Mode

**File**: `xllm/xllm/core/runtime/llm_worker_impl.cpp:97-241`

```cpp
std::optional<ForwardOutput> LLMWorkerImpl::step_internal(
    const ForwardInput& input) {
  // ... existing code ...
  
  ForwardOutput output;
  
  if (sampling_params.selected_token_idxes.defined()) {
    auto sample_output = sampler_->forward(logits, sampling_params);
    
    if (enable_async_d2h_) {
      // Async path: start D2H copy, don't sync
      output.async_sample_output = AsyncSampleOutput{
        .next_tokens_gpu = sample_output.next_tokens,
        .next_tokens_cpu = sampled_tokens_pinned_cpu_,
        .copy_ready_event = transfer_event_,
      };
      
      // Start async copy on separate stream
      {
        at::cuda::CUDAStreamGuard guard(transfer_stream_);
        sampled_tokens_pinned_cpu_.copy_(
          sample_output.next_tokens, /*non_blocking=*/true
        );
        transfer_event_.record(transfer_stream_);
      }
      
      // Don't sync! Return immediately
    } else {
      // Sync path: convert to CPU list in Worker
      output.sample_output = sample_output;
      
      // Do NOT call synchronize_default_stream() here!
      // Let Engine sync when needed
    }
  }
  
  // Remove this sync in async mode:
  // auto ret = device_.synchronize_default_stream();
  
  return output;
}
```

### Improvement 4: Add to_list Helper (Similar to vLLM)

**File**: `xllm/xllm/core/runtime/llm_worker_impl.cpp`

```cpp
std::vector<std::vector<int64_t>> LLMWorkerImpl::to_list(
    const torch::Tensor& sampled_token_ids) {
  // Pre-allocated pinned memory
  auto pinned = sampled_tokens_pinned_cpu_(
    torch::indexing::Slice(0, sampled_token_ids.size(0))
  );
  
  // Async copy
  pinned.copy_(sampled_token_ids, /*non_blocking=*/true);
  
  // Event sync (not default stream sync)
  transfer_event_.record();
  transfer_event_.block();
  
  // Convert to vector
  return pinned.toVector<std::vector<int64_t>>();
}
```

### Improvement 5: Modify Engine to Handle Async Output

**File**: `xllm/xllm/core/distributed_runtime/llm_engine.cpp`

```cpp
void LLMEngine::process_forward_output(ForwardOutput& output) {
  if (output.async_sample_output.has_value()) {
    // Async path: sync only when we need values
    auto& async_output = output.async_sample_output.value();
    
    // Sync and convert
    async_output.copy_ready_event.block();
    
    // Now safe to access CPU tensor
    auto token_ids_cpu = async_output.next_tokens_cpu;
    
    // Process without .item() calls
    for (int64_t i = 0; i < token_ids_cpu.size(0); ++i) {
      auto tokens = token_ids_cpu[i].toVector<int64_t>();
      // Update sequences...
    }
  } else {
    // Sync path: output is already CPU-side or needs sync
    // ... existing code ...
  }
}
```

### Improvement 6: Add AsyncScheduling Flag

**File**: `xllm/xllm/core/runtime/runtime_configs.h`

```cpp
struct RuntimeConfigs {
  // Existing fields...
  
  bool enable_async_d2h = false;  // Enable async D2H copy for sampling
  
  // For split-phase scheduling
  bool enable_schedule_overlap = false;  // Already exists
};
```

## Implementation Phases

### Phase 1: Infrastructure (Low Risk)

1. Add pinned memory pool to LLMWorkerImpl
2. Add CUDA/NPU event and stream for async transfer
3. Add `enable_async_d2h` config flag

**Files Modified**:
- `xllm/xllm/core/runtime/llm_worker_impl.h`
- `xllm/xllm/core/runtime/llm_worker_impl.cpp`
- `xllm/xllm/core/runtime/runtime_configs.h`

### Phase 2: Sync Mode Optimization (Medium Risk)

1. Implement `to_list()` helper using pinned memory + event sync
2. Replace `synchronize_default_stream()` with event-based sync
3. Return Python list from Worker in sync mode

**Files Modified**:
- `xllm/xllm/core/runtime/llm_worker_impl.cpp`
- `xllm/xllm/core/runtime/forward_params.h`

### Phase 3: Async Mode (High Risk)

1. Introduce `AsyncSampleOutput` struct
2. Implement async D2H copy on separate stream
3. Modify Engine to handle async output
4. Integrate with existing split-phase scheduling

**Files Modified**:
- `xllm/xllm/core/runtime/forward_params.h`
- `xllm/xllm/core/runtime/llm_worker_impl.cpp`
- `xllm/xllm/core/distributed_runtime/llm_engine.cpp`
- `xllm/xllm/core/framework/batch/batch.cpp`

### Phase 4: NPU-Specific Optimization

1. Use NPU-specific pinned memory APIs (`aclrtMallocHost`)
2. Use NPU events (`aclrtEvent`)
3. Use NPU streams (`aclrtStream`)

**Files Modified**:
- `xllm/xllm/core/runtime/llm_worker_impl.cpp`
- `xllm/xllm/core/platform/npu/` (new files)

## Performance Expectations

### Before (Current xLLM)

```
Timeline:
  [Forward][Sample][Synchronize][D2H Copy][Schedule][Forward]
           |       |            |          |         |
           |       |            |          |         └── Next iteration
           |       |            |          └── sync point #2 (.item())
           |       |            └── sync point #1 (synchronize_default_stream)
           |       └── Blocking sync
           └── Needs sync to access values
```

### After (Async Mode)

```
Timeline:
  [Forward][Sample][Async D2H Start][Schedule (cached prev)][Forward]
           |       |                 |                       |
           |       |                 |                       └── Next iteration
           |       |                 |                           uses prev tokens
           |       |                 └── Overlaps with scheduling
           |       └── Non-blocking
           └── Use prev iteration's tokens (already in CPU)
```

### Expected Improvement

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Worker sync time | ~5-10ms | 0ms | 100% reduction |
| Engine sync time | ~1-2ms | 0ms | 100% reduction |
| Overlap potential | None | Full | Decode+Scheduling overlap |
| Throughput (decode) | Baseline | +10-20% | Depends on model size |

## Risks and Mitigations

### Risk 1: Pinned Memory Overhead

**Mitigation**: Pre-allocate fixed-size pool, reuse across iterations

### Risk 2: Event Sync Still Blocks

**Mitigation**: Use split-phase scheduling to hide sync latency

### Risk 3: NPU API Differences

**Mitigation**: Abstract behind Device interface, use CUDA fallback for testing

### Risk 4: Regression in Non-Async Mode

**Mitigation**: Feature flag `enable_async_d2h`, default off

## Testing Strategy

### Unit Tests

1. Test pinned memory allocation
2. Test async D2H copy
3. Test event synchronization

### Integration Tests

1. Test with `enable_async_d2h=false` (existing behavior)
2. Test with `enable_async_d2h=true` (new behavior)
3. Test with `enable_schedule_overlap=true` (split-phase)

### Performance Tests

1. Measure sync time before/after
2. Measure throughput improvement
3. Measure latency distribution

## References

- vLLM GPU Model Runner: `vllm/vllm/v1/worker/gpu_model_runner.py`
- vLLM Async Output: `vllm/vllm/v1/worker/gpu_model_runner.py:230-292`
- vLLM Scheduler: `vllm/vllm/v1/engine/scheduler.py:1299-1361`
- xLLM Worker: `xllm/xllm/core/runtime/llm_worker_impl.cpp`
- xLLM Token Builder: `xllm/xllm/core/runtime/params_utils.cpp:793-810`