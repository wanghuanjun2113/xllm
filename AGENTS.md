# xLLM Coding Agent Instructions

## Build Commands

```bash
# Build (device required)
python setup.py build --device <npu|mlu|cuda|ilu|musa> --arch <x86|arm>

# Build wheel
python setup.py bdist_wheel --device <device>

# Skip tests/export during build
SKIP_TEST=1 python setup.py build --device <device>
SKIP_EXPORT=1 python setup.py build --device <device>
```

CUDA builds require: `export TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0"`

## Test Commands

```bash
# Run all tests (via ctest, parallel except sequential tests)
python setup.py test --device <device>

# Run single test target
python setup.py test --test-name <test_target> --device <device>
# Example: python setup.py test --test-name platform_vmm_test --device npu
```

Sequential tests (fork conflicts): `ReduceScatterMultiDeviceTest`, `DeepEPMultiDeviceTest`, `AttentionMultiDeviceTest`, `FusedMoEAll2AllMultiDeviceTest`

## Format Check

```bash
# Pre-commit hook (clang-format 20.1.6)
pre-commit run --all-files

# CI format check uses git-clang-format against base commit
```

## Hardware Backends

NPU (Ascend), MLU, ILU, MUSA, CUDA. Docker build scripts in `cibuild/build_<device>.sh`.

## Code Style

**Required reading before editing `xllm/`:** [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md)

Key rules:
- Google-style clang-format (2-char indent, 80 column limit)
- `#pragma once` headers, no relative includes
- Copyright header required on all new files
- `torch::` namespace, `CHECK`/`LOG(FATAL)` over `TORCH_CHECK`/exceptions
- `snake_case_` member variables, `PascalCase` classes
- Fixed-width integers (`int32_t`, `int64_t`), `static_cast` only

## Code Review

For review tasks: read [code-review/SKILL.md](.agents/skills/code-review/SKILL.md) then apply custom-code-style.md.

## Git Workflow

Commit format: `<type>: <subject>` (e.g., `feat: add rope kernel for npu.`)

Types: `feat`, `bugfix`, `docs`, `test`, `refactor`, `chore`, `perf`, `model`, `build`, `release`

Use `bugfix:` (not `fix:`). See [commit-format.md](.agents/skills/git-workflow/references/commit-format.md).

## TileLang Ascend Kernels

For NPU kernel work: read [tilelang-ascend-kernel/SKILL.md](.agents/skills/tilelang-ascend-kernel/SKILL.md).

Key: run TileLang from repo root with `export TL_ROOT=$PWD/third_party/tilelang-ascend`.

## Directory Structure

```
xllm/
├── core/           # engine: framework/, runtime/, scheduler/, kernels/, layers/, platform/
├── api_service/    # OpenAI/Anthropic service impl
├── c_api/          # C API
├── cc_api/         # C++ API
├── models/         # model definitions
├── pybind/         # Python bindings
├── server/         # xLLM server entry
├── parser/         # reasoning parser
├── processors/     # VLM preprocessing
├── function_call/  # tool call parser
└── proto/          # gRPC/protobuf
```