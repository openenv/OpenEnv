---
title: kernrl Environment Server
emoji: ⚡
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# kernrl

RL environment for GPU kernel optimization. Train LLM agents to write fast CUDA/Triton kernels.

## Overview

Agents receive a PyTorch reference implementation and must write an optimized GPU kernel that:
1. Produces the same output (within tolerance)
2. Runs faster than the baseline

Each submission is evaluated with:
- Compilation checking
- Correctness verification against reference
- Benchmark timing for speedup measurement
- NSight Systems profiling (optional)
- NSight Compute profiling (optional)

## Installation

```bash
cd envs/kernrl
pip install -e .
```

Requires: NVIDIA GPU with CUDA toolkit, PyTorch, Triton

## Quick Start

```python
from kernrl import KernelAction, kernrl_env

# Connect to server
env = kernrl_env(base_url="http://localhost:8000")

# Start episode
result = env.reset(problem_id="L1_23_Softmax")
print(result.observation.problem_description)

# Submit a kernel
action = KernelAction(code='''
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start = row_idx * n_cols
    row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(output_ptr + row_start + col_offsets, softmax_output, mask=mask)

class Model(torch.nn.Module):
    def forward(self, x):
        n_rows, n_cols = x.shape
        output = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        softmax_kernel[(n_rows,)](x, output, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        return output
''')

result = env.step(action)
print(f"Speedup: {result.observation.speedup}x")
print(f"Correct: {result.observation.correctness_pass}")
```

## Running the Server

```bash
# Development
uvicorn kernrl.server.app:app --reload --host 0.0.0.0 --port 8000

# Docker (GPU required)
cd envs/kernrl
docker build -t kernrl -f server/Dockerfile .
docker run --gpus all -p 8000:8000 kernrl
```

## Problem Levels

| Level | Name | Count | Description |
|-------|------|-------|-------------|
| 1 | Simple Operators | 15 | matmul, softmax, conv, norms |
| 2 | Fused Operations | 15 | matmul+activation chains |
| 3 | Single Blocks | 3 | attention, transformer block |
| 4 | Novel Layers | 8 | MLA, MoE, GQA, FP8, INT4 |
| 5 | Scientific Computing | 8 | N-body, stencil, SpMV |
| 6 | Graphics | 8 | ray tracing, histogram, blur |
| 7 | Signal Processing | 8 | FFT, convolution, median filter |
| 8 | Video Processing | 8 | motion estimation, optical flow |
| 9 | Parallel Primitives | 8 | scan, reduction, radix sort |
| 10 | Cryptography | 8 | SHA-256, AES, ChaCha20 |

**Total: 89 problems**

## Reward Structure

Rewards are designed so that **only speedup > 1.0x baseline produces positive reward**.
Compilation and correctness alone do not give positive reward - they are necessary but not sufficient.

| Condition | Reward | Description |
|-----------|--------|-------------|
| Compilation failure | -0.5 | Penalty for code that doesn't compile |
| Correctness failure | -0.25 | Penalty for incorrect output |
| Correct but slower | (speedup - 1.0) * 0.5 | Small negative for being slower than baseline |
| Correct and faster | min(speedup - 1.0, 2.0) | Positive, capped at 2.0 |

**Examples:**
- Compile fail: reward = -0.5
- Compiles, wrong output: reward = -0.25
- Compiles, correct, 0.8x speed: reward = -0.1
- Compiles, correct, 1.0x speed: reward = 0.0
- Compiles, correct, 1.5x speed: reward = 0.5
- Compiles, correct, 3.0x speed: reward = 2.0 (capped)

## Security Considerations

**Warning:** This environment executes user-submitted kernel code with full Python/CUDA privileges.
While Docker provides container isolation, there is no sandboxing within the container for:
- Filesystem access
- Network requests
- Resource consumption (GPU memory, CPU)
- Module imports

This is acceptable for trusted research environments but should be documented as a security consideration.
For production deployments, consider additional isolation measures.

## License

BSD-3-Clause (following OpenEnv licensing)
