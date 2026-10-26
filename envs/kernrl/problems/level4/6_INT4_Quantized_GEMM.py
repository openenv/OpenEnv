import torch
import torch.nn as nn

# INT4 Weight-Only Quantized GEMM with Symmetric Quantization
# Reference: GPTQ (https://arxiv.org/abs/2210.17323)
# Used in: llama.cpp, exllama, vLLM Marlin, TensorRT-LLM
#
# Weight-only quantization stores weights in INT4 while keeping activations in FP16.
# This reduces memory bandwidth for LLM inference where weights dominate memory.
#
# Symmetric quantization (GPTQ default):
# - INT4 weights in range [0, 15], with 8 as the zero-point (center)
# - No per-group zero-points stored - implicit zero = 8
# - Dequantization: W_dequant = scale * (W_q - 8)
#
# Key concepts:
# - INT4 weights: 4-bit integers packed 2 per byte (low nibble first)
# - Group-wise quantization: Each group of G weights shares a scale
# - Packing format: byte = (high_nibble << 4) | low_nibble
#
# This problem tests:
# 1. INT4 unpacking (2 weights per byte, bit manipulation)
# 2. Group-wise dequantization with symmetric zero-point
# 3. Fused unpack-dequant-GEMM to avoid memory round-trip


class Model(nn.Module):
    """
    INT4 Weight-Only Quantized Linear Layer with Symmetric Quantization.

    Weights are stored as packed INT4 (2 weights per uint8 byte).
    Each group of G consecutive weights along K dimension shares a scale.
    Zero-point is implicitly 8 (center of [0, 15] range) for all groups.

    Key optimization targets:
    1. Efficient INT4 unpacking (bit manipulation in registers)
    2. Fused dequantization within GEMM (avoid memory write of dequantized weights)
    3. Tensor core utilization with on-the-fly dequant
    4. Optimal memory access pattern for packed weights + scales

    The naive implementation:
    - Unpacks INT4 to INT32
    - Applies group-wise scale with implicit zero-point of 8
    - Performs FP16 matmul

    An optimized kernel should fuse unpacking + dequant + GEMM.
    """

    def __init__(self, K: int, N: int, group_size: int = 128):
        super().__init__()
        self.K = K
        self.N = N
        self.group_size = group_size
        self.num_groups = K // group_size

        assert K % group_size == 0, "K must be divisible by group_size"
        assert K % 2 == 0, "K must be even for INT4 packing"

        # Packed INT4 weights: 2 weights per byte, stored as uint8
        # Shape: (N, K//2) - each byte holds 2 INT4 values
        # Packing: byte = (high_nibble << 4) | low_nibble
        self.register_buffer(
            "weight_packed", torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        )

        # Per-group scales: (N, num_groups) in FP16
        # Scale maps the INT4 range to the original weight range
        self.register_buffer(
            "scales", torch.randn(N, self.num_groups, dtype=torch.float16).abs() * 0.1
        )

    def unpack_int4(self, packed: torch.Tensor) -> torch.Tensor:
        """
        Unpack INT4 weights from packed uint8 format.

        Input: (N, K//2) uint8 where each byte holds 2 INT4 values
        Output: (N, K) int32 with values in [0, 15]

        Packing format: byte = (high_nibble << 4) | low_nibble
        low_nibble (bits 0-3) is the first weight in the pair
        high_nibble (bits 4-7) is the second weight in the pair
        """
        # Extract low nibble (first weight in pair)
        low = (packed & 0x0F).to(torch.int32)
        # Extract high nibble (second weight in pair)
        high = ((packed >> 4) & 0x0F).to(torch.int32)
        # Interleave: [low0, high0, low1, high1, ...]
        unpacked = torch.stack([low, high], dim=-1).view(packed.shape[0], -1)
        return unpacked

    def dequantize_weights(self) -> torch.Tensor:
        """
        Dequantize INT4 weights to FP16 using symmetric quantization.

        Symmetric quantization formula:
            W_dequant[n, k] = scales[n, g] * (W_q[n, k] - 8)

        where g = k // group_size and 8 is the implicit zero-point (center of [0,15])
        """
        # Unpack INT4 to int32: (N, K)
        w_int = self.unpack_int4(self.weight_packed)

        # Expand scales to match weight dimensions
        # scales: (N, num_groups) -> (N, K)
        scales_expanded = self.scales.repeat_interleave(self.group_size, dim=1)

        # Symmetric dequantization: scale * (w_int - 8)
        # 8 is the center of [0, 15] range, implicit zero-point
        w_dequant = scales_expanded * (w_int.to(torch.float16) - 8.0)

        return w_dequant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        INT4 quantized linear: Y = X @ W_dequant.T

        Input x: (batch, seq_len, K) in FP16
        Output: (batch, seq_len, N) in FP16

        INEFFICIENT: This naive implementation:
        1. Unpacks all INT4 weights to FP16
        2. Dequantizes entire weight matrix
        3. Performs standard matmul

        A fused kernel would do unpacking + dequant on-the-fly during GEMM,
        reading packed weights once and never materializing the full FP16 matrix.
        """
        batch_size, seq_len, _ = x.shape

        # INEFFICIENT: Full dequantization before matmul
        # This writes K*N FP16 values to memory unnecessarily
        w_dequant = self.dequantize_weights()  # (N, K)

        # Reshape for matmul
        x_2d = x.view(-1, self.K)  # (batch*seq, K)

        # Standard matmul with dequantized weights
        out = torch.matmul(x_2d, w_dequant.T)  # (batch*seq, N)

        return out.view(batch_size, seq_len, self.N)


# Configuration sized for LLM inference workloads
batch_size = 4
seq_len = 2048
K = 4096  # Input features (hidden dim)
N = 11008  # Output features (MLP intermediate, typical for 7B models)
group_size = 128  # Standard group size for GPTQ


def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16)]


def get_init_inputs():
    return [K, N, group_size]
