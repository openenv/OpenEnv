"""
1D Fast Fourier Transform (FFT)

Computes the Discrete Fourier Transform using the Cooley-Tukey algorithm.
Fundamental operation in signal processing, audio analysis, and convolution.

Optimization opportunities:
- Radix-2/4/8 algorithms
- Shared memory for butterfly operations
- Bank-conflict-free shared memory access
- Warp-synchronous programming
- Stockham auto-sort algorithm
"""

import torch
import torch.fft
import torch.nn as nn


class Model(nn.Module):
    """
    1D Fast Fourier Transform.

    Computes DFT of complex or real signals.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Compute 1D FFT.

        Args:
            signal: (N,) or (B, N) real or complex signal

        Returns:
            spectrum: (N,) or (B, N) complex frequency components
        """
        return torch.fft.fft(signal)


# Problem configuration
signal_length = 1024 * 1024  # 1M samples


def get_inputs():
    # Real signal
    signal = torch.randn(signal_length)
    return [signal]


def get_init_inputs():
    return []
