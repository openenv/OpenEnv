"""
Short-Time Fourier Transform (STFT)

Computes the STFT of a signal using sliding window analysis.
Fundamental for audio processing, speech recognition, and spectrograms.

STFT(t, f) = sum_n x[n] * w[n-t] * exp(-j*2*pi*f*n/N)

Optimization opportunities:
- Batched FFTs for all windows
- Shared memory for window overlap
- Fused windowing + FFT
- Streaming for long signals
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Short-Time Fourier Transform.
    """

    def __init__(self, n_fft: int = 1024, hop_length: int = 256, window: str = "hann"):
        super(Model, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Create window function
        if window == "hann":
            w = torch.hann_window(n_fft)
        elif window == "hamming":
            w = torch.hamming_window(n_fft)
        else:
            w = torch.ones(n_fft)

        self.register_buffer("window", w)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT.

        Args:
            signal: (N,) time-domain signal

        Returns:
            stft: (num_frames, n_fft//2+1) complex spectrogram
        """
        return torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )


# Problem configuration
signal_length = 16000 * 10  # 10 seconds at 16kHz


def get_inputs():
    # Audio signal
    signal = torch.randn(signal_length)
    return [signal]


def get_init_inputs():
    return [1024, 256, "hann"]  # n_fft, hop_length, window
