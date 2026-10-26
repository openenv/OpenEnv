"""
2D Fast Fourier Transform

Computes 2D DFT, commonly used in image processing for:
- Frequency domain filtering
- Convolution via multiplication
- Pattern detection

Optimization opportunities:
- Row-column decomposition
- Shared memory for partial transforms
- Batched 1D FFTs
- Tiled computation for large images
"""

import torch
import torch.fft
import torch.nn as nn


class Model(nn.Module):
    """
    2D Fast Fourier Transform.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D FFT.

        Args:
            image: (H, W) real or complex 2D array

        Returns:
            spectrum: (H, W) complex 2D frequency components
        """
        return torch.fft.fft2(image)


# Problem configuration
image_height = 2048
image_width = 2048


def get_inputs():
    image = torch.randn(image_height, image_width)
    return [image]


def get_init_inputs():
    return []
