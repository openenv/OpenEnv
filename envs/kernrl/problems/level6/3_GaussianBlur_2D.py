"""
2D Gaussian Blur

Applies a Gaussian blur filter to a 2D image.
This is a separable filter, commonly implemented as two 1D passes.

Optimization opportunities:
- Separable implementation (row pass + column pass)
- Shared memory for input caching
- Texture memory for interpolation
- Row-wise processing for coalesced access
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Applies Gaussian blur to a 2D image.

    Uses a configurable kernel size and sigma.
    """

    def __init__(self, kernel_size: int = 15, sigma: float = 3.0):
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2

        # Create Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gaussian_1d = torch.exp(-(x**2) / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()

        # 2D kernel as outer product
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        gaussian_2d = gaussian_2d / gaussian_2d.sum()

        # Register as buffer (moves with model to device)
        self.register_buffer("kernel", gaussian_2d.unsqueeze(0).unsqueeze(0))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur.

        Args:
            image: (H, W) or (C, H, W) or (B, C, H, W) image tensor

        Returns:
            blurred: same shape as input
        """
        # Handle different input shapes
        original_shape = image.shape
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif image.dim() == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)

        B, C, H, W = image.shape

        # Apply same kernel to each channel
        # Expand kernel for all channels
        kernel = self.kernel.repeat(C, 1, 1, 1)

        # Apply convolution (groups=C for depthwise)
        blurred = F.conv2d(image, kernel, padding=self.padding, groups=C)

        # Restore original shape
        if len(original_shape) == 2:
            blurred = blurred.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            blurred = blurred.squeeze(0)

        return blurred


# Problem configuration
image_height = 1920
image_width = 1080


def get_inputs():
    # Grayscale image
    image = torch.rand(image_height, image_width)
    return [image]


def get_init_inputs():
    return [15, 3.0]  # kernel_size, sigma
