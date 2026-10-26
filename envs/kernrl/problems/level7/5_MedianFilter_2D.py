"""
2D Median Filter

Non-linear filter that replaces each pixel with the median of its neighborhood.
Excellent for removing salt-and-pepper noise while preserving edges.

Challenge: Finding median requires sorting - not a simple convolution.

Optimization opportunities:
- Histogram-based median (for byte images)
- Partial sort using selection networks
- Running median algorithms
- Separable approximations
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    2D median filter for noise removal.
    """

    def __init__(self, kernel_size: int = 5):
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.radius = kernel_size // 2

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply median filter.

        Args:
            image: (H, W) input image

        Returns:
            filtered: (H, W) median filtered image
        """
        H, W = image.shape
        radius = self.radius
        ks = self.kernel_size

        # Pad image
        padded = F.pad(image, (radius, radius, radius, radius), mode="reflect")

        # Unfold to get all windows
        # Shape: (H, W, ks, ks)
        windows = padded.unfold(0, ks, 1).unfold(1, ks, 1)

        # Flatten windows and compute median
        # Shape: (H, W, ks*ks)
        windows_flat = windows.reshape(H, W, -1)

        # Median along last dimension
        filtered = windows_flat.median(dim=-1)[0]

        return filtered


# Need F for padding
import torch.nn.functional as F

# Problem configuration
image_height = 512
image_width = 512


def get_inputs():
    # Image with salt-and-pepper noise
    image = torch.rand(image_height, image_width)
    # Add noise
    noise_mask = torch.rand(image_height, image_width)
    image[noise_mask < 0.05] = 0.0  # Salt
    image[noise_mask > 0.95] = 1.0  # Pepper
    return [image]


def get_init_inputs():
    return [5]  # kernel_size
