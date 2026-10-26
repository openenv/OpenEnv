"""
Morphological Erosion

Applies morphological erosion with a structuring element.
Erosion is a minimum filter within the structuring element window.

Optimization opportunities:
- Separable structuring elements (rectangle = row + column)
- Van Herk/Gil-Werman algorithm for O(1) per pixel
- Shared memory for window data
- Parallel prefix operations for running min
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Morphological erosion operation.

    For binary images: erodes (shrinks) foreground regions.
    For grayscale: minimum filter within structuring element.
    """

    def __init__(self, kernel_size: int = 5):
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply morphological erosion.

        Args:
            image: (H, W) image (binary or grayscale)

        Returns:
            eroded: (H, W) eroded image
        """
        # Use max pooling with negative values = min pooling
        x = image.unsqueeze(0).unsqueeze(0)

        # Erosion = min filter = -max(-x)
        eroded = -F.max_pool2d(
            -x, kernel_size=self.kernel_size, stride=1, padding=self.padding
        )

        return eroded.squeeze(0).squeeze(0)


# Problem configuration
image_height = 1920
image_width = 1080


def get_inputs():
    # Binary image (for morphology)
    image = (torch.rand(image_height, image_width) > 0.5).float()
    return [image]


def get_init_inputs():
    return [5]  # kernel_size
