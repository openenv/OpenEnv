"""
Bilinear Resampling (Image Resize)

Resamples an image to a different resolution using bilinear interpolation.
Core operation in image processing, rendering, and neural networks.

Optimization opportunities:
- Texture memory for hardware interpolation
- Separable implementation (horizontal + vertical)
- Vectorized coefficient computation
- Coalesced output writes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Bilinear image resampling.
    """

    def __init__(self, output_height: int = 1080, output_width: int = 1920):
        super(Model, self).__init__()
        self.output_height = output_height
        self.output_width = output_width

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Resample image to target size.

        Args:
            image: (H, W) or (C, H, W) input image

        Returns:
            resampled: (output_height, output_width) or (C, output_height, output_width)
        """
        original_shape = image.shape

        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif image.dim() == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)

        # Use grid_sample for bilinear interpolation
        resampled = F.interpolate(
            image,
            size=(self.output_height, self.output_width),
            mode="bilinear",
            align_corners=False,
        )

        # Restore dimensions
        if len(original_shape) == 2:
            resampled = resampled.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            resampled = resampled.squeeze(0)

        return resampled


# Problem configuration - downscale
input_height = 3840
input_width = 2160


def get_inputs():
    # 4K image
    image = torch.rand(input_height, input_width)
    return [image]


def get_init_inputs():
    return [1080, 1920]  # output_height, output_width (1080p)
