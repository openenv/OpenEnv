"""
Color Space Conversion - RGB to YUV

Converts RGB image to YUV color space.
This is a pixel-wise matrix multiplication, highly parallelizable.

Y =  0.299*R + 0.587*G + 0.114*B
U = -0.147*R - 0.289*G + 0.436*B
V =  0.615*R - 0.515*G - 0.100*B

Optimization opportunities:
- Vectorized pixel processing
- Fused multiply-add (FMA) instructions
- Coalesced memory access patterns
- In-place conversion if possible
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    RGB to YUV color space conversion.
    """

    def __init__(self):
        super(Model, self).__init__()

        # Conversion matrix
        rgb_to_yuv = torch.tensor(
            [[0.299, 0.587, 0.114], [-0.147, -0.289, 0.436], [0.615, -0.515, -0.100]],
            dtype=torch.float32,
        )

        self.register_buffer("conversion_matrix", rgb_to_yuv)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB to YUV.

        Args:
            rgb: (H, W, 3) or (B, H, W, 3) RGB image in [0, 1]

        Returns:
            yuv: same shape, YUV values
        """
        # Matrix multiply on last dimension
        yuv = torch.matmul(rgb, self.conversion_matrix.T)
        return yuv


# Problem configuration
image_height = 1920
image_width = 1080


def get_inputs():
    # RGB image
    rgb = torch.rand(image_height, image_width, 3)
    return [rgb]


def get_init_inputs():
    return []
