"""
Chroma Upsampling (YUV 4:2:0 to 4:4:4)

Upsamples subsampled chroma channels to full resolution.
Essential for video decoding and color processing.

In 4:2:0 format, U and V channels are half resolution in both dimensions.
This kernel upsamples them to match Y channel resolution.

Optimization opportunities:
- Separable bilinear/bicubic interpolation
- Texture memory for source
- Vectorized output writes
- Fused luma/chroma processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Upsamples chroma from 4:2:0 to 4:4:4.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self, y_full: torch.Tensor, u_half: torch.Tensor, v_half: torch.Tensor
    ) -> tuple:
        """
        Upsample chroma channels.

        Args:
            y_full: (H, W) full resolution luma
            u_half: (H//2, W//2) half resolution U chroma
            v_half: (H//2, W//2) half resolution V chroma

        Returns:
            y: (H, W) unchanged luma
            u_full: (H, W) upsampled U
            v_full: (H, W) upsampled V
        """
        H, W = y_full.shape

        # Upsample U and V using bilinear interpolation
        u_4d = u_half.unsqueeze(0).unsqueeze(0)
        v_4d = v_half.unsqueeze(0).unsqueeze(0)

        u_full = F.interpolate(u_4d, size=(H, W), mode="bilinear", align_corners=False)
        v_full = F.interpolate(v_4d, size=(H, W), mode="bilinear", align_corners=False)

        u_full = u_full.squeeze(0).squeeze(0)
        v_full = v_full.squeeze(0).squeeze(0)

        return y_full, u_full, v_full


# Problem configuration - 1080p
frame_height = 1080
frame_width = 1920


def get_inputs():
    y = torch.rand(frame_height, frame_width)
    u = torch.rand(frame_height // 2, frame_width // 2)
    v = torch.rand(frame_height // 2, frame_width // 2)
    return [y, u, v]


def get_init_inputs():
    return []
