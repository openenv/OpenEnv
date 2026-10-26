"""
Frame Interpolation (Motion-Compensated)

Generates an intermediate frame between two input frames using motion vectors.
Used for frame rate conversion, slow motion, and video compression.

Optimization opportunities:
- Bilinear/bicubic warping
- Bidirectional motion compensation
- Occlusion handling
- Parallel pixel warping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Motion-compensated frame interpolation.

    Uses motion vectors to warp frames and blend.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        flow_01: torch.Tensor,
        t: float = 0.5,
    ) -> torch.Tensor:
        """
        Interpolate frame at time t between frame0 (t=0) and frame1 (t=1).

        Args:
            frame0: (H, W) or (C, H, W) frame at t=0
            frame1: (H, W) or (C, H, W) frame at t=1
            flow_01: (H, W, 2) optical flow from frame0 to frame1 (u, v)
            t: interpolation position in [0, 1]

        Returns:
            interpolated: same shape as input frames
        """
        # Handle shapes
        if frame0.dim() == 2:
            frame0 = frame0.unsqueeze(0)
            frame1 = frame1.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        C, H, W = frame0.shape

        # Create sampling grid
        y_coords = torch.linspace(-1, 1, H, device=frame0.device)
        x_coords = torch.linspace(-1, 1, W, device=frame0.device)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid = torch.stack([X, Y], dim=-1)  # (H, W, 2)

        # Normalize flow to [-1, 1] range
        flow_normalized = flow_01.clone()
        flow_normalized[..., 0] = flow_01[..., 0] / (W / 2)
        flow_normalized[..., 1] = flow_01[..., 1] / (H / 2)

        # Backward warp from t to 0
        grid_t_to_0 = grid - t * flow_normalized

        # Backward warp from t to 1
        grid_t_to_1 = grid + (1 - t) * flow_normalized

        # Add batch dimension for grid_sample
        frame0_batch = frame0.unsqueeze(0)
        frame1_batch = frame1.unsqueeze(0)
        grid_t_to_0 = grid_t_to_0.unsqueeze(0)
        grid_t_to_1 = grid_t_to_1.unsqueeze(0)

        # Warp frames
        warped_0 = F.grid_sample(
            frame0_batch,
            grid_t_to_0,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        warped_1 = F.grid_sample(
            frame1_batch,
            grid_t_to_1,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # Blend warped frames (simple linear blend)
        interpolated = (1 - t) * warped_0 + t * warped_1
        interpolated = interpolated.squeeze(0)

        if squeeze_output:
            interpolated = interpolated.squeeze(0)

        return interpolated


# Problem configuration
frame_height = 720
frame_width = 1280


def get_inputs():
    frame0 = torch.rand(frame_height, frame_width)
    frame1 = torch.rand(frame_height, frame_width)
    # Random small flow
    flow = torch.randn(frame_height, frame_width, 2) * 5
    return [frame0, frame1, flow, 0.5]


def get_init_inputs():
    return []
