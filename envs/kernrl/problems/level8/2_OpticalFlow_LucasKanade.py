"""
Lucas-Kanade Optical Flow

Estimates dense optical flow using the Lucas-Kanade method with pyramids.
Assumes brightness constancy: I(x,y,t) = I(x+u, y+v, t+1)

For each pixel, solves:
[Ix^2    IxIy] [u]   [IxIt]
[IxIy   Iy^2] [v] = [IyIt]

Optimization opportunities:
- Image pyramid for large displacements
- Shared memory for gradient computation
- Warp-level matrix solves (2x2)
- Coalesced gradient loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Lucas-Kanade optical flow estimation.
    """

    def __init__(self, window_size: int = 15):
        super(Model, self).__init__()
        self.window_size = window_size
        self.half_win = window_size // 2

        # Sobel kernels for gradients
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )

        self.register_buffer("sobel_x", sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer("sobel_y", sobel_y.unsqueeze(0).unsqueeze(0))

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> tuple:
        """
        Compute optical flow from frame1 to frame2.

        Args:
            frame1: (H, W) first frame
            frame2: (H, W) second frame

        Returns:
            flow_u: (H, W) horizontal flow
            flow_v: (H, W) vertical flow
        """
        H, W = frame1.shape

        # Compute spatial gradients on average frame
        avg = (frame1 + frame2) / 2
        avg_4d = avg.unsqueeze(0).unsqueeze(0)

        Ix = F.conv2d(avg_4d, self.sobel_x, padding=1).squeeze()
        Iy = F.conv2d(avg_4d, self.sobel_y, padding=1).squeeze()

        # Temporal gradient
        It = frame2 - frame1

        # Initialize output
        flow_u = torch.zeros_like(frame1)
        flow_v = torch.zeros_like(frame1)

        # Pad images
        hw = self.half_win
        Ix_pad = F.pad(Ix, (hw, hw, hw, hw), mode="reflect")
        Iy_pad = F.pad(Iy, (hw, hw, hw, hw), mode="reflect")
        It_pad = F.pad(It, (hw, hw, hw, hw), mode="reflect")

        # For each pixel
        for y in range(H):
            for x in range(W):
                # Extract window
                Ix_win = Ix_pad[
                    y : y + self.window_size, x : x + self.window_size
                ].flatten()
                Iy_win = Iy_pad[
                    y : y + self.window_size, x : x + self.window_size
                ].flatten()
                It_win = It_pad[
                    y : y + self.window_size, x : x + self.window_size
                ].flatten()

                # Build A^T A and A^T b
                A00 = (Ix_win * Ix_win).sum()
                A01 = (Ix_win * Iy_win).sum()
                A11 = (Iy_win * Iy_win).sum()

                b0 = -(Ix_win * It_win).sum()
                b1 = -(Iy_win * It_win).sum()

                # Solve 2x2 system
                det = A00 * A11 - A01 * A01
                if det.abs() > 1e-6:
                    flow_u[y, x] = (A11 * b0 - A01 * b1) / det
                    flow_v[y, x] = (A00 * b1 - A01 * b0) / det

        return flow_u, flow_v


# Problem configuration - smaller for dense flow
frame_height = 240
frame_width = 320


def get_inputs():
    frame1 = torch.rand(frame_height, frame_width)
    frame2 = torch.rand(frame_height, frame_width)
    return [frame1, frame2]


def get_init_inputs():
    return [15]  # window_size
