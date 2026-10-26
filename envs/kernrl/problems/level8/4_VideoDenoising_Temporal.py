"""
Temporal Video Denoising

Denoises video by averaging aligned frames over time.
More effective than single-frame denoising by using temporal redundancy.

Optimization opportunities:
- Motion-compensated temporal averaging
- Adaptive weighting based on motion confidence
- Sliding window temporal filter
- Parallel processing of temporal neighborhoods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Temporal averaging denoiser for video.

    Averages multiple frames with optional motion compensation.
    """

    def __init__(self, num_frames: int = 5):
        super(Model, self).__init__()
        self.num_frames = num_frames

    def forward(self, frames: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
        """
        Denoise the middle frame using temporal averaging.

        Args:
            frames: (T, H, W) stack of T frames centered on frame to denoise
            flows: (T-1, H, W, 2) optical flows between consecutive frames

        Returns:
            denoised: (H, W) denoised middle frame
        """
        T, H, W = frames.shape
        mid = T // 2

        # Accumulate warped frames
        accumulated = frames[mid].clone()
        weight = torch.ones(H, W, device=frames.device)

        # Create base grid
        y_coords = torch.linspace(-1, 1, H, device=frames.device)
        x_coords = torch.linspace(-1, 1, W, device=frames.device)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")
        base_grid = torch.stack([X, Y], dim=-1)

        # Warp frames to middle frame and accumulate
        for t in range(T):
            if t == mid:
                continue

            # Compute cumulative flow from frame t to middle frame
            cumulative_flow = torch.zeros(H, W, 2, device=frames.device)

            if t < mid:
                for i in range(t, mid):
                    cumulative_flow += flows[i]
            else:
                for i in range(mid, t):
                    cumulative_flow -= flows[i]

            # Normalize flow
            flow_normalized = cumulative_flow.clone()
            flow_normalized[..., 0] = cumulative_flow[..., 0] / (W / 2)
            flow_normalized[..., 1] = cumulative_flow[..., 1] / (H / 2)

            # Warp frame
            grid = base_grid - flow_normalized
            frame_batch = frames[t : t + 1].unsqueeze(0)  # (1, 1, H, W)
            grid_batch = grid.unsqueeze(0)  # (1, H, W, 2)

            warped = F.grid_sample(
                frame_batch,
                grid_batch,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            warped = warped.squeeze()

            # Compute motion confidence (simple: inverse of flow magnitude)
            flow_mag = cumulative_flow.norm(dim=-1)
            confidence = torch.exp(-flow_mag / 10)

            accumulated += warped * confidence
            weight += confidence

        # Normalize
        denoised = accumulated / weight

        return denoised


# Problem configuration
num_temporal_frames = 5
frame_height = 480
frame_width = 640


def get_inputs():
    frames = torch.rand(num_temporal_frames, frame_height, frame_width)
    # Small random flows between frames
    flows = torch.randn(num_temporal_frames - 1, frame_height, frame_width, 2) * 2
    return [frames, flows]


def get_init_inputs():
    return [num_temporal_frames]
