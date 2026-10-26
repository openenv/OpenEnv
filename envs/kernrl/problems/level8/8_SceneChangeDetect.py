"""
Scene Change Detection

Detects scene changes (cuts) in video by comparing frame similarity.
Used for video segmentation, summarization, and compression optimization.

Computes various similarity metrics between consecutive frames.

Optimization opportunities:
- Hierarchical comparison (thumbnail first)
- Histogram-based comparison
- Parallel metric computation
- Early termination for obvious cuts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Scene change detection using multiple metrics.
    """

    def __init__(self, sad_threshold: float = 0.3, hist_threshold: float = 0.5):
        super(Model, self).__init__()
        self.sad_threshold = sad_threshold
        self.hist_threshold = hist_threshold

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> tuple:
        """
        Detect if scene change occurred between frames.

        Args:
            frame1: (H, W) first frame
            frame2: (H, W) second frame

        Returns:
            is_scene_change: bool tensor
            sad_score: normalized SAD score
            hist_diff: histogram difference score
        """
        H, W = frame1.shape

        # Metric 1: Normalized SAD
        sad = (frame1 - frame2).abs().mean()
        sad_score = sad / frame1.abs().mean().clamp(min=1e-6)

        # Metric 2: Histogram difference (chi-squared)
        # Quantize to 32 bins
        bins = 32
        frame1_q = (frame1 * (bins - 1)).clamp(0, bins - 1).long().flatten()
        frame2_q = (frame2 * (bins - 1)).clamp(0, bins - 1).long().flatten()

        hist1 = torch.bincount(frame1_q, minlength=bins).float()
        hist2 = torch.bincount(frame2_q, minlength=bins).float()

        # Normalize histograms
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        # Chi-squared distance
        chi_sq = ((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10)).sum() / 2

        # Metric 3: Edge difference (structural change)
        # Simple gradient magnitude comparison
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=frame1.device,
        )
        sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)

        f1 = frame1.unsqueeze(0).unsqueeze(0)
        f2 = frame2.unsqueeze(0).unsqueeze(0)

        edge1 = F.conv2d(f1, sobel_x, padding=1).abs().mean()
        edge2 = F.conv2d(f2, sobel_x, padding=1).abs().mean()
        edge_diff = (edge1 - edge2).abs() / (edge1 + edge2 + 1e-10)

        # Combine metrics for final decision
        is_scene_change = (sad_score > self.sad_threshold) | (
            chi_sq > self.hist_threshold
        )

        return is_scene_change, sad_score, chi_sq


# Problem configuration
frame_height = 480
frame_width = 640


def get_inputs():
    frame1 = torch.rand(frame_height, frame_width)
    frame2 = torch.rand(frame_height, frame_width)
    return [frame1, frame2]


def get_init_inputs():
    return [0.3, 0.5]  # sad_threshold, hist_threshold
