"""
256-bin Histogram Computation

Computes a histogram of 8-bit values (0-255).
This is a fundamental operation in image processing and statistics.

Challenge: Atomic operations for bin updates create contention.

Optimization opportunities:
- Per-thread or per-warp private histograms with final reduction
- Shared memory histograms per thread block
- Vote/ballot for conflict detection
- Sorting-based histogram
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Computes a 256-bin histogram of byte values.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute histogram of input data.

        Args:
            data: (N,) tensor of values in range [0, 255], dtype=uint8 or int

        Returns:
            histogram: (256,) bin counts
        """
        # Ensure integer type and valid range
        data = data.long()
        data = torch.clamp(data, 0, 255)

        # Use bincount for histogram
        histogram = torch.bincount(data, minlength=256).float()

        return histogram


# Problem configuration
num_pixels = 4 * 1024 * 1024  # 4 megapixels


def get_inputs():
    # Random byte values (simulating grayscale image)
    data = torch.randint(0, 256, (num_pixels,), dtype=torch.long)
    return [data]


def get_init_inputs():
    return []
