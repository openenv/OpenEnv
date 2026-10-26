"""
Segmented Prefix Sum

Computes prefix sum within segments defined by a flag array.
Resets accumulator at segment boundaries.

Used in graph algorithms, sparse operations, and more.

Optimization opportunities:
- Head flags for segment boundaries
- Warp-level segmented scan
- Decoupled lookback with segment handling
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Segmented exclusive prefix sum.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self, values: torch.Tensor, segment_heads: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute segmented exclusive prefix sum.

        Args:
            values: (N,) input values
            segment_heads: (N,) boolean tensor, True at segment starts

        Returns:
            output: (N,) segmented exclusive prefix sums
        """
        N = values.shape[0]
        output = torch.zeros_like(values)

        # Find segment starts
        segment_starts = torch.where(segment_heads)[0].tolist()
        if 0 not in segment_starts:
            segment_starts = [0] + segment_starts
        segment_starts.append(N)

        # Process each segment
        for i in range(len(segment_starts) - 1):
            start = segment_starts[i]
            end = segment_starts[i + 1]
            segment = values[start:end]
            output[start:end] = torch.cumsum(segment, dim=0) - segment

        return output


# Problem configuration
array_size = 16 * 1024 * 1024
num_segments = 1000


def get_inputs():
    values = torch.rand(array_size)
    # Random segment heads
    segment_heads = torch.zeros(array_size, dtype=torch.bool)
    segment_heads[0] = True  # First element always starts a segment
    head_positions = torch.randperm(array_size - 1)[: num_segments - 1] + 1
    segment_heads[head_positions] = True
    return [values, segment_heads]


def get_init_inputs():
    return []
