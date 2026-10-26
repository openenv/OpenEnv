"""
Stream Compaction (Filter)

Removes elements that don't satisfy a predicate, compacting the result.
Also known as filtering or partition.

Example: Remove all zeros from array.

Optimization opportunities:
- Scan-based compaction
- Warp-level ballot for predicate evaluation
- Per-block compaction + global gather
- Decoupled lookback for single-pass
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Stream compaction - removes elements not satisfying predicate.
    """

    def __init__(self, threshold: float = 0.5):
        super(Model, self).__init__()
        self.threshold = threshold

    def forward(self, input: torch.Tensor) -> tuple:
        """
        Compact array keeping only elements >= threshold.

        Args:
            input: (N,) input array

        Returns:
            output: (M,) compacted array (M <= N)
            count: number of elements kept
        """
        mask = input >= self.threshold
        output = input[mask]
        count = mask.sum()
        return output, count


# Problem configuration
array_size = 16 * 1024 * 1024


def get_inputs():
    data = torch.rand(array_size)
    return [data]


def get_init_inputs():
    return [0.5]  # threshold
