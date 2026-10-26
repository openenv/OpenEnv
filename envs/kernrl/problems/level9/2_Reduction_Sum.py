"""
Parallel Reduction - Sum

Computes the sum of all elements in an array.
Classic GPU algorithm with multiple reduction strategies.

Optimization opportunities:
- Sequential addressing to avoid bank conflicts
- Loop unrolling for the last warp
- Warp-level reduction using shuffle
- Grid-stride loops for large arrays
- Persistent kernels
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Parallel sum reduction.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute sum of all elements.

        Args:
            input: (N,) input array

        Returns:
            sum: scalar tensor
        """
        return input.sum()


# Problem configuration
array_size = 64 * 1024 * 1024  # 64M elements


def get_inputs():
    data = torch.rand(array_size)
    return [data]


def get_init_inputs():
    return []
