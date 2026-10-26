"""
Parallel Reduction - Maximum

Finds the maximum element in an array.
Similar structure to sum reduction but with max operation.

Optimization opportunities:
- Same as sum reduction
- Can use warp vote for early termination
- Max with index tracking (argmax)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Parallel max reduction.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Find maximum element.

        Args:
            input: (N,) input array

        Returns:
            max_val: scalar tensor
        """
        return input.max()


# Problem configuration
array_size = 64 * 1024 * 1024


def get_inputs():
    data = torch.rand(array_size)
    return [data]


def get_init_inputs():
    return []
