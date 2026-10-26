"""
Scatter Operation

Scatters values to specified indices in output array.
out[indices[i]] = values[i]

Challenge: Multiple values may scatter to same index (race condition).

Optimization opportunities:
- Atomic operations for conflicts
- Sorting by destination for coalescing
- Segmented scatter
- Conflict detection with warp ballot
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Scatter values to indices.
    """

    def __init__(self, output_size: int = 1000000):
        super(Model, self).__init__()
        self.output_size = output_size

    def forward(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Scatter values to indices.

        Args:
            values: (N,) values to scatter
            indices: (N,) destination indices

        Returns:
            output: (output_size,) scattered values
        """
        output = torch.zeros(self.output_size, device=values.device, dtype=values.dtype)
        output.scatter_(0, indices, values)
        return output


# Problem configuration
num_values = 4 * 1024 * 1024
output_size = 1000000


def get_inputs():
    values = torch.rand(num_values)
    indices = torch.randint(0, output_size, (num_values,))
    return [values, indices]


def get_init_inputs():
    return [output_size]
