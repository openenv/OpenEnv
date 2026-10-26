"""
Gather Operation

Gathers values from source array based on index array.
out[i] = source[indices[i]]

Optimization opportunities:
- Coalesced reads by sorting indices
- Texture memory for cached reads
- Prefetching for sequential access patterns
- Vectorized loads when possible
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Gather values from indices.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Gather values from source at indices.

        Args:
            source: (M,) source array
            indices: (N,) indices into source

        Returns:
            output: (N,) gathered values
        """
        return source[indices]


# Problem configuration
source_size = 1000000
num_gathers = 16 * 1024 * 1024


def get_inputs():
    source = torch.rand(source_size)
    indices = torch.randint(0, source_size, (num_gathers,))
    return [source, indices]


def get_init_inputs():
    return []
