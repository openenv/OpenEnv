"""
Particle-in-Cell (PIC) Charge Deposition

Deposits particle charges onto a grid using linear interpolation (CIC - Cloud-in-Cell).
This is a key operation in plasma physics simulations.

Challenge: Atomic operations needed due to race conditions when multiple particles
deposit to the same grid cell.

Optimization opportunities:
- Sorting particles by cell for coalesced access
- Shared memory atomics with global reduction
- Histogram-style optimizations
- Warp-level vote/ballot for conflict detection
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Deposits particle charges onto a 2D grid using Cloud-in-Cell (CIC) interpolation.

    Each particle contributes charge to its 4 nearest grid points with
    bilinear weighting based on distance.
    """

    def __init__(self, grid_size: int = 256):
        super(Model, self).__init__()
        self.grid_size = grid_size

    def forward(self, positions: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        """
        Deposit particle charges onto grid.

        Args:
            positions: (N, 2) particle positions in [0, grid_size)
            charges: (N,) particle charges

        Returns:
            grid: (grid_size, grid_size) charge density grid
        """
        N = positions.shape[0]
        grid = torch.zeros(
            self.grid_size,
            self.grid_size,
            device=positions.device,
            dtype=positions.dtype,
        )

        # Get cell indices and fractional positions
        # Cell index is floor of position
        cell_x = positions[:, 0].floor().long()
        cell_y = positions[:, 1].floor().long()

        # Fractional position within cell [0, 1)
        fx = positions[:, 0] - cell_x.float()
        fy = positions[:, 1] - cell_y.float()

        # Clamp to valid range
        cell_x = torch.clamp(cell_x, 0, self.grid_size - 2)
        cell_y = torch.clamp(cell_y, 0, self.grid_size - 2)

        # CIC weights for 4 corners
        w00 = (1 - fx) * (1 - fy) * charges
        w10 = fx * (1 - fy) * charges
        w01 = (1 - fx) * fy * charges
        w11 = fx * fy * charges

        # Scatter-add to grid (this is the bottleneck - atomic operations)
        for i in range(N):
            ix, iy = cell_x[i].item(), cell_y[i].item()
            grid[ix, iy] += w00[i]
            grid[ix + 1, iy] += w10[i]
            grid[ix, iy + 1] += w01[i]
            grid[ix + 1, iy + 1] += w11[i]

        return grid


# Problem configuration
num_particles = 100000
grid_size = 256


def get_inputs():
    # Random particles uniformly distributed in grid
    positions = torch.rand(num_particles, 2) * (grid_size - 1)
    charges = torch.randn(num_particles)  # Can be positive or negative
    return [positions, charges]


def get_init_inputs():
    return [grid_size]
