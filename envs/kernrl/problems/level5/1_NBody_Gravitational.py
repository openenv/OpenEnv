"""
N-Body Gravitational Simulation

Computes gravitational forces between N particles using direct O(N^2) summation.
This is a classic GPU workload with high arithmetic intensity.

Optimization opportunities:
- Shared memory tiling to reduce global memory bandwidth
- Thread coarsening to increase work per thread
- Vectorized loads/stores
- Fast reciprocal square root (rsqrt)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Computes gravitational acceleration on each particle due to all other particles.

    Uses the gravitational formula: a_i = sum_j (m_j * (r_j - r_i) / |r_j - r_i|^3)
    with softening to avoid singularities: |r|^3 -> (|r|^2 + eps^2)^(3/2)
    """

    def __init__(self, softening: float = 0.01):
        super(Model, self).__init__()
        self.softening = softening
        self.softening_sq = softening * softening

    def forward(self, positions: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """
        Compute gravitational accelerations.

        Args:
            positions: (N, 3) particle positions [x, y, z]
            masses: (N,) particle masses

        Returns:
            accelerations: (N, 3) acceleration vectors for each particle
        """
        N = positions.shape[0]

        # Compute pairwise displacements: r_j - r_i
        # Shape: (N, N, 3)
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Compute squared distances with softening
        # Shape: (N, N)
        dist_sq = (diff**2).sum(dim=2) + self.softening_sq

        # Compute 1 / |r|^3 = 1 / (dist_sq)^(3/2)
        # Shape: (N, N)
        inv_dist_cubed = dist_sq ** (-1.5)

        # Compute force contributions: m_j * (r_j - r_i) / |r_j - r_i|^3
        # Shape: (N, N, 3)
        force_contributions = (
            masses.unsqueeze(0).unsqueeze(2) * diff * inv_dist_cubed.unsqueeze(2)
        )

        # Sum over all j to get total acceleration on each particle i
        # Shape: (N, 3)
        accelerations = force_contributions.sum(dim=1)

        return accelerations


# Problem configuration
num_particles = 4096  # O(N^2) = 16M interactions


def get_inputs():
    # Random positions in unit cube
    positions = torch.randn(num_particles, 3)
    # Random masses (positive)
    masses = torch.rand(num_particles) + 0.1
    return [positions, masses]


def get_init_inputs():
    return [0.01]  # softening parameter
