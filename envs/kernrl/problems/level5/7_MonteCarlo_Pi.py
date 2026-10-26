"""
Monte Carlo Pi Estimation

Estimates Pi using Monte Carlo integration: count random points falling inside unit circle.
This represents a broader class of Monte Carlo integration problems.

The kernel involves:
- Random number generation (or processing pre-generated randoms)
- Point classification (inside/outside circle)
- Reduction to count hits

Optimization opportunities:
- Parallel random number generation (cuRAND)
- Warp-level reductions
- Persistent kernel for streaming random numbers
- Fused generation + classification + reduction
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Monte Carlo estimation of Pi using random sampling.

    Points (x, y) in [0, 1]^2 that satisfy x^2 + y^2 <= 1 fall inside
    the quarter circle. Ratio of hits to total * 4 estimates Pi.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, random_points: torch.Tensor) -> torch.Tensor:
        """
        Compute Pi estimate from random points.

        Args:
            random_points: (N, 2) random points in [0, 1]^2

        Returns:
            pi_estimate: scalar tensor with Pi estimate
        """
        # Compute distance from origin
        x = random_points[:, 0]
        y = random_points[:, 1]

        # Points inside quarter circle: x^2 + y^2 <= 1
        dist_sq = x * x + y * y
        inside = (dist_sq <= 1.0).float()

        # Ratio * 4 = Pi estimate
        N = random_points.shape[0]
        pi_estimate = 4.0 * inside.sum() / N

        return pi_estimate


# Problem configuration - many samples for accuracy
num_samples = 10_000_000


def get_inputs():
    # Pre-generate random points
    random_points = torch.rand(num_samples, 2)
    return [random_points]


def get_init_inputs():
    return []
