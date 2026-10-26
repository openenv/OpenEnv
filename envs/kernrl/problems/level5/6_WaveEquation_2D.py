"""
2D Wave Equation Finite Difference

Explicit time stepping for the 2D wave equation:
u_tt = c^2 * (u_xx + u_yy)

Uses leapfrog integration:
u_new = 2*u_curr - u_prev + c^2*dt^2*(Laplacian(u_curr))

This has similar structure to heat equation stencil but with temporal dependence.

Optimization opportunities:
- Temporal blocking to keep multiple time steps in cache
- Shared memory tiling
- Vectorized loads
- Prefetching for next timestep
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    One timestep of the 2D wave equation using finite differences.

    Implements leapfrog integration which is second-order accurate in time.
    """

    def __init__(self, c: float = 1.0, dt: float = 0.01, dx: float = 0.1):
        super(Model, self).__init__()
        self.c = c
        self.dt = dt
        self.dx = dx
        # CFL stability requires c*dt/dx < 1/sqrt(2) for 2D
        self.coeff = (c * dt / dx) ** 2

    def forward(self, u_curr: torch.Tensor, u_prev: torch.Tensor) -> torch.Tensor:
        """
        Compute next timestep of wave equation.

        Args:
            u_curr: (H, W) current displacement field
            u_prev: (H, W) previous displacement field

        Returns:
            u_next: (H, W) next displacement field
        """
        # Compute Laplacian using 5-point stencil
        laplacian = (
            (
                u_curr[:-2, 1:-1]  # North
                + u_curr[2:, 1:-1]  # South
                + u_curr[1:-1, :-2]  # West
                + u_curr[1:-1, 2:]  # East
                - 4 * u_curr[1:-1, 1:-1]
            )
            / (self.dx**2)
        )

        # Initialize output with boundary values (Dirichlet BC)
        u_next = torch.zeros_like(u_curr)

        # Leapfrog update for interior
        u_next[1:-1, 1:-1] = (
            2 * u_curr[1:-1, 1:-1]
            - u_prev[1:-1, 1:-1]
            + self.coeff * laplacian * (self.dx**2)
        )

        return u_next


# Problem configuration
grid_height = 1024
grid_width = 1024


def get_inputs():
    # Initial condition: Gaussian pulse in center
    x = torch.linspace(0, 1, grid_width)
    y = torch.linspace(0, 1, grid_height)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Gaussian centered at (0.5, 0.5)
    u_curr = torch.exp(-100 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
    u_prev = u_curr.clone()  # Zero initial velocity

    return [u_curr, u_prev]


def get_init_inputs():
    return [1.0, 0.001, 0.01]  # c, dt, dx
