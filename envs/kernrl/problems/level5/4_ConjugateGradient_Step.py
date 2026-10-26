"""
Conjugate Gradient Solver Step

One iteration of the Conjugate Gradient method for solving Ax = b.
This combines multiple BLAS operations that can be fused:
- Matrix-vector product (SpMV or dense)
- Multiple dot products
- Vector updates (AXPY)

Optimization opportunities:
- Kernel fusion to reduce memory traffic
- Persistent threads to keep intermediate results in registers
- Overlapping computation with memory operations
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    One iteration of the Conjugate Gradient method.

    Given current state (x, r, p, rsold), computes next iteration.
    This is a key building block for large-scale linear system solvers.

    CG iteration:
        Ap = A @ p
        alpha = rsold / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r @ r
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        A: torch.Tensor,
        x: torch.Tensor,
        r: torch.Tensor,
        p: torch.Tensor,
        rsold: torch.Tensor,
    ) -> tuple:
        """
        Perform one CG iteration.

        Args:
            A: (N, N) symmetric positive definite matrix
            x: (N,) current solution estimate
            r: (N,) current residual (b - Ax)
            p: (N,) current search direction
            rsold: scalar, r @ r from previous iteration

        Returns:
            x_new, r_new, p_new, rsnew: updated CG state
        """
        # Matrix-vector product
        Ap = A @ p

        # Compute step size
        pAp = torch.dot(p, Ap)
        alpha = rsold / pAp

        # Update solution
        x_new = x + alpha * p

        # Update residual
        r_new = r - alpha * Ap

        # Compute new residual norm squared
        rsnew = torch.dot(r_new, r_new)

        # Update search direction
        beta = rsnew / rsold
        p_new = r_new + beta * p

        return x_new, r_new, p_new, rsnew


# Problem configuration
matrix_size = 4096


def get_inputs():
    # Create a symmetric positive definite matrix
    # A = Q @ D @ Q.T where D has positive eigenvalues
    Q = torch.randn(matrix_size, matrix_size)
    Q, _ = torch.linalg.qr(Q)  # Orthogonal matrix
    D = torch.diag(torch.rand(matrix_size) + 0.1)  # Positive eigenvalues
    A = Q @ D @ Q.T

    # Random initial state
    x = torch.randn(matrix_size)
    b = torch.randn(matrix_size)
    r = b - A @ x  # Initial residual
    p = r.clone()  # Initial search direction
    rsold = torch.dot(r, r)

    return [A, x, r, p, rsold]


def get_init_inputs():
    return []
