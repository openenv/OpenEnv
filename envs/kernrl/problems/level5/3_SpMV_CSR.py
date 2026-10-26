"""
Sparse Matrix-Vector Multiplication (SpMV) in CSR Format

Computes y = A * x where A is sparse in Compressed Sparse Row (CSR) format.
This is the core operation in iterative linear solvers (CG, GMRES, etc.).

CSR format stores:
- values: non-zero values
- col_indices: column index for each non-zero
- row_ptrs: start/end indices for each row in values array

This is memory-bound with irregular access patterns (load balancing challenge).

Optimization opportunities:
- Warp-level parallelism for load balancing
- Vectorized loads for the dense vector x
- Shared memory caching of x values
- CSR-adaptive algorithms (different strategies based on row length)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Sparse matrix-vector multiplication: y = A * x

    The sparse matrix A is stored in CSR format for efficient row-wise access.
    """

    def __init__(self, num_rows: int, num_cols: int):
        super(Model, self).__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols

    def forward(
        self,
        values: torch.Tensor,
        col_indices: torch.Tensor,
        row_ptrs: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute y = A * x using CSR format.

        Args:
            values: (nnz,) non-zero values of A
            col_indices: (nnz,) column indices of non-zeros
            row_ptrs: (num_rows + 1,) row pointers
            x: (num_cols,) dense input vector

        Returns:
            y: (num_rows,) result vector
        """
        y = torch.zeros(self.num_rows, device=x.device, dtype=x.dtype)

        # Row-wise SpMV
        for i in range(self.num_rows):
            start = row_ptrs[i].item()
            end = row_ptrs[i + 1].item()

            # Dot product of row i with x
            if end > start:
                row_values = values[start:end]
                row_cols = col_indices[start:end]
                y[i] = (row_values * x[row_cols]).sum()

        return y


# Problem configuration - sparse matrix
num_rows = 10000
num_cols = 10000
avg_nnz_per_row = 50  # ~0.5% density
total_nnz = num_rows * avg_nnz_per_row


def get_inputs():
    # Generate random sparse matrix in CSR format
    # Each row has random number of non-zeros
    row_lengths = torch.randint(1, avg_nnz_per_row * 2, (num_rows,))
    row_lengths = torch.clamp(row_lengths, max=num_cols)

    # Build row pointers
    row_ptrs = torch.zeros(num_rows + 1, dtype=torch.int64)
    row_ptrs[1:] = torch.cumsum(row_lengths, dim=0)
    actual_nnz = row_ptrs[-1].item()

    # Generate random column indices and values
    col_indices = torch.zeros(actual_nnz, dtype=torch.int64)
    values = torch.randn(actual_nnz)

    for i in range(num_rows):
        start = row_ptrs[i].item()
        end = row_ptrs[i + 1].item()
        if end > start:
            # Random unique column indices for this row
            num_nnz = end - start
            cols = torch.randperm(num_cols)[:num_nnz].sort()[0]
            col_indices[start:end] = cols

    # Dense vector
    x = torch.randn(num_cols)

    return [values, col_indices, row_ptrs, x]


def get_init_inputs():
    return [num_rows, num_cols]
