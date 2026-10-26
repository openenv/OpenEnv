"""
Merkle Tree Root Computation

Computes the root hash of a Merkle tree from leaf hashes.
Used in blockchain, certificate transparency, and data verification.

Tree structure: leaves at bottom, each internal node is hash of children.
                    root
                   /    \
               node      node
              /   \     /    \
            leaf leaf leaf  leaf

Optimization opportunities:
- Parallel hashing at each level
- Coalesced memory access for hash pairs
- Persistent kernel across levels
- Shared memory for intermediate hashes
"""

import hashlib

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Merkle tree root computation from leaf hashes.

    Uses simple concatenation + hash for internal nodes:
    parent = hash(left || right)
    """

    def __init__(self):
        super(Model, self).__init__()

    def _simple_hash(self, data: torch.Tensor) -> torch.Tensor:
        """Simple hash function using XOR and rotation (for demo)."""
        # In practice, use SHA-256; this is a simplified version
        result = torch.zeros(32, dtype=torch.int64, device=data.device)

        # Mix input bytes
        for i in range(len(data)):
            result[i % 32] = (result[i % 32] ^ data[i] + data[i] * 31) & 0xFF

        # Additional mixing
        for _ in range(4):
            for i in range(32):
                result[i] = (
                    result[i] ^ result[(i + 7) % 32] + result[(i + 13) % 32]
                ) & 0xFF

        return result

    def forward(self, leaves: torch.Tensor) -> torch.Tensor:
        """
        Compute Merkle tree root from leaf hashes.

        Args:
            leaves: (N, 32) N leaf hashes, each 32 bytes

        Returns:
            root: (32,) root hash
        """
        N = leaves.shape[0]
        device = leaves.device

        # Ensure N is power of 2 (pad with zeros if needed)
        if N & (N - 1) != 0:
            next_pow2 = 1 << (N - 1).bit_length()
            padding = torch.zeros(next_pow2 - N, 32, dtype=leaves.dtype, device=device)
            leaves = torch.cat([leaves, padding], dim=0)
            N = next_pow2

        current_level = leaves

        # Build tree bottom-up
        while current_level.shape[0] > 1:
            num_nodes = current_level.shape[0]
            next_level = torch.zeros(
                num_nodes // 2, 32, dtype=leaves.dtype, device=device
            )

            for i in range(num_nodes // 2):
                # Concatenate children
                left = current_level[2 * i]
                right = current_level[2 * i + 1]
                combined = torch.cat([left, right])

                # Hash to get parent
                next_level[i] = self._simple_hash(combined)

            current_level = next_level

        return current_level[0]


# Problem configuration
num_leaves = 1024


def get_inputs():
    # Random leaf hashes
    leaves = torch.randint(0, 256, (num_leaves, 32), dtype=torch.int64)
    return [leaves]


def get_init_inputs():
    return []
