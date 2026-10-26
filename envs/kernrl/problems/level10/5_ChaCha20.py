"""
ChaCha20 Stream Cipher

Modern stream cipher used in TLS 1.3 and WireGuard.
Based on ARX (Add-Rotate-XOR) operations.

Core operation is the quarter-round:
a += b; d ^= a; d <<<= 16
c += d; b ^= c; b <<<= 12
a += b; d ^= a; d <<<= 8
c += d; b ^= c; b <<<= 7

Optimization opportunities:
- SIMD vectorization (4 parallel quarter-rounds)
- Unrolled rounds
- Parallel block generation
- Register-resident state
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ChaCha20 stream cipher.
    """

    def __init__(self):
        super(Model, self).__init__()

        # ChaCha20 constants "expand 32-byte k"
        constants = torch.tensor(
            [
                0x61707865,  # "expa"
                0x3320646E,  # "nd 3"
                0x79622D32,  # "2-by"
                0x6B206574,  # "te k"
            ],
            dtype=torch.int64,
        )
        self.register_buffer("constants", constants)

    def _rotl(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Left rotation for 32-bit values."""
        return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF

    def _quarter_round(
        self, state: torch.Tensor, a: int, b: int, c: int, d: int
    ) -> torch.Tensor:
        """Perform ChaCha20 quarter-round."""
        state = state.clone()

        state[a] = (state[a] + state[b]) & 0xFFFFFFFF
        state[d] = self._rotl(state[d] ^ state[a], 16)

        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] = self._rotl(state[b] ^ state[c], 12)

        state[a] = (state[a] + state[b]) & 0xFFFFFFFF
        state[d] = self._rotl(state[d] ^ state[a], 8)

        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] = self._rotl(state[b] ^ state[c], 7)

        return state

    def forward(
        self, key: torch.Tensor, nonce: torch.Tensor, counter: int = 0
    ) -> torch.Tensor:
        """
        Generate 64 bytes of keystream.

        Args:
            key: (8,) 256-bit key as 8 32-bit words
            nonce: (3,) 96-bit nonce as 3 32-bit words
            counter: 32-bit block counter

        Returns:
            keystream: (16,) 64-byte block as 16 32-bit words
        """
        device = key.device

        # Initialize state
        state = torch.zeros(16, dtype=torch.int64, device=device)
        state[0:4] = self.constants
        state[4:12] = key
        state[12] = counter
        state[13:16] = nonce

        # Working state
        working = state.clone()

        # 20 rounds (10 double rounds)
        for _ in range(10):
            # Column rounds
            working = self._quarter_round(working, 0, 4, 8, 12)
            working = self._quarter_round(working, 1, 5, 9, 13)
            working = self._quarter_round(working, 2, 6, 10, 14)
            working = self._quarter_round(working, 3, 7, 11, 15)

            # Diagonal rounds
            working = self._quarter_round(working, 0, 5, 10, 15)
            working = self._quarter_round(working, 1, 6, 11, 12)
            working = self._quarter_round(working, 2, 7, 8, 13)
            working = self._quarter_round(working, 3, 4, 9, 14)

        # Add original state
        keystream = (working + state) & 0xFFFFFFFF

        return keystream


# Problem configuration
def get_inputs():
    key = torch.randint(0, 2**32, (8,), dtype=torch.int64)
    nonce = torch.randint(0, 2**32, (3,), dtype=torch.int64)
    return [key, nonce, 0]


def get_init_inputs():
    return []
