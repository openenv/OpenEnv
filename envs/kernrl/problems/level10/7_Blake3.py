"""
BLAKE3 Hash Function

Modern cryptographic hash function designed for speed.
Based on BLAKE2 and Bao tree hashing.

Key features:
- 4 rounds (vs 10 in BLAKE2)
- Merkle tree structure for parallelism
- SIMD-friendly design

Optimization opportunities:
- SIMD vectorization of G function
- Parallel chunk processing
- Persistent threads for tree hashing
- Register-heavy implementation
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    BLAKE3 hash function (simplified single-chunk version).
    """

    def __init__(self):
        super(Model, self).__init__()

        # BLAKE3 IV (same as BLAKE2s)
        IV = torch.tensor(
            [
                0x6A09E667,
                0xBB67AE85,
                0x3C6EF372,
                0xA54FF53A,
                0x510E527F,
                0x9B05688C,
                0x1F83D9AB,
                0x5BE0CD19,
            ],
            dtype=torch.int64,
        )
        self.register_buffer("IV", IV)

        # Message schedule permutation
        MSG_SCHEDULE = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
                [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
                [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
                [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
                [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
                [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
            ],
            dtype=torch.long,
        )
        self.register_buffer("MSG_SCHEDULE", MSG_SCHEDULE)

    def _rotl(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Right rotation (BLAKE3 uses right rotation)."""
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    def _g(
        self,
        state: torch.Tensor,
        a: int,
        b: int,
        c: int,
        d: int,
        mx: torch.Tensor,
        my: torch.Tensor,
    ) -> torch.Tensor:
        """BLAKE3 G function (mixing function)."""
        state = state.clone()

        state[a] = (state[a] + state[b] + mx) & 0xFFFFFFFF
        state[d] = self._rotl(state[d] ^ state[a], 16)

        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] = self._rotl(state[b] ^ state[c], 12)

        state[a] = (state[a] + state[b] + my) & 0xFFFFFFFF
        state[d] = self._rotl(state[d] ^ state[a], 8)

        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] = self._rotl(state[b] ^ state[c], 7)

        return state

    def _round(
        self, state: torch.Tensor, m: torch.Tensor, schedule: torch.Tensor
    ) -> torch.Tensor:
        """One round of mixing."""
        msg = m[schedule]

        # Column step
        state = self._g(state, 0, 4, 8, 12, msg[0], msg[1])
        state = self._g(state, 1, 5, 9, 13, msg[2], msg[3])
        state = self._g(state, 2, 6, 10, 14, msg[4], msg[5])
        state = self._g(state, 3, 7, 11, 15, msg[6], msg[7])

        # Diagonal step
        state = self._g(state, 0, 5, 10, 15, msg[8], msg[9])
        state = self._g(state, 1, 6, 11, 12, msg[10], msg[11])
        state = self._g(state, 2, 7, 8, 13, msg[12], msg[13])
        state = self._g(state, 3, 4, 9, 14, msg[14], msg[15])

        return state

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        """
        Compute BLAKE3 hash of a single chunk (64 bytes).

        Args:
            message: (64,) message bytes (one chunk)

        Returns:
            hash: (32,) 256-bit hash as bytes
        """
        device = message.device

        # Parse message into 16 32-bit words
        m = torch.zeros(16, dtype=torch.int64, device=device)
        for i in range(16):
            m[i] = (
                message[i * 4].long()
                | (message[i * 4 + 1].long() << 8)
                | (message[i * 4 + 2].long() << 16)
                | (message[i * 4 + 3].long() << 24)
            )

        # Initialize state
        state = torch.zeros(16, dtype=torch.int64, device=device)
        state[0:8] = self.IV
        state[8:12] = self.IV[0:4]
        state[12] = 0  # counter low
        state[13] = 0  # counter high
        state[14] = 64  # block len
        state[15] = 0b00001011  # flags: CHUNK_START | CHUNK_END | ROOT

        # 7 rounds (BLAKE3 uses 7 rounds)
        for r in range(7):
            schedule = self.MSG_SCHEDULE[r % 7]
            state = self._round(state, m, schedule)

        # Finalize: XOR first half with second half, then with IV
        h = (state[0:8] ^ state[8:16]) & 0xFFFFFFFF

        # Convert to bytes
        result = torch.zeros(32, dtype=torch.int64, device=device)
        for i in range(8):
            result[i * 4] = h[i] & 0xFF
            result[i * 4 + 1] = (h[i] >> 8) & 0xFF
            result[i * 4 + 2] = (h[i] >> 16) & 0xFF
            result[i * 4 + 3] = (h[i] >> 24) & 0xFF

        return result


# Problem configuration
def get_inputs():
    message = torch.randint(0, 256, (64,), dtype=torch.int64)
    return [message]


def get_init_inputs():
    return []
