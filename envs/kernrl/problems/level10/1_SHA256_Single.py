"""
SHA-256 Hash - Single Message

Computes SHA-256 hash of a message block.
Fundamental cryptographic primitive used in Bitcoin, TLS, etc.

SHA-256 operates on 512-bit (64-byte) blocks, producing 256-bit hash.

Optimization opportunities:
- Unroll compression rounds
- Use registers for working variables
- Vectorized message schedule computation
- Parallel hashing of multiple messages
"""

import hashlib

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    SHA-256 hash computation using PyTorch operations.

    This is a naive implementation - the optimized version should use
    bit manipulation intrinsics and unrolled loops.
    """

    def __init__(self):
        super(Model, self).__init__()

        # SHA-256 constants (first 32 bits of fractional parts of cube roots of first 64 primes)
        K = torch.tensor(
            [
                0x428A2F98,
                0x71374491,
                0xB5C0FBCF,
                0xE9B5DBA5,
                0x3956C25B,
                0x59F111F1,
                0x923F82A4,
                0xAB1C5ED5,
                0xD807AA98,
                0x12835B01,
                0x243185BE,
                0x550C7DC3,
                0x72BE5D74,
                0x80DEB1FE,
                0x9BDC06A7,
                0xC19BF174,
                0xE49B69C1,
                0xEFBE4786,
                0x0FC19DC6,
                0x240CA1CC,
                0x2DE92C6F,
                0x4A7484AA,
                0x5CB0A9DC,
                0x76F988DA,
                0x983E5152,
                0xA831C66D,
                0xB00327C8,
                0xBF597FC7,
                0xC6E00BF3,
                0xD5A79147,
                0x06CA6351,
                0x14292967,
                0x27B70A85,
                0x2E1B2138,
                0x4D2C6DFC,
                0x53380D13,
                0x650A7354,
                0x766A0ABB,
                0x81C2C92E,
                0x92722C85,
                0xA2BFE8A1,
                0xA81A664B,
                0xC24B8B70,
                0xC76C51A3,
                0xD192E819,
                0xD6990624,
                0xF40E3585,
                0x106AA070,
                0x19A4C116,
                0x1E376C08,
                0x2748774C,
                0x34B0BCB5,
                0x391C0CB3,
                0x4ED8AA4A,
                0x5B9CCA4F,
                0x682E6FF3,
                0x748F82EE,
                0x78A5636F,
                0x84C87814,
                0x8CC70208,
                0x90BEFFFA,
                0xA4506CEB,
                0xBEF9A3F7,
                0xC67178F2,
            ],
            dtype=torch.int64,
        )
        self.register_buffer("K", K)

        # Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
        H0 = torch.tensor(
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
        self.register_buffer("H0", H0)

    def _rotr(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Right rotation."""
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    def _ch(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return (x & y) ^ (~x & z) & 0xFFFFFFFF

    def _maj(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return (x & y) ^ (x & z) ^ (y & z)

    def _sigma0(self, x: torch.Tensor) -> torch.Tensor:
        return self._rotr(x, 2) ^ self._rotr(x, 13) ^ self._rotr(x, 22)

    def _sigma1(self, x: torch.Tensor) -> torch.Tensor:
        return self._rotr(x, 6) ^ self._rotr(x, 11) ^ self._rotr(x, 25)

    def _gamma0(self, x: torch.Tensor) -> torch.Tensor:
        return self._rotr(x, 7) ^ self._rotr(x, 18) ^ (x >> 3)

    def _gamma1(self, x: torch.Tensor) -> torch.Tensor:
        return self._rotr(x, 17) ^ self._rotr(x, 19) ^ (x >> 10)

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        """
        Compute SHA-256 hash.

        Args:
            message: (64,) bytes as int64 tensor (one 512-bit block)

        Returns:
            hash: (8,) 32-bit words as int64 tensor (256-bit hash)
        """
        # Parse message into 16 32-bit words
        W = torch.zeros(64, dtype=torch.int64, device=message.device)
        for i in range(16):
            W[i] = (
                (message[i * 4].long() << 24)
                | (message[i * 4 + 1].long() << 16)
                | (message[i * 4 + 2].long() << 8)
                | message[i * 4 + 3].long()
            )

        # Extend to 64 words
        for i in range(16, 64):
            W[i] = (
                self._gamma1(W[i - 2]) + W[i - 7] + self._gamma0(W[i - 15]) + W[i - 16]
            ) & 0xFFFFFFFF

        # Initialize working variables
        a, b, c, d, e, f, g, h = self.H0.clone()

        # Compression function main loop
        for i in range(64):
            T1 = (
                h + self._sigma1(e) + self._ch(e, f, g) + self.K[i] + W[i]
            ) & 0xFFFFFFFF
            T2 = (self._sigma0(a) + self._maj(a, b, c)) & 0xFFFFFFFF
            h = g
            g = f
            f = e
            e = (d + T1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (T1 + T2) & 0xFFFFFFFF

        # Compute final hash
        H = torch.stack(
            [
                (self.H0[0] + a) & 0xFFFFFFFF,
                (self.H0[1] + b) & 0xFFFFFFFF,
                (self.H0[2] + c) & 0xFFFFFFFF,
                (self.H0[3] + d) & 0xFFFFFFFF,
                (self.H0[4] + e) & 0xFFFFFFFF,
                (self.H0[5] + f) & 0xFFFFFFFF,
                (self.H0[6] + g) & 0xFFFFFFFF,
                (self.H0[7] + h) & 0xFFFFFFFF,
            ]
        )

        return H


# Problem configuration
def get_inputs():
    # One 512-bit block (64 bytes)
    message = torch.randint(0, 256, (64,), dtype=torch.int64)
    return [message]


def get_init_inputs():
    return []
