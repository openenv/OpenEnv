"""
SHA-256 Hash - Batch Processing

Computes SHA-256 hashes for multiple messages in parallel.
Critical for cryptocurrency mining and batch verification.

Optimization opportunities:
- Parallel hashing across messages
- Coalesced memory access for message words
- Shared memory for constants
- Warp-level parallelism within hash
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Batch SHA-256 computation.

    Processes multiple 512-bit messages in parallel.
    """

    def __init__(self):
        super(Model, self).__init__()

        # SHA-256 constants
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

    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """
        Compute SHA-256 hashes for batch of messages.

        Args:
            messages: (B, 64) batch of 512-bit messages (bytes as int64)

        Returns:
            hashes: (B, 8) batch of 256-bit hashes (32-bit words as int64)
        """
        B = messages.shape[0]
        device = messages.device

        # Parse messages into 32-bit words: (B, 16)
        words = torch.zeros(B, 16, dtype=torch.int64, device=device)
        for i in range(16):
            words[:, i] = (
                (messages[:, i * 4].long() << 24)
                | (messages[:, i * 4 + 1].long() << 16)
                | (messages[:, i * 4 + 2].long() << 8)
                | messages[:, i * 4 + 3].long()
            )

        # Process each message (could be parallelized better)
        hashes = torch.zeros(B, 8, dtype=torch.int64, device=device)

        for b in range(B):
            W = torch.zeros(64, dtype=torch.int64, device=device)
            W[:16] = words[b]

            # Extend to 64 words
            for i in range(16, 64):
                s0 = (
                    ((W[i - 15] >> 7) | (W[i - 15] << 25))
                    ^ ((W[i - 15] >> 18) | (W[i - 15] << 14))
                    ^ (W[i - 15] >> 3)
                ) & 0xFFFFFFFF
                s1 = (
                    ((W[i - 2] >> 17) | (W[i - 2] << 15))
                    ^ ((W[i - 2] >> 19) | (W[i - 2] << 13))
                    ^ (W[i - 2] >> 10)
                ) & 0xFFFFFFFF
                W[i] = (W[i - 16] + s0 + W[i - 7] + s1) & 0xFFFFFFFF

            # Working variables
            a, b_, c, d, e, f, g, h = self.H0.clone()

            # 64 rounds
            for i in range(64):
                S1 = (
                    ((e >> 6) | (e << 26))
                    ^ ((e >> 11) | (e << 21))
                    ^ ((e >> 25) | (e << 7))
                ) & 0xFFFFFFFF
                ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
                temp1 = (h + S1 + ch + self.K[i] + W[i]) & 0xFFFFFFFF
                S0 = (
                    ((a >> 2) | (a << 30))
                    ^ ((a >> 13) | (a << 19))
                    ^ ((a >> 22) | (a << 10))
                ) & 0xFFFFFFFF
                maj = ((a & b_) ^ (a & c) ^ (b_ & c)) & 0xFFFFFFFF
                temp2 = (S0 + maj) & 0xFFFFFFFF

                h = g
                g = f
                f = e
                e = (d + temp1) & 0xFFFFFFFF
                d = c
                c = b_
                b_ = a
                a = (temp1 + temp2) & 0xFFFFFFFF

            hashes[b] = torch.stack(
                [
                    (self.H0[0] + a) & 0xFFFFFFFF,
                    (self.H0[1] + b_) & 0xFFFFFFFF,
                    (self.H0[2] + c) & 0xFFFFFFFF,
                    (self.H0[3] + d) & 0xFFFFFFFF,
                    (self.H0[4] + e) & 0xFFFFFFFF,
                    (self.H0[5] + f) & 0xFFFFFFFF,
                    (self.H0[6] + g) & 0xFFFFFFFF,
                    (self.H0[7] + h) & 0xFFFFFFFF,
                ]
            )

        return hashes


# Problem configuration
batch_size = 1024


def get_inputs():
    messages = torch.randint(0, 256, (batch_size, 64), dtype=torch.int64)
    return [messages]


def get_init_inputs():
    return []
