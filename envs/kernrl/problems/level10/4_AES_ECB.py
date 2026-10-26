"""
AES-128 ECB Encryption

Encrypts data using AES-128 in ECB mode (for simplicity).
Note: ECB is insecure for real use; this is for kernel optimization practice.

AES operates on 16-byte blocks through:
1. SubBytes - S-box substitution
2. ShiftRows - row rotation
3. MixColumns - column mixing
4. AddRoundKey - XOR with round key

Optimization opportunities:
- T-table implementation (combined operations)
- Parallel block processing
- Shared memory for S-box/T-tables
- Bitsliced implementation
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    AES-128 ECB encryption.
    """

    def __init__(self):
        super(Model, self).__init__()

        # AES S-box (substitution box)
        SBOX = [
            0x63,
            0x7C,
            0x77,
            0x7B,
            0xF2,
            0x6B,
            0x6F,
            0xC5,
            0x30,
            0x01,
            0x67,
            0x2B,
            0xFE,
            0xD7,
            0xAB,
            0x76,
            0xCA,
            0x82,
            0xC9,
            0x7D,
            0xFA,
            0x59,
            0x47,
            0xF0,
            0xAD,
            0xD4,
            0xA2,
            0xAF,
            0x9C,
            0xA4,
            0x72,
            0xC0,
            0xB7,
            0xFD,
            0x93,
            0x26,
            0x36,
            0x3F,
            0xF7,
            0xCC,
            0x34,
            0xA5,
            0xE5,
            0xF1,
            0x71,
            0xD8,
            0x31,
            0x15,
            0x04,
            0xC7,
            0x23,
            0xC3,
            0x18,
            0x96,
            0x05,
            0x9A,
            0x07,
            0x12,
            0x80,
            0xE2,
            0xEB,
            0x27,
            0xB2,
            0x75,
            0x09,
            0x83,
            0x2C,
            0x1A,
            0x1B,
            0x6E,
            0x5A,
            0xA0,
            0x52,
            0x3B,
            0xD6,
            0xB3,
            0x29,
            0xE3,
            0x2F,
            0x84,
            0x53,
            0xD1,
            0x00,
            0xED,
            0x20,
            0xFC,
            0xB1,
            0x5B,
            0x6A,
            0xCB,
            0xBE,
            0x39,
            0x4A,
            0x4C,
            0x58,
            0xCF,
            0xD0,
            0xEF,
            0xAA,
            0xFB,
            0x43,
            0x4D,
            0x33,
            0x85,
            0x45,
            0xF9,
            0x02,
            0x7F,
            0x50,
            0x3C,
            0x9F,
            0xA8,
            0x51,
            0xA3,
            0x40,
            0x8F,
            0x92,
            0x9D,
            0x38,
            0xF5,
            0xBC,
            0xB6,
            0xDA,
            0x21,
            0x10,
            0xFF,
            0xF3,
            0xD2,
            0xCD,
            0x0C,
            0x13,
            0xEC,
            0x5F,
            0x97,
            0x44,
            0x17,
            0xC4,
            0xA7,
            0x7E,
            0x3D,
            0x64,
            0x5D,
            0x19,
            0x73,
            0x60,
            0x81,
            0x4F,
            0xDC,
            0x22,
            0x2A,
            0x90,
            0x88,
            0x46,
            0xEE,
            0xB8,
            0x14,
            0xDE,
            0x5E,
            0x0B,
            0xDB,
            0xE0,
            0x32,
            0x3A,
            0x0A,
            0x49,
            0x06,
            0x24,
            0x5C,
            0xC2,
            0xD3,
            0xAC,
            0x62,
            0x91,
            0x95,
            0xE4,
            0x79,
            0xE7,
            0xC8,
            0x37,
            0x6D,
            0x8D,
            0xD5,
            0x4E,
            0xA9,
            0x6C,
            0x56,
            0xF4,
            0xEA,
            0x65,
            0x7A,
            0xAE,
            0x08,
            0xBA,
            0x78,
            0x25,
            0x2E,
            0x1C,
            0xA6,
            0xB4,
            0xC6,
            0xE8,
            0xDD,
            0x74,
            0x1F,
            0x4B,
            0xBD,
            0x8B,
            0x8A,
            0x70,
            0x3E,
            0xB5,
            0x66,
            0x48,
            0x03,
            0xF6,
            0x0E,
            0x61,
            0x35,
            0x57,
            0xB9,
            0x86,
            0xC1,
            0x1D,
            0x9E,
            0xE1,
            0xF8,
            0x98,
            0x11,
            0x69,
            0xD9,
            0x8E,
            0x94,
            0x9B,
            0x1E,
            0x87,
            0xE9,
            0xCE,
            0x55,
            0x28,
            0xDF,
            0x8C,
            0xA1,
            0x89,
            0x0D,
            0xBF,
            0xE6,
            0x42,
            0x68,
            0x41,
            0x99,
            0x2D,
            0x0F,
            0xB0,
            0x54,
            0xBB,
            0x16,
        ]
        self.register_buffer("sbox", torch.tensor(SBOX, dtype=torch.int64))

        # Round constants
        RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]
        self.register_buffer("rcon", torch.tensor(RCON, dtype=torch.int64))

    def _sub_bytes(self, state: torch.Tensor) -> torch.Tensor:
        """Apply S-box substitution."""
        return self.sbox[state.long()]

    def _shift_rows(self, state: torch.Tensor) -> torch.Tensor:
        """Shift rows of state matrix."""
        # state is (4, 4) - rows are shifted by 0, 1, 2, 3 positions
        result = state.clone()
        result[1] = torch.roll(state[1], -1)
        result[2] = torch.roll(state[2], -2)
        result[3] = torch.roll(state[3], -3)
        return result

    def _xtime(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply by x in GF(2^8)."""
        return ((x << 1) ^ (((x >> 7) & 1) * 0x1B)) & 0xFF

    def _mix_column(self, col: torch.Tensor) -> torch.Tensor:
        """Mix one column."""
        t = col[0] ^ col[1] ^ col[2] ^ col[3]
        result = torch.zeros(4, dtype=col.dtype, device=col.device)
        result[0] = (col[0] ^ t ^ self._xtime(col[0] ^ col[1])) & 0xFF
        result[1] = (col[1] ^ t ^ self._xtime(col[1] ^ col[2])) & 0xFF
        result[2] = (col[2] ^ t ^ self._xtime(col[2] ^ col[3])) & 0xFF
        result[3] = (col[3] ^ t ^ self._xtime(col[3] ^ col[0])) & 0xFF
        return result

    def _mix_columns(self, state: torch.Tensor) -> torch.Tensor:
        """Apply MixColumns transformation."""
        result = torch.zeros_like(state)
        for i in range(4):
            result[:, i] = self._mix_column(state[:, i])
        return result

    def _add_round_key(
        self, state: torch.Tensor, round_key: torch.Tensor
    ) -> torch.Tensor:
        """XOR state with round key."""
        return state ^ round_key

    def forward(self, plaintext: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Encrypt plaintext block with AES-128.

        Args:
            plaintext: (16,) 16-byte block
            key: (16,) 16-byte key

        Returns:
            ciphertext: (16,) encrypted block
        """
        device = plaintext.device

        # Key expansion (simplified - generates 11 round keys)
        round_keys = torch.zeros(11, 4, 4, dtype=torch.int64, device=device)
        round_keys[0] = key.reshape(4, 4).T

        for i in range(1, 11):
            prev = round_keys[i - 1]
            temp = prev[:, 3].clone()
            # RotWord
            temp = torch.roll(temp, -1)
            # SubWord
            temp = self.sbox[temp.long()]
            # Add Rcon
            temp[0] = temp[0] ^ self.rcon[i - 1]
            # Generate round key
            round_keys[i, :, 0] = prev[:, 0] ^ temp
            for j in range(1, 4):
                round_keys[i, :, j] = round_keys[i, :, j - 1] ^ prev[:, j]

        # Initial state
        state = plaintext.reshape(4, 4).T.clone()

        # Initial round
        state = self._add_round_key(state, round_keys[0])

        # Main rounds (1-9)
        for r in range(1, 10):
            state = self._sub_bytes(state)
            state = self._shift_rows(state)
            state = self._mix_columns(state)
            state = self._add_round_key(state, round_keys[r])

        # Final round (no MixColumns)
        state = self._sub_bytes(state)
        state = self._shift_rows(state)
        state = self._add_round_key(state, round_keys[10])

        return state.T.flatten()


# Problem configuration
def get_inputs():
    plaintext = torch.randint(0, 256, (16,), dtype=torch.int64)
    key = torch.randint(0, 256, (16,), dtype=torch.int64)
    return [plaintext, key]


def get_init_inputs():
    return []
