"""
PBKDF2 Key Derivation

Password-Based Key Derivation Function 2.
Derives cryptographic keys from passwords with salt and iteration count.

Used for secure password storage and key generation.

DK = PBKDF2(Password, Salt, c, dkLen)
where c is iteration count (high for security).

Optimization opportunities:
- Parallel HMAC computation
- Unrolled inner loops
- Shared memory for intermediate hashes
- Multiple derived blocks in parallel
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    PBKDF2-HMAC-SHA256 key derivation.

    Simplified implementation for kernel optimization practice.
    """

    def __init__(self, iterations: int = 1000, dk_len: int = 32):
        super(Model, self).__init__()
        self.iterations = iterations
        self.dk_len = dk_len

    def _xor(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """XOR two byte tensors."""
        return a ^ b

    def _simple_hmac(self, key: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Simplified HMAC (not cryptographically secure - for demo)."""
        # Real HMAC-SHA256 would be: H(key ^ opad || H(key ^ ipad || message))
        # This is a placeholder that produces consistent output
        result = torch.zeros(32, dtype=torch.int64, device=key.device)

        # Mix key and message
        combined = torch.cat([key, message])
        for i in range(len(combined)):
            result[i % 32] = (result[i % 32] * 31 + combined[i]) & 0xFF

        # Additional mixing
        for _ in range(4):
            for i in range(32):
                result[i] = (
                    result[i] ^ result[(i + 17) % 32] + result[(i + 11) % 32]
                ) & 0xFF

        return result

    def forward(self, password: torch.Tensor, salt: torch.Tensor) -> torch.Tensor:
        """
        Derive key from password using PBKDF2.

        Args:
            password: (P,) password bytes
            salt: (S,) salt bytes

        Returns:
            derived_key: (dk_len,) derived key bytes
        """
        device = password.device

        # Number of blocks needed
        num_blocks = (self.dk_len + 31) // 32

        derived_key = torch.zeros(num_blocks * 32, dtype=torch.int64, device=device)

        for block_idx in range(num_blocks):
            # First iteration: U_1 = PRF(Password, Salt || INT(i))
            block_num = torch.tensor(
                [0, 0, 0, block_idx + 1], dtype=torch.int64, device=device
            )
            U = self._simple_hmac(password, torch.cat([salt, block_num]))

            # Accumulator
            F = U.clone()

            # Remaining iterations: U_j = PRF(Password, U_{j-1})
            for _ in range(self.iterations - 1):
                U = self._simple_hmac(password, U)
                F = self._xor(F, U)

            # Store block
            derived_key[block_idx * 32 : (block_idx + 1) * 32] = F

        return derived_key[: self.dk_len]


# Problem configuration
def get_inputs():
    password = torch.randint(0, 256, (16,), dtype=torch.int64)  # 16-byte password
    salt = torch.randint(0, 256, (16,), dtype=torch.int64)  # 16-byte salt
    return [password, salt]


def get_init_inputs():
    return [1000, 32]  # iterations, dk_len
