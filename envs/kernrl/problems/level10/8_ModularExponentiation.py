"""
Modular Exponentiation (Big Integer)

Computes base^exponent mod modulus for large integers.
Core operation in RSA, Diffie-Hellman, and other public-key cryptography.

Uses square-and-multiply algorithm:
result = 1
for each bit b in exponent (MSB to LSB):
    result = result^2 mod m
    if b == 1:
        result = result * base mod m

Optimization opportunities:
- Montgomery multiplication for fast mod
- Window-based exponentiation
- Parallel modular multiplications
- Barrett reduction
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Modular exponentiation for large integers.

    Simplified implementation using Python integers converted to tensors.
    Real GPU implementation would use multi-precision arithmetic.
    """

    def __init__(self, num_bits: int = 256):
        super(Model, self).__init__()
        self.num_bits = num_bits
        self.words_per_int = (num_bits + 63) // 64

    def _to_limbs(self, x: int, device) -> torch.Tensor:
        """Convert integer to tensor of 64-bit limbs."""
        limbs = torch.zeros(self.words_per_int, dtype=torch.int64, device=device)
        for i in range(self.words_per_int):
            limbs[i] = x & ((1 << 64) - 1)
            x >>= 64
        return limbs

    def _from_limbs(self, limbs: torch.Tensor) -> int:
        """Convert tensor of limbs back to integer."""
        result = 0
        for i in range(len(limbs) - 1, -1, -1):
            result = (result << 64) | int(limbs[i].item())
        return result

    def forward(
        self, base: torch.Tensor, exponent: torch.Tensor, modulus: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute base^exponent mod modulus.

        Args:
            base: (words_per_int,) base as 64-bit limbs
            exponent: (words_per_int,) exponent as 64-bit limbs
            modulus: (words_per_int,) modulus as 64-bit limbs

        Returns:
            result: (words_per_int,) result as 64-bit limbs
        """
        device = base.device

        # Convert to Python integers for computation
        # (Real GPU implementation would do this in parallel with multi-precision arithmetic)
        base_int = self._from_limbs(base)
        exp_int = self._from_limbs(exponent)
        mod_int = self._from_limbs(modulus)

        if mod_int == 0:
            return torch.zeros_like(base)

        # Square-and-multiply
        result = 1
        base_int = base_int % mod_int

        while exp_int > 0:
            if exp_int & 1:
                result = (result * base_int) % mod_int
            exp_int >>= 1
            base_int = (base_int * base_int) % mod_int

        return self._to_limbs(result, device)


# Problem configuration
num_bits = 256  # 256-bit integers
words_per_int = (num_bits + 63) // 64


def get_inputs():
    import random

    # Generate random large integers
    base_int = random.randint(2, 2**num_bits - 1)
    exp_int = random.randint(2, 2**num_bits - 1)
    mod_int = random.randint(2, 2**num_bits - 1)

    # Convert to limbs
    def to_limbs(x):
        limbs = []
        for _ in range(words_per_int):
            limbs.append(x & ((1 << 64) - 1))
            x >>= 64
        return torch.tensor(limbs, dtype=torch.int64)

    base = to_limbs(base_int)
    exponent = to_limbs(exp_int)
    modulus = to_limbs(mod_int)

    return [base, exponent, modulus]


def get_init_inputs():
    return [num_bits]
