"""Deterministic randomness. Every stream is derived from (seed, tier, version, name...)
with SHA-256, so an episode is reproducible from those values on any platform."""
from __future__ import annotations

import hashlib
import math
import random


def derive_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


def make_rng(*parts) -> random.Random:
    return random.Random(derive_seed(*parts))


def poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam <= 60:
        # Knuth
        limit = math.exp(-lam)
        k, p = 0, 1.0
        while True:
            p *= rng.random()
            if p <= limit:
                return k
            k += 1
    return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))


def stoch_round(rng: random.Random, x: float) -> int:
    """Round x to an int, up with probability frac(x). Unbiased."""
    if x <= 0:
        return 0
    f = math.floor(x)
    return int(f) + (1 if rng.random() < (x - f) else 0)
