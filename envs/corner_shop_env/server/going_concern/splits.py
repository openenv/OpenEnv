"""Seed splits. Train/dev seeds and held-out seeds live in disjoint ranges, so a held-out firm
is never generated during training. Held-out seeds are for final evaluation only."""
from __future__ import annotations

RANGES = {"train": (0, 900_000), "dev": (900_000, 1_000_000), "heldout": (1_000_000, 2_000_000)}


def split_of(seed: int) -> str:
    for name, (lo, hi) in RANGES.items():
        if lo <= seed < hi:
            return name
    return "other"


def seeds(split: str, n: int, offset: int = 0) -> list[int]:
    lo, hi = RANGES[split]
    if lo + offset + n > hi:
        raise ValueError("not enough seeds in split")
    return list(range(lo + offset, lo + offset + n))
