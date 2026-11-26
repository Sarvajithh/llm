"""Deterministic weighted sampling helpers for dataset mixtures."""

from __future__ import annotations

import random
import math
from collections.abc import Mapping, Sequence
from typing import TypeVar

T = TypeVar("T")


def sample_mixture(datasets: Mapping[str, Sequence[T]], weights: Mapping[str, float],
                   size: int, seed: int = 42) -> list[tuple[str, T]]:
    """Select up to ``size`` records without replacement, favoring weighted sources."""
    if size < 0:
        raise ValueError("size must be non-negative")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Dataset weights must be non-negative")
    names = [name for name, rows in datasets.items() if rows and weights.get(name, 0) > 0]
    if size and not names:
        raise ValueError("No non-empty datasets have positive weights")
    rng = random.Random(seed)
    candidates: list[tuple[float, str, T]] = []
    for name in names:
        weight = float(weights[name])
        for row in datasets[name]:
            # Exponential-race keys provide weighted sampling without duplicates.
            key = -math.log(max(rng.random(), 1e-15)) / weight
            candidates.append((key, name, row))
    candidates.sort(key=lambda item: item[0])
    return [(name, row) for _, name, row in candidates[:min(size, len(candidates))]]
