"""Deterministic weighted sampling helpers for dataset mixtures."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TypeVar

T = TypeVar("T")


def sample_mixture(datasets: Mapping[str, Sequence[T]], weights: Mapping[str, float],
                   size: int, seed: int = 42) -> list[tuple[str, T]]:
    """Sample ``size`` records with replacement according to dataset weights."""
    if size < 0:
        raise ValueError("size must be non-negative")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Dataset weights must be non-negative")
    names = [name for name, rows in datasets.items() if rows and weights.get(name, 0) > 0]
    if size and not names:
        raise ValueError("No non-empty datasets have positive weights")
    probs = [float(weights[name]) for name in names]
    rng = random.Random(seed)
    selected: list[tuple[str, T]] = []
    for _ in range(size):
        name = rng.choices(names, weights=probs, k=1)[0]
        selected.append((name, rng.choice(datasets[name])))
    return selected
