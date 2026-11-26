"""Evaluation metrics independent of model inference frameworks."""

from __future__ import annotations

from collections.abc import Sequence


def exact_match(prediction: str, reference: str, *, case_sensitive: bool = False) -> float:
    """Normalized exact string match."""
    left, right = prediction.strip(), reference.strip()
    if not case_sensitive:
        left, right = left.casefold(), right.casefold()
    return float(left == right)


def pass_at_k(samples: Sequence[Sequence[bool]], k: int) -> float:
    """Compute empirical pass@k over per-problem boolean sample outcomes."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not samples:
        return 0.0
    values = [any(row[:k]) for row in samples]
    return sum(values) / len(values)
