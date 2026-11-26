"""Unit tests for deterministic weighted mixture sampling."""

import pytest

from src.data.mixture import sample_mixture


def test_mixture_is_reproducible() -> None:
    data = {"a": [1, 2], "b": [3, 4]}
    weights = {"a": 1, "b": 2}
    assert sample_mixture(data, weights, 10, seed=7) == sample_mixture(data, weights, 10, seed=7)


def test_mixture_rejects_empty_positive_pool() -> None:
    with pytest.raises(ValueError):
        sample_mixture({"a": []}, {"a": 1}, 1)
