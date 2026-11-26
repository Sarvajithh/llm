"""Explicit dataset readers. Remote loading occurs only when the CLI is invoked."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_records(source: str, fmt: str, split: str = "train", *,
                 subset: str | None = None, max_records: int | None = None,
                 seed: int = 42, where: dict[str, Any] | None = None,
                 shuffle_buffer: int = 10_000) -> list[dict[str, Any]]:
    """Load a bounded, deterministic subset of local or Hub data on explicit request."""
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    if fmt == "jsonl":
        path = Path(source)
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif fmt == "parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(source).to_pylist()
    elif fmt == "huggingface":
        from datasets import load_dataset
        dataset = load_dataset(source, name=subset, split=split, streaming=True)
        if where:
            dataset = dataset.filter(lambda row: all(row.get(key) == value
                                                     for key, value in where.items()))
        dataset = dataset.shuffle(seed=seed, buffer_size=max(1, shuffle_buffer))
        rows = []
        for row in dataset:
            rows.append(dict(row))
            if max_records is not None and len(rows) >= max_records:
                break
        return rows
    else:
        raise ValueError(f"Unsupported dataset format: {fmt}")

    if where:
        rows = [row for row in rows if all(row.get(key) == value
                                           for key, value in where.items())]
    import random
    random.Random(seed).shuffle(rows)
    return rows[:max_records] if max_records is not None else rows
