"""Explicit dataset readers. Remote loading occurs only when the CLI is invoked."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_records(source: str, fmt: str, split: str = "train") -> list[dict[str, Any]]:
    """Load JSONL/Parquet local files or a Hugging Face dataset on explicit request."""
    if fmt == "jsonl":
        path = Path(source)
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if fmt == "parquet":
        import pyarrow.parquet as pq
        return pq.read_table(source).to_pylist()
    if fmt == "huggingface":
        from datasets import load_dataset
        return list(load_dataset(source, split=split))
    raise ValueError(f"Unsupported dataset format: {fmt}")
