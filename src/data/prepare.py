"""Dataset preparation orchestration and preprocessing reports."""

from __future__ import annotations

import json
import random
import hashlib
from pathlib import Path
from typing import Any

from src.data.loader import load_records
from src.data.mixture import sample_mixture
from src.data.normalize import DataValidationError, filter_record, normalize_record


def prepare_dataset(config: dict[str, Any], group: str, output_dir: str) -> dict[str, Any]:
    """Load, normalize, filter, deduplicate, split, and persist one configured dataset group."""
    specs = config.get("datasets", {}).get(group)
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"No dataset entries configured for group {group!r}")
    filters = config.get("filters", {})
    pools: dict[str, list[dict[str, Any]]] = {}
    weights: dict[str, float] = {}
    dropped_invalid = dropped_filter = duplicates = 0
    raw_count = 0
    for index, spec in enumerate(specs):
        name = str(spec.get("name", f"dataset_{index}"))
        pool: list[dict[str, Any]] = []
        weights[name] = float(spec.get("weight", 1.0))
        raw = load_records(
            spec["source"], spec["format"], spec.get("split", "train"),
            subset=spec.get("subset"), max_records=spec.get("max_samples"),
            seed=int(config.get("seed", 42)), where=spec.get("where"),
            shuffle_buffer=int(spec.get("shuffle_buffer", 10_000)))
        raw_count += len(raw)
        for item in raw:
            try:
                normalized = normalize_record(item, spec.get("schema", "auto"))
            except DataValidationError:
                dropped_invalid += 1
                continue
            if not filter_record(
                    normalized,
                    min_prompt_chars=int(spec.get("min_prompt_chars", filters.get("min_prompt_chars", 1))),
                    min_response_chars=int(spec.get("min_response_chars", filters.get("min_response_chars", 1))),
                    max_chars=int(spec.get("max_chars", filters.get("max_chars", 100_000)))):
                dropped_filter += 1
                continue
            pool.append(normalized)
        pools[name] = pool
    nonempty = [rows for rows in pools.values() if rows]
    requested_size = int(config.get("max_total_samples_by_group", {}).get(
        group, config.get("max_total_samples", sum(len(rows) for rows in nonempty))))
    records = [row for _, row in sample_mixture(pools, weights, requested_size,
                                                 seed=int(config.get("seed", 42)))] if requested_size else []
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in records:
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(row)
    # Keep all responses for a prompt in one partition. This prevents alternative
    # Nemotron traces or preference pairs for the same problem leaking into eval.
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in unique:
        prompt = " ".join(str(row.get("prompt", "")).split()).casefold()
        group_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        groups.setdefault(group_key, []).append(row)
    group_keys = list(groups)
    random.Random(int(config.get("seed", 42))).shuffle(group_keys)
    split = config.get("split", {"train": 1.0})
    boundaries = [int(len(group_keys) * split.get("train", 0)),
                  int(len(group_keys) * (split.get("train", 0) + split.get("validation", 0)))]
    train_keys = set(group_keys[:boundaries[0]])
    validation_keys = set(group_keys[boundaries[0]:boundaries[1]])
    partitions = {"train": [], "validation": [], "test": []}
    for key, rows in groups.items():
        target = "train" if key in train_keys else "validation" if key in validation_keys else "test"
        partitions[target].extend(rows)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in partitions.items():
        (root / f"{group}_{name}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    prompt_lengths = [len(str(row.get("prompt", ""))) for row in unique]
    response_lengths = [len(str(row.get(key, ""))) for row in unique
                        for key in ("response", "chosen", "reference") if key in row]
    report = {"group": group, "input_records": raw_count, "sampled_records": len(records),
              "unique_records": len(unique), "dropped_invalid": dropped_invalid,
              "dropped_by_filter": dropped_filter, "duplicates": duplicates,
              "unique_problem_groups": len(groups),
              "statistics": {"mean_prompt_chars": round(sum(prompt_lengths) / max(1, len(prompt_lengths)), 2),
                             "max_prompt_chars": max(prompt_lengths, default=0),
                             "mean_response_chars": round(sum(response_lengths) / max(1, len(response_lengths)), 2),
                             "max_response_chars": max(response_lengths, default=0)},
              "split_counts": {key: len(value) for key, value in partitions.items()}}
    (root / f"{group}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
