"""Offline integration test for preparation components using temporary JSONL input."""

import json

from src.data.prepare import prepare_dataset


def test_prepare_pipeline_normalizes_splits_and_reports(tmp_path) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"prompt": "Q", "response": "A"}) + "\n", encoding="utf-8")
    config = {"seed": 2, "datasets": {"sft": [{"source": str(source), "format": "jsonl",
                                                 "schema": "prompt_response"}]},
              "filters": {}, "split": {"train": 1.0, "validation": 0.0, "test": 0.0}}
    report = prepare_dataset(config, "sft", str(tmp_path / "processed"))
    assert report["unique_records"] == 1
    rows = (tmp_path / "processed" / "sft_train.jsonl").read_text(encoding="utf-8")
    assert json.loads(rows)["response"] == "A"
