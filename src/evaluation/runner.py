"""Common JSONL evaluation and model-comparison reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from src.evaluation.metrics import exact_match, pass_at_k
from src.rewards.math import extract_final_answer, extract_numeric_answer


class Generator(Protocol):
    """Minimal interface implemented by a local model adapter."""

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """Generate one completion per prompt."""
        ...


def evaluate_jsonl(generator: Generator, dataset_path: str, output_path: str,
                   max_new_tokens: int = 2048, samples_per_prompt: int = 1,
                   pass_k: list[int] | None = None,
                   metric: str = "exact") -> dict[str, float]:
    """Evaluate sampled completions per record and write per-example predictions."""
    rows = [json.loads(line) for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = [row for row in rows if row.get("reference") not in (None, "")]
    if not rows:
        raise ValueError(f"Evaluation split contains no rows with references: {dataset_path}")
    if metric not in {"exact", "math"}:
        raise ValueError("metric must be 'exact' or 'math'")
    prompts = [row["prompt"] for row in rows]
    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive")
    samples = [[] for _ in rows]
    for _ in range(samples_per_prompt):
        predictions = generator.generate(prompts, max_new_tokens)
        if len(predictions) != len(rows):
            raise ValueError("Generator returned a different number of completions than prompts")
        for index, prediction in enumerate(predictions):
            samples[index].append(prediction)
    def score(prediction: str, reference: str) -> float:
        if metric == "exact":
            return exact_match(prediction, reference)
        predicted_text = extract_final_answer(prediction).strip().strip("$").replace(" ", "")
        reference_text = extract_final_answer(reference).strip().strip("$").replace(" ", "")
        predicted_number = extract_numeric_answer(predicted_text)
        reference_number = extract_numeric_answer(reference_text)
        if predicted_number is not None and reference_number is not None:
            import math
            return float(math.isclose(predicted_number, reference_number,
                                      rel_tol=1e-6, abs_tol=1e-6))
        return exact_match(predicted_text, reference_text)

    outcomes = [[bool(score(prediction, str(row.get("reference", ""))))
                 for prediction in predictions] for row, predictions in zip(rows, samples, strict=True)]
    scores = [score(predictions[0], str(row.get("reference", "")))
              for row, predictions in zip(rows, samples, strict=True)]
    result = {"answer_accuracy": sum(scores) / max(1, len(scores)),
              "pass_at_1": pass_at_k(outcomes, 1)}
    for k in pass_k or []:
        result[f"pass_at_{k}"] = pass_at_k(outcomes, k)
    output = [{**row, "predictions": predictions, "answer_match": score}
              for row, predictions, score in zip(rows, samples, scores, strict=True)]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output), encoding="utf-8")
    return result
