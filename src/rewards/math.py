"""Mathematical answer extraction and verification helpers."""

from __future__ import annotations

import math
import re
from typing import Any

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _completion_text(completion: Any) -> str:
    """Normalize a plain string or chat-style completion to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(str(message.get("content", "")) for message in completion
                          if isinstance(message, dict))
    return str(completion)


def extract_numeric_answer(text: str) -> float | None:
    """Extract the last numeric token, supporting a simple comma-formatted number."""
    matches = _NUMBER.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        value = float(matches[-1])
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def numeric_answer_reward(completions: list[str], answers: list[Any] | None = None,
                          answer: list[Any] | None = None,
                          tolerance: float = 1e-6, **_: Any) -> list[float]:
    """Reward numeric matches using relative and absolute tolerance."""
    answers = answers if answers is not None else answer
    if answers is None:
        return [0.0 for _ in completions]
    scores: list[float] = []
    for completion, answer_value in zip(completions, answers, strict=False):
        predicted = extract_numeric_answer(_completion_text(completion))
        try:
            expected = float(answer_value)
        except (TypeError, ValueError):
            scores.append(0.0)
            continue
        scores.append(float(predicted is not None and math.isclose(
            predicted, expected, rel_tol=tolerance, abs_tol=tolerance)))
    return scores


def symbolic_equivalence_reward(completions: list[str], answers: list[str] | None = None,
                                answer: list[str] | None = None, **_: Any) -> list[float]:
    """Compare simple symbolic expressions with SymPy when it is installed."""
    try:
        import sympy
    except ImportError as exc:
        raise RuntimeError("Install the optional 'math' dependencies for symbolic rewards") from exc
    answers = answers if answers is not None else answer
    if answers is None:
        return [0.0 for _ in completions]
    scores: list[float] = []
    for completion, answer_value in zip(completions, answers, strict=False):
        predicted = extract_final_answer(_completion_text(completion))
        try:
            scores.append(float(sympy.simplify(sympy.sympify(predicted) - sympy.sympify(answer_value)) == 0))
        except (sympy.SympifyError, TypeError, ValueError):
            scores.append(0.0)
    return scores


def extract_final_answer(text: str) -> str:
    """Use a boxed LaTeX answer if present, otherwise the final non-empty line."""
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""
