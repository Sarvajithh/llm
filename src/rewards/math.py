"""Mathematical answer extraction and verification helpers."""

from __future__ import annotations

import math
import re
from fractions import Fraction
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
    """Read a numeric final answer, including simple fractions and boxed LaTeX."""
    answer_text = extract_final_answer(text).replace(",", "").strip()
    fraction = re.fullmatch(r"\\?frac\s*\{?\s*(-?\d+)\s*\}?\s*\{?\s*(-?\d+)\s*\}?", answer_text)
    if fraction:
        try:
            return float(Fraction(int(fraction.group(1)), int(fraction.group(2))))
        except ZeroDivisionError:
            return None
    simple_fraction = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", answer_text)
    if simple_fraction:
        try:
            return float(Fraction(int(simple_fraction.group(1)), int(simple_fraction.group(2))))
        except ZeroDivisionError:
            return None
    matches = _NUMBER.findall(answer_text)
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
            expected_numeric = extract_numeric_answer(str(answer_value))
            expected = float(answer_value) if expected_numeric is None else expected_numeric
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
        predicted = _sympy_expression(extract_final_answer(_completion_text(completion)))
        expected = _sympy_expression(str(answer_value))
        try:
            scores.append(float(sympy.simplify(sympy.sympify(predicted) - sympy.sympify(expected)) == 0))
        except (sympy.SympifyError, TypeError, ValueError):
            scores.append(0.0)
    return scores


def boxed_format_reward(completions: list[str], **_: Any) -> list[float]:
    """Small formatting signal for a final answer enclosed in ``\\boxed{...}``."""
    scores: list[float] = []
    for completion in completions:
        text = _completion_text(completion)
        answer = extract_final_answer(text)
        scores.append(float(bool(answer) and bool(re.search(r"\\boxed\s*\{", text))))
    return scores


def extract_final_answer(text: str) -> str:
    """Extract the last boxed answer while supporting nested LaTeX braces."""
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        opening = text.find("{", start)
        depth = 0
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[opening + 1:index].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _sympy_expression(value: str) -> str:
    """Convert common final-answer LaTeX forms into a simple SymPy expression."""
    expression = value.strip().strip("$").replace("\\left", "").replace("\\right", "")
    expression = expression.replace("\\,", "").replace("\\!", "").replace("\\cdot", "*")
    expression = expression.replace("\\times", "*").replace("\\pi", "pi").replace("^", "**")
    expression = re.sub(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
                        r"((\1)/(\2))", expression)
    expression = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", expression)
    expression = expression.replace("\\", "")
    return expression
