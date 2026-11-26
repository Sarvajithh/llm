"""Reward for explicit response-format and task constraints."""

from __future__ import annotations

from typing import Any


def constraint_compliance_reward(completions: list[str], required: list[Any] | None = None,
                                  forbidden: list[Any] | None = None, **_: Any) -> list[float]:
    """Score each completion as the fraction of configured constraints satisfied."""
    required, forbidden = required or [], forbidden or []
    scores: list[float] = []
    for index, completion in enumerate(completions):
        text = ("\n".join(str(message.get("content", "")) for message in completion
                           if isinstance(message, dict)) if isinstance(completion, list)
                else str(completion))
        req = required[index] if index < len(required) and isinstance(required[index], list) else required
        forb = forbidden[index] if index < len(forbidden) and isinstance(forbidden[index], list) else forbidden
        total = len(req) + len(forb)
        scores.append(1.0 if not total else
            (sum(str(item).casefold() in text.casefold() for item in req)
             + sum(str(item).casefold() not in text.casefold() for item in forb)) / total)
    return scores
