"""Normalize common conversational and preference dataset schemas."""

from __future__ import annotations

from typing import Any


class DataValidationError(ValueError):
    """Raised for malformed or unsupported dataset records."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{field} must be a non-empty string")
    return value.strip()


def normalize_record(record: dict[str, Any], schema: str = "auto") -> dict[str, Any]:
    """Normalize supported source schemas into canonical prompt/response fields.

    Supported schemas: ``prompt_response``, ``messages``,
    ``prompt_chosen_rejected``, and ``prompt_reference``.
    """
    if not isinstance(record, dict):
        raise DataValidationError("Each record must be a mapping")
    if schema == "auto":
        schema = ("messages" if "messages" in record else
                  "prompt_chosen_rejected" if {"prompt", "chosen", "rejected"} <= record.keys() else
                  "prompt_reference" if {"prompt", "reference"} <= record.keys() else
                  "prompt_response")
    if schema == "messages":
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise DataValidationError("messages must be a list")
        if not all(isinstance(message, dict) for message in messages):
            raise DataValidationError("Each messages entry must be a mapping")
        user = next((m.get("content") for m in messages if m.get("role") == "user"), None)
        assistant = next((m.get("content") for m in reversed(messages) if m.get("role") == "assistant"), None)
        return {"prompt": _text(user, "user message"), "response": _text(assistant, "assistant message")}
    prompt = _text(record.get("prompt", record.get("instruction")), "prompt")
    if schema == "prompt_response":
        response = record.get("response", record.get("output", record.get("completion")))
        normalized = {"prompt": prompt, "response": _text(response, "response")}
        for key in ("answer", "reference", "tests", "required", "forbidden"):
            if key in record:
                normalized[key] = record[key]
        return normalized
    if schema == "prompt_chosen_rejected":
        return {"prompt": prompt, "chosen": _text(record.get("chosen"), "chosen"),
                "rejected": _text(record.get("rejected"), "rejected")}
    if schema == "prompt_reference":
        return {"prompt": prompt, "reference": _text(record.get("reference"), "reference")}
    raise DataValidationError(f"Unsupported schema: {schema}")


def filter_record(record: dict[str, Any], *, min_prompt_chars: int = 1,
                  min_response_chars: int = 1, max_chars: int = 100_000) -> bool:
    """Return whether a normalized record meets configured length constraints."""
    prompt = record.get("prompt", "")
    answers = [record.get(k, "") for k in ("response", "chosen", "rejected", "reference") if k in record]
    return (isinstance(prompt, str) and len(prompt) >= min_prompt_chars
            and all(isinstance(a, str) and len(a) >= min_response_chars for a in answers)
            and len(prompt) + sum(map(len, answers)) <= max_chars)
