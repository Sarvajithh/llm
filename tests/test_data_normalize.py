"""Unit tests for canonical dataset schema normalization."""

import pytest

from src.data.normalize import DataValidationError, filter_record, normalize_record


def test_normalize_prompt_response_aliases() -> None:
    assert normalize_record({"instruction": "Question", "output": "Answer"}) == {
        "prompt": "Question", "response": "Answer"}


def test_normalize_preserves_reward_metadata() -> None:
    row = normalize_record({"prompt": "Q", "response": "teacher", "answer": "42", "tests": "assert 1"})
    assert row["answer"] == "42" and row["tests"] == "assert 1"


def test_normalize_messages_uses_last_assistant() -> None:
    record = {"messages": [{"role": "user", "content": "Q"},
                            {"role": "assistant", "content": "A1"},
                            {"role": "assistant", "content": "A2"}]}
    assert normalize_record(record, "messages")["response"] == "A2"


def test_invalid_record_is_rejected() -> None:
    with pytest.raises(DataValidationError):
        normalize_record({"prompt": ""}, "prompt_response")


def test_filter_rejects_oversized_record() -> None:
    assert not filter_record({"prompt": "p", "response": "r"}, max_chars=1)
