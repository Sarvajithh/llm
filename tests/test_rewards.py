"""Reward function unit tests; no generated code is executed."""

from src.rewards.constraints import constraint_compliance_reward
from src.rewards.math import extract_numeric_answer, numeric_answer_reward
from src.rewards.sandbox import SandboxResult, code_compilation_reward


class FakeSandbox:
    def validate(self, source: str, tests: str = "", timeout_seconds: int = 10) -> SandboxResult:
        return SandboxResult(passed="pass" in source)


def test_numeric_answer_reward() -> None:
    assert numeric_answer_reward(["The answer is 42."], [42]) == [1.0]


def test_extract_numeric_answer() -> None:
    assert extract_numeric_answer("x=3; final: 2.5") == 2.5


def test_constraint_reward() -> None:
    assert constraint_compliance_reward(["Use Python"], required=["python"]) == [1.0]


def test_code_reward_uses_injected_sandbox_only() -> None:
    assert code_compilation_reward(["pass"], sandbox=FakeSandbox()) == [1.0]
