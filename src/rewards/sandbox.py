"""Sandbox reward adapters; this module never executes submitted code itself."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class SandboxResult:
    """Normalized result returned by a separately configured isolated runner."""

    passed: bool
    output: str = ""
    error: str = ""


class IsolatedSandbox(Protocol):
    """Required boundary for remote/containerized code validation services."""

    def validate(self, source: str, tests: str = "", timeout_seconds: int = 10) -> SandboxResult:
        """Compile/run in a sandbox with filesystem, process, resource and network limits."""
        ...


def _sandbox_score(completions: list[str], tests: list[str], sandbox: IsolatedSandbox | None,
                   timeout_seconds: int = 10) -> list[float]:
    if sandbox is None:
        raise RuntimeError("Code reward needs an explicitly configured isolated sandbox backend")
    scores: list[float] = []
    for index, completion in enumerate(completions):
        code = ("\n".join(str(message.get("content", "")) for message in completion
                            if isinstance(message, dict)) if isinstance(completion, list)
                else str(completion))
        test = tests[index] if index < len(tests) else ""
        scores.append(float(sandbox.validate(code, str(test), timeout_seconds).passed))
    return scores


def code_compilation_reward(completions: list[str], sandbox: IsolatedSandbox | None = None,
                            timeout_seconds: int = 10, **_: object) -> list[float]:
    """Score syntax/compilation only, by asking an isolated sandbox backend."""
    return _sandbox_score(completions, [""] * len(completions), sandbox, timeout_seconds)


def unit_test_reward(completions: list[str], tests: list[str] | None = None,
                     sandbox: IsolatedSandbox | None = None,
                     timeout_seconds: int = 10, **_: object) -> list[float]:
    """Score unit-test results through an isolated sandbox backend."""
    return _sandbox_score(completions, tests or [""] * len(completions), sandbox, timeout_seconds)
