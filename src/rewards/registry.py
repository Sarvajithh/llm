"""Reward function lookup for training configurations."""

from functools import partial

from src.rewards.constraints import constraint_compliance_reward
from src.rewards.math import boxed_format_reward, numeric_answer_reward, symbolic_equivalence_reward
from src.rewards.sandbox import code_compilation_reward, unit_test_reward

REWARDS = {
    "numeric_answer": numeric_answer_reward,
    "symbolic_equivalence": symbolic_equivalence_reward,
    "boxed_format": boxed_format_reward,
    "constraint_compliance": constraint_compliance_reward,
    "code_compilation": code_compilation_reward,
    "unit_tests": unit_test_reward,
}


def resolve_rewards(names: list[str], sandbox=None, timeout_seconds: int = 10):
    """Return registered reward callables or reject unknown names."""
    unknown = set(names) - REWARDS.keys()
    if unknown:
        raise ValueError(f"Unknown reward functions: {sorted(unknown)}")
    resolved = []
    for name in names:
        function = REWARDS[name]
        if name in {"code_compilation", "unit_tests"}:
            function = partial(function, sandbox=sandbox, timeout_seconds=timeout_seconds)
        resolved.append(function)
    return resolved
