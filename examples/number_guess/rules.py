"""Example: number-guess domain — rule engine fill.

Domain rules:
    HARD  — guess value must be an int
    HARD  — guess value must be within current [low, high]
    SOFT  — repeating a previously-tried guess (redundancy)
"""

from __future__ import annotations

from models import Action, RuleViolation, State, TaskSpec
from server.rules.engine import Rule, RuleEngine


def guess_value_is_int(action: Action, state: State, task: TaskSpec) -> list[RuleViolation]:
    if action.command == "guess":
        val = action.args.get("value")
        if not isinstance(val, int):
            return [RuleViolation(
                severity="hard",
                category="domain",
                message=f"guess.args.value must be int, got {type(val).__name__}",
            )]
    return []


def guess_within_range(action: Action, state: State, task: TaskSpec) -> list[RuleViolation]:
    if action.command == "guess":
        val = action.args.get("value")
        if not isinstance(val, int):
            return []  # covered by previous rule
        low = state.hidden_truth.get("low", 1)
        high = state.hidden_truth.get("high", 10000)
        if val < low or val > high:
            return [RuleViolation(
                severity="hard",
                category="prerequisite",
                message=f"guess {val} out of current range [{low}, {high}]",
            )]
    return []


def no_repeated_guesses(action: Action, state: State, task: TaskSpec) -> list[RuleViolation]:
    if action.command == "guess":
        val = action.args.get("value")
        prior = {s.get("action", {}).get("args", {}).get("value") for s in state.step_history}
        if val in prior:
            return [RuleViolation(
                severity="soft",
                category="redundancy",
                message=f"guess {val} was already tried",
            )]
    return []


class NumberGuessRules(RuleEngine):
    """Extends the template's base rules with domain-specific ones."""

    def _domain_rules(self) -> list[Rule]:
        return [guess_value_is_int, guess_within_range, no_repeated_guesses]
