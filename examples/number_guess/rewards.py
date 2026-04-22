"""Example: number-guess domain — reward computer fill.

Components (all in [0, 1]):
    correctness  — 1.0 when the last guess was correct, else 0.0
    efficiency   — budget remaining / max_steps (penalizes step spam)
    quality      — search-space reduction: (initial_range - current_range) / initial_range
    safety       — 1.0 always (no "breaking" state in this domain)

Using multiplicative mode: zero correctness → zero reward until correct guess.
"""

from __future__ import annotations

from models import Action, State, TaskSpec
from server.rewards.reward import RewardComputer


class NumberGuessRewards(RewardComputer):

    def _component_scores(
        self,
        action: Action,
        state: State,
        task: TaskSpec,
    ) -> dict[str, float]:
        truth = state.hidden_truth
        low = truth.get("low", 1)
        high = truth.get("high", 10000)
        initial_range = truth.get("initial_range", high - low + 1)
        current_range = max(1, high - low + 1)

        # Correctness — did the last guess match?
        last = state.step_history[-1] if state.step_history else None
        correctness = 0.0
        if last:
            val = last.get("action", {}).get("args", {}).get("value")
            target = truth.get("target")
            if val == target:
                correctness = 1.0

        # Efficiency — steps remaining
        efficiency = max(0.0, 1.0 - state.step / max(1, task.max_steps))

        # Quality — how much of the search space has been eliminated
        quality = 1.0 - (current_range / initial_range)

        return {
            "correctness": correctness,
            "efficiency": efficiency,
            "quality": quality,
            "safety": 1.0,  # no harmful side-effects in this domain
        }

    def _regression_penalty(
        self,
        action: Action,
        state: State,
        task: TaskSpec,
    ) -> float:
        # Regression = widening the known range. Impossible in binary search
        # with sane rules, but would flag an off-by-one bug in environment.py.
        prior = state.step_history[-2:] if len(state.step_history) >= 2 else []
        if len(prior) < 2:
            return 0.0
        prev_low = prior[-2].get("observation", {}).get("budget_remaining", None)
        # Actual tracking happens in the Environment override — this is a
        # placeholder showing the hook exists.
        return 0.0
