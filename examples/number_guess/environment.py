"""Example: number-guess domain — environment fill.

Overrides two methods from the template Environment:
    _execute_action    — mutate hidden low/high based on guess feedback
    _build_observation — surface last comparison result to the agent
    _check_termination — end on correct guess (not just max_steps)
"""

from __future__ import annotations

import random
from typing import Optional

from models import Action, Observation, RewardConfig, RuleViolation
from server.environment import Environment as _BaseEnvironment
from server.tasks.generator import TaskGenerator

# Example-specific overlays
from examples.number_guess.scenarios import SCENARIOS
from examples.number_guess.rules import NumberGuessRules
from examples.number_guess.rewards import NumberGuessRewards


class NumberGuessEnvironment(_BaseEnvironment):

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        domain_randomise: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            reward_config=reward_config,
            domain_randomise=domain_randomise,
            seed=seed,
        )
        # Swap plugins for domain-specific ones
        self._rules = NumberGuessRules()
        self._reward = NumberGuessRewards(self.reward_config)

        # Override task generator to use our scenarios
        self._tasks = _NumberGuessTaskGen(seed=seed)

        self._last_feedback: str = ""

    # ------------------------------------------------------------------ state transitions

    def _execute_action(self, action: Action) -> None:
        if self._state is None:
            return
        truth = self._state.hidden_truth
        target = truth.get("target")

        if action.command == "guess":
            val = action.args.get("value")
            if val == target:
                self._last_feedback = "correct"
            elif val < target:
                self._last_feedback = "higher"
                truth["low"] = max(truth["low"], val + 1)
            else:
                self._last_feedback = "lower"
                truth["high"] = min(truth["high"], val - 1)
        elif action.command == "submit":
            # Allow explicit submit, but only valid if last guess was correct
            if self._last_feedback != "correct":
                self._last_feedback = "submit_without_correct_guess"

    def _build_observation(
        self,
        reward: float,
        violations: list[RuleViolation],
    ) -> Observation:
        obs = super()._build_observation(reward=reward, violations=violations)
        if self._state is not None:
            truth = self._state.hidden_truth
            obs.summary = (
                f"Step {self._state.step}/{self._state.max_steps} · "
                f"range=[{truth.get('low')}, {truth.get('high')}] · "
                f"last={self._last_feedback or '(no guess yet)'}"
            )
            obs.latest_output = {
                "feedback": self._last_feedback,
                "current_range": [truth.get("low"), truth.get("high")],
            }
        return obs

    def _check_termination(self) -> bool:
        # End immediately on correct guess (don't waste steps)
        if self._last_feedback == "correct":
            return True
        return super()._check_termination()


class _NumberGuessTaskGen(TaskGenerator):
    """Task generator that picks from number-guess SCENARIOS and seeds a target."""

    def generate(self, scenario_name=None, domain_randomise=False):  # type: ignore[override]
        if scenario_name:
            base = next((s for s in SCENARIOS if s.task_id == scenario_name), None)
            if base is None:
                raise ValueError(f"Unknown scenario: {scenario_name}")
        else:
            base = self._rng.choice(SCENARIOS)

        # Populate hidden state: pick a target in the range
        low, high = base.expected_findings["range"]
        target = self._rng.randint(low, high)

        # Cache the hidden truth for the environment to pull on reset
        result = base.model_copy()
        result.expected_findings = dict(result.expected_findings)
        result.expected_findings["_target"] = target  # env reads this on reset
        result.expected_findings["_initial_range"] = high - low + 1
        return result
