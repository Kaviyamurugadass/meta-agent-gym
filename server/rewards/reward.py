"""Reward computer — multi-component rewards with RLVR approach.

META-AGENT EXTENSION:
    - Uses HYBRID mode with hard verifiers as gates
    - Multiple independent reward functions (RLVR)
    - Anti-hacking penalties for common exploits
    - Decomposed reward breakdown for observation

The reward system tracks ALL components separately for GRPO variance.
"""

from __future__ import annotations

import logging

from models import (
    Action,
    ActionCommand,
    AgentSpec,
    ModelType,
    RewardConfig,
    RewardMode,
    RuleViolation,
    State,
    TaskSpec,
)

try:
    from server.verifiers import HardVerifiers
except ImportError:
    HardVerifiers = None  # type: ignore[assignment,misc]

logger = logging.getLogger("server.rewards.reward")


class MetaAgentRewardComputer:
    """Compute per-step scalar reward with multi-component breakdown.

    RLVR Approach:
        1. Hard verifiers (100% of steps, free) — YAML, fields, format
        2. Judge rewards (90% of steps) — quality scoring
        3. Anti-hacking penalties — empty specs, over-engineering
        4. Regression penalty — breaking previously passing checks
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self._last_breakdown: dict[str, float] = {}
        self._previous_passing: set[str] = set()

    def compute(
        self,
        action: Action,
        state: State,
        task: TaskSpec,
        violations: list[RuleViolation],
    ) -> float:
        """Return scalar reward. Breakdown stored in `self._last_breakdown`."""

        # Get current spec from state
        current_spec = state.current_spec

        # 1. Hard verifier rewards (gates in HYBRID mode)
        hard_rewards = self._hard_verifier_rewards(current_spec)

        # 2. Judge-based component rewards (90% of steps)
        judge_rewards = self._judge_component_rewards(current_spec, task, action)

        # 3. Check if gates pass (HYBRID mode) — check both hard and judge rewards
        if self.config.mode == RewardMode.HYBRID:
            for gate_key in self.config.gate_components:
                # Check hard rewards first, then judge rewards
                gate_value = hard_rewards.get(gate_key, judge_rewards.get(gate_key, 0.0))
                if gate_value < self.config.gate_threshold:
                    # Store which gate failed separately (not in numeric breakdown)
                    self._gate_failed_on = gate_key
                    self._last_breakdown = {
                        **hard_rewards,
                        **judge_rewards,
                        "gated": 1.0,  # Boolean as float
                        "gate_failed": 1.0,  # Boolean as float
                        "total": 0.0,
                    }
                    return 0.0

        # 4. Anti-hacking penalties
        anti_hack_penalties = self._anti_hack_penalties(current_spec, action)

        # 4.5. Progress reward - incremental credit for building the spec
        progress_reward = self._progress_reward(current_spec, action)

        # 5. Calculate core score
        if self.config.mode == RewardMode.MULTIPLICATIVE:
            core = self._multiplicative({**hard_rewards, **judge_rewards})
        elif self.config.mode == RewardMode.HYBRID:
            core = 10.0 * self._additive(judge_rewards)
        else:  # ADDITIVE
            core = self._additive({**hard_rewards, **judge_rewards})

        # 6. Soft violation penalty
        soft_violations = [v for v in violations if v.severity == "soft"]
        penalty, per_category = self._aggregate_soft_penalty(soft_violations)

        # 7. Regression penalty
        regression = self._regression_penalty(current_spec, state)

        # 8. Novelty bonus
        bonus = self.config.novelty_bonus if not soft_violations else 0.0

        # 9. Total
        # anti_hack_penalties values are stored as negative numbers
        # (see RewardConfig.anti_hack_*), so we ADD them — subtracting would
        # flip the sign and turn penalties into bonuses (historical bug that
        # caused the policy to collapse to noop-submit by exploiting empty-spec
        # as a +5 reward instead of the intended -5 penalty).
        total = core + bonus + progress_reward - penalty - regression + sum(anti_hack_penalties.values())

        self._last_breakdown = {
            **hard_rewards,
            **judge_rewards,
            "core": core,
            "novelty_bonus": bonus,
            "progress": progress_reward,
            "soft_violation_penalty": -penalty,
            "regression_penalty": -regression,
            **{f"anti_hack_{k}": v for k, v in anti_hack_penalties.items()},
            **{f"violation_penalty_{cat}": -pen for cat, pen in per_category.items()},
            "total": total,
        }

        return total

    def _hard_verifier_rewards(self, spec: dict) -> dict[str, float]:
        """Run hard verifiers (free, 100% of steps)."""
        if HardVerifiers is None:
            return {"yaml_valid": 1.0, "has_required_fields": 1.0}

        results = HardVerifiers.verify_all(spec)
        return {k: v.score for k, v in results.items()}

    def _judge_component_rewards(
        self,
        spec: dict,
        task: TaskSpec,
        action: Action,
    ) -> dict[str, float]:
        """Compute judge-based component scores.

        These are normally computed by Claude Sonnet (90% of steps).
        For now, we use heuristic approximations.
        """

        # Skip judge on investigation commands
        if action.command in [ActionCommand.CHECK_SCORE, ActionCommand.INSPECT_EXAMPLE]:
            return {}

        return {
            "skill_selection": self._score_skill_selection(spec, task),
            "description_quality": self._score_description(spec),
            "workflow_clarity": self._score_workflow(spec),
            "model_appropriateness": self._score_model(spec, task),
            "best_practices": self._score_best_practices(spec),
            "efficiency": self._score_efficiency(spec),
        }

    def _score_skill_selection(self, spec: dict, task: TaskSpec) -> float:
        """Score: are skills appropriate for the task?"""
        required = set(task.required_skills)
        has = set(spec.get("skills", []))

        if not required:
            return 1.0  # No requirements = full credit

        # Coverage: how many required skills are present
        coverage = len(required & has) / len(required)

        # Penalize extra skills (over-engineering)
        extra = len(has - required)
        extra_penalty = min(extra * 0.1, 0.3)

        return max(0.0, coverage - extra_penalty)

    def _score_description(self, spec: dict) -> float:
        """Score: is description clear with delegation guidance?"""
        desc = spec.get("description", "")

        # Check for delegation keywords
        delegation_words = ["proactively", "use", "when", "specialist", "expert", "handles"]
        has_delegation = any(word in desc.lower() for word in delegation_words)

        # Check length (should be substantive but not verbose)
        good_length = 20 <= len(desc.split()) <= 100

        return min(1.0, 0.3 * has_delegation + 0.4 * good_length + 0.3 * (len(desc) > 0))

    def _score_workflow(self, spec: dict) -> float:
        """Score: does the system prompt have clear workflow steps?"""
        prompt = spec.get("system_prompt", "")

        # Check for step indicators
        step_patterns = ["1.", "2.", "step", "first", "then", "finally", "workflow"]
        has_steps = sum(1 for p in step_patterns if p.lower() in prompt.lower())

        # Normalize to 0-1
        return min(1.0, has_steps / 3.0)

    def _score_model(self, spec: dict, task: TaskSpec) -> float:
        """Score: is model appropriate for task complexity?"""
        model = spec.get("model", "sonnet")

        # Map difficulty to recommended model
        model_recommendations = {
            "easy": ModelType.HAIKU,
            "medium": ModelType.SONNET,
            "hard": ModelType.SONNET,
            "expert": ModelType.OPUS,
        }

        recommended = model_recommendations.get(task.difficulty, ModelType.SONNET)

        # Exact match = 1.0
        if model == recommended.value:
            return 1.0

        # Close enough (haiku for easy, sonnet for most) = 0.8
        if model in [ModelType.HAIKU.value, ModelType.SONNET.value]:
            return 0.8

        # Overkill (opus for easy) = 0.5
        return 0.5

    def _score_best_practices(self, spec: dict) -> float:
        """Score: does the spec follow domain best practices?"""
        prompt = spec.get("system_prompt", "")

        # Check for best practice keywords
        practice_keywords = [
            "handle error",
            "validate",
            "check",
            "ensure",
            "safely",
            "gracefully",
        ]

        matches = sum(1 for kw in practice_keywords if kw.lower() in prompt.lower())

        return min(1.0, matches / 3.0)

    def _score_efficiency(self, spec: dict) -> float:
        """Score: is the spec efficient (not over-engineered)?"""
        skills = spec.get("skills", [])

        # Penalize too many skills
        skill_count = len(skills)
        if skill_count <= 3:
            return 1.0
        elif skill_count <= 5:
            return 0.8
        elif skill_count <= self.config.max_skills_limit:
            return 0.5
        else:
            return 0.2  # Too many skills

    def _anti_hack_penalties(self, spec: dict, action: Action) -> dict[str, float]:
        """Detect and penalize reward hacking attempts."""
        penalties: dict[str, float] = {}

        # Empty spec check
        prompt = spec.get("system_prompt", "")
        if len(prompt) < self.config.min_prompt_length:
            penalties["empty_spec"] = self.config.anti_hack_empty_spec

        # Over-engineering check
        skills = spec.get("skills", [])
        if len(skills) > self.config.max_skills_limit:
            penalties["over_engineered"] = self.config.anti_hack_over_engineered

        # Wrong model tier check
        model = spec.get("model", "sonnet")
        if model == ModelType.OPUS.value and not spec.get("requires_opus", False):
            penalties["over_qualified_model"] = self.config.anti_hack_over_engineered

        return penalties

    def _regression_penalty(self, current_spec: dict, state: State) -> float:
        """Penalty for breaking previously-passing checks."""
        # Get current passing status
        if HardVerifiers is None:
            return 0.0

        current_results = HardVerifiers.verify_all(current_spec)
        currently_passing = {k for k, v in current_results.items() if v.passed}

        # Count regressions
        regressions = len(self._previous_passing - currently_passing)

        # Update for next step
        self._previous_passing = currently_passing

        return regressions * self.config.regression_penalty

    def _aggregate_soft_penalty(
        self,
        soft_violations: list[RuleViolation],
    ) -> tuple[float, dict[str, float]]:
        """Sum soft-violation penalties."""
        total = 0.0
        per_category: dict[str, float] = {}

        for v in soft_violations:
            weight = v.penalty if v.penalty is not None else self.config.soft_violation_penalty
            total += weight
            per_category[v.category] = per_category.get(v.category, 0.0) + weight

        return total, per_category

    def _progress_reward(self, spec: dict, action: Action) -> float:
        """Incremental reward for making progress on the spec.

        Gives small rewards for each meaningful action, even if the spec
        isn't complete yet. This ensures GRPO has variance during training.
        """
        from models import ActionCommand

        # Base progress for taking action (small participation reward)
        progress = 0.1

        # Extra progress for adding required fields
        if action.command == ActionCommand.SET_NAME and spec.get("name"):
            progress += 0.2
        if action.command == ActionCommand.SET_DESCRIPTION and spec.get("description"):
            progress += 0.3
        if action.command == ActionCommand.ADD_SKILL and spec.get("skills"):
            progress += 0.2
        if action.command == ActionCommand.SET_MODEL and spec.get("model"):
            progress += 0.1
        if action.command == ActionCommand.WRITE_PROMPT:
            prompt_len = len(spec.get("system_prompt", ""))
            if prompt_len >= 50:
                progress += 0.5
            elif prompt_len >= 20:
                progress += 0.2

        return progress

    def _multiplicative(self, components: dict[str, float]) -> float:
        """R = 10 × ∏ c_i^w_i — zero in any dim → zero reward."""
        product = 1.0
        for name, value in components.items():
            weight = self.config.component_weights.get(name, 1.0)
            product *= (value ** weight) if weight > 0 else value
        return 10.0 * product

    def _additive(self, components: dict[str, float]) -> float:
        """R = sum(weight × component)."""
        total = 0.0
        for name, value in components.items():
            weight = self.config.component_weights.get(name, 0.0)
            total += weight * value
        return total

    @property
    def last_breakdown(self) -> dict[str, float]:
        """Expose the last computed breakdown for observation building."""
        return dict(self._last_breakdown)


# For backward compatibility with template
class RewardComputer(MetaAgentRewardComputer):
    """Alias for backward compatibility."""
    pass
