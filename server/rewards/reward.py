"""Reward computer — composable rubrics via `openenv.core.rubrics`.

Implements the hackathon's "composable rubrics > monolithic scoring" guideline
using native OpenEnv primitives: every reward component is a `Rubric` subclass
in `rubric_reward.py`, composed here with `RubricDict`, `WeightedSum`, and
`Gate` from `openenv.core.rubrics.containers`.

Internal composition (see `__init__` below):
    - `_hard_rubrics`  : RubricDict of 5 hard-verifier Rubrics (yaml_valid, etc.)
    - `_judge_rubrics` : dict of 6 judge Rubrics (skill_selection, etc.)
    - `_judge_weighted`: WeightedSum aggregating judge Rubrics with config weights
    - `_hard_weighted` : uniform-weighted sum of hard Rubrics
    - `_hard_gate`     : Gate(hard_weighted, threshold=0.99) — HYBRID gating

Public API unchanged: `compute()` returns a scalar; `last_breakdown` exposes
the per-component dict the dashboard and GRPO trainer both consume.

RLVR approach preserved:
    1. Hard verifiers (100% of steps) — YAML, fields, format
    2. Judge rewards (90% of steps) — quality scoring
    3. Anti-hacking penalties — empty specs, over-engineering
    4. Regression penalty — breaking previously passing checks
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

from server.rewards.rubric_reward import (
    HARD_COMPONENT_KEYS,
    JUDGE_COMPONENT_KEYS,
    build_hard_gate,
    build_hard_rubric_dict,
    build_hard_rubrics,
    build_hard_weighted,
    build_judge_rubrics,
    build_judge_weighted_sum,
    pack_obs,
)

try:
    from server.verifiers import HardVerifiers
except ImportError:
    HardVerifiers = None  # type: ignore[assignment,misc]

logger = logging.getLogger("server.rewards.reward")


class MetaAgentRewardComputer:
    """Compute per-step scalar reward with multi-component breakdown.

    Composes OpenEnv `Rubric` objects (`RubricDict`, `WeightedSum`, `Gate`)
    to produce the breakdown. Scoring logic lives in `rubric_reward.py` —
    this class orchestrates the composition and applies the domain-specific
    penalties (anti-hacking, regression, soft violations) that aren't part
    of the Rubric set.
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self._last_breakdown: dict[str, float] = {}
        self._previous_passing: set[str] = set()

        # --- OpenEnv Rubric composition -----------------------------------
        self._hard_rubrics = build_hard_rubrics()
        self._judge_rubrics = build_judge_rubrics(
            max_skills_limit=config.max_skills_limit,
        )
        # Containers (used for introspection + aggregate scoring)
        self._hard_dict = build_hard_rubric_dict(self._hard_rubrics)
        self._hard_weighted = build_hard_weighted(self._hard_rubrics)
        self._hard_gate = build_hard_gate(
            self._hard_weighted,
            threshold=config.gate_threshold,
        )
        self._judge_weighted = build_judge_weighted_sum(
            self._judge_rubrics,
            config.component_weights,
        )

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
        """Run hard-verifier Rubrics (free, 100% of steps).

        Each key comes from `HARD_COMPONENT_KEYS` — yaml_valid, has_required_fields,
        prompt_length_ok, model_valid, skills_format_ok. Rubric objects call into
        `HardVerifiers.verify_all()` internally, same logic as before.
        """
        obs = pack_obs(current_spec=spec, task=None, action=None)
        # Direct call on each Rubric (bypasses container aggregation so we
        # get per-key values for the breakdown dict the dashboard expects).
        return {key: float(r(None, obs)) for key, r in self._hard_rubrics.items()}

    def _judge_component_rewards(
        self,
        spec: dict,
        task: TaskSpec,
        action: Action,
    ) -> dict[str, float]:
        """Compute judge-based component scores via Rubric objects.

        Normally these would be scored by Claude Sonnet (90% of steps). The
        current implementation uses heuristic Rubrics that produce identical
        values to the pre-Rubric-refactor scoring; upgrading individual
        Rubrics to call `openenv.core.rubrics.LLMJudge` is a drop-in change.
        """
        # Skip judge on investigation commands (unchanged from pre-refactor)
        if action.command in [ActionCommand.CHECK_SCORE, ActionCommand.INSPECT_EXAMPLE]:
            return {}

        obs = pack_obs(current_spec=spec, task=task, action=action)
        return {key: float(r(action, obs)) for key, r in self._judge_rubrics.items()}

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
