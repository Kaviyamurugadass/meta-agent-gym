"""Rule engine — hard + soft violations.

    - Hard violations BLOCK the action entirely (unmet prerequisites, resource exhaustion, illegal target)
    - Soft violations ALLOW but penalize (redundancy, low confidence, stylistic issues)

Categories: prerequisite | resource | redundancy | causal | domain

TEMPLATE: subclass `RuleEngine` or extend `_domain_rules` on finale day.
The base class ships generic checks that apply to any domain.
"""

from __future__ import annotations

from typing import Callable

from models import Action, RuleViolation, State, TaskSpec

Rule = Callable[[Action, State, TaskSpec], list[RuleViolation]]


class RuleEngine:
    """Base rule engine. Extend `_domain_rules` per-domain."""

    def __init__(self, extra_rules: list[Rule] | None = None) -> None:
        self._rules: list[Rule] = [
            self._check_step_budget,
            self._check_budget_remaining,
            self._check_time_remaining,
        ]
        if extra_rules:
            self._rules.extend(extra_rules)
        self._rules.extend(self._domain_rules())

    def check(self, action: Action, state: State, task: TaskSpec) -> list[RuleViolation]:
        """Run all rules; return flat list of violations (hard + soft)."""
        violations: list[RuleViolation] = []
        for rule in self._rules:
            try:
                violations.extend(rule(action, state, task))
            except Exception:  # noqa: BLE001
                # Never let a buggy rule crash the env — log and skip
                continue
        return violations

    # ------------------------------------------------------------------ Built-in rules

    @staticmethod
    def _check_step_budget(action: Action, state: State, task: TaskSpec) -> list[RuleViolation]:
        if state.step >= task.max_steps:
            return [RuleViolation(
                severity="hard",
                category="resource",
                message=f"Step budget exhausted ({state.step}/{task.max_steps}).",
            )]
        return []

    @staticmethod
    def _check_budget_remaining(
        action: Action,  # noqa: ARG004
        state: State,    # noqa: ARG004
        task: TaskSpec,
    ) -> list[RuleViolation]:
        if task.budget is not None and task.budget < 0:
            return [RuleViolation(
                severity="hard",
                category="resource",
                message="Budget exhausted.",
            )]
        return []

    @staticmethod
    def _check_time_remaining(
        action: Action,  # noqa: ARG004
        state: State,    # noqa: ARG004
        task: TaskSpec,
    ) -> list[RuleViolation]:
        if task.time_limit is not None and task.time_limit < 0:
            return [RuleViolation(
                severity="hard",
                category="resource",
                message="Time limit reached.",
            )]
        return []

    # ------------------------------------------------------------------ Domain hook

    def _domain_rules(self) -> list[Rule]:
        """DOMAIN: meta-agent specific rules.

        Rules for agent specification generation:
            - Don't add duplicate skills
            - Don't use overkill model for task difficulty
            - Don't submit incomplete specs
            - Don't over-engineer (too many skills)
        """
        return [
            self._check_duplicate_skill,
            self._check_overkill_model,
            self._check_submit_readiness,
            self._check_over_engineering,
        ]

    # ------------------------------------------------------------------ Meta-agent domain rules

    @staticmethod
    def _check_duplicate_skill(action: Action, state: State, task: TaskSpec) -> list[RuleViolation]:
        """Prevent adding the same skill twice."""
        from models import ActionCommand

        if action.command == ActionCommand.ADD_SKILL:
            skill = action.args.get("skill")
            current_skills = state.current_spec.get("skills", [])
            if skill in current_skills:
                return [RuleViolation(
                    severity="soft",
                    category="redundancy",
                    message=f"Skill '{skill}' already added.",
                    penalty=0.05,
                )]
        return []

    @staticmethod
    def _check_overkill_model(action: Action, state: State, task: TaskSpec) -> list[RuleViolation]:
        """Prevent using opus for easy tasks (overkill)."""
        from models import ActionCommand

        if action.command == ActionCommand.SET_MODEL:
            model = action.args.get("model", "sonnet")
            if model == "opus" and task.difficulty in ["easy", "medium"]:
                return [RuleViolation(
                    severity="soft",
                    category="efficiency",
                    message=f"Using opus for {task.difficulty} task is overkill.",
                    penalty=0.10,
                )]
        return []

    @staticmethod
    def _check_submit_readiness(action: Action, state: State, task: TaskSpec) -> list[RuleViolation]:
        """Prevent submitting incomplete agent specifications."""
        from models import ActionCommand

        if action.command == ActionCommand.SUBMIT:
            spec = state.current_spec
            missing = []

            if not spec.get("name"):
                missing.append("name")
            if not spec.get("description"):
                missing.append("description")
            if not spec.get("system_prompt") or len(spec.get("system_prompt", "")) < 50:
                missing.append("system_prompt (too short)")

            if missing:
                return [RuleViolation(
                    severity="hard",
                    category="prerequisite",
                    message=f"Cannot submit: missing {', '.join(missing)}.",
                )]
        return []

    @staticmethod
    def _check_over_engineering(action: Action, state: State, task: TaskSpec) -> list[RuleViolation]:
        """Prevent over-engineering: too many skills for task difficulty."""
        from models import ActionCommand

        if action.command == ActionCommand.ADD_SKILL:
            current_skills = state.current_spec.get("skills", [])
            skill_count = len(current_skills) + 1  # Including the one being added

            # Limits per difficulty
            limits = {
                "easy": 3,
                "medium": 5,
                "hard": 7,
                "expert": 10,
            }

            limit = limits.get(task.difficulty, 5)
            if skill_count > limit:
                return [RuleViolation(
                    severity="soft",
                    category="efficiency",
                    message=f"Over-engineering: {skill_count} skills exceeds {task.difficulty} limit ({limit}).",
                    penalty=0.10,
                )]
        return []


# ---------------------------------------------------------------------------
# Utility rules domain code can import and register via `extra_rules`
# ---------------------------------------------------------------------------


def redundancy_rule(seen_actions: set[str], penalty: float = 0.05) -> Rule:
    """Factory: soft-violation when an identical action appears twice.

    Per-violation penalty (default 0.05) = minor redundancy. Judges see
    exactly what this cost in `reward_breakdown.violation_penalty_redundancy`.
    """
    def _check(action: Action, state: State, task: TaskSpec) -> list[RuleViolation]:  # noqa: ARG001
        key = f"{action.command.value}:{sorted(action.args.items())}"
        if key in seen_actions:
            return [RuleViolation(
                severity="soft",
                category="redundancy",
                message=f"Redundant action: {action.command.value}",
                penalty=penalty,
            )]
        seen_actions.add(key)
        return []
    return _check


def low_confidence_rule(threshold: float = 0.3, penalty: float = 0.10) -> Rule:
    """Factory: soft-violation when action confidence is below threshold.

    Per-violation penalty (default 0.10) = uncertain action ~2x worse than
    redundancy. Domain code can tune both thresholds independently.
    """
    def _check(action: Action, state: State, task: TaskSpec) -> list[RuleViolation]:  # noqa: ARG001
        if action.confidence is not None and action.confidence < threshold:
            return [RuleViolation(
                severity="soft",
                category="causal",
                message=f"Low confidence ({action.confidence:.2f} < {threshold}).",
                penalty=penalty,
            )]
        return []
    return _check
