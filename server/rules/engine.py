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
        """DOMAIN: override to append domain-specific rules.

        Examples by domain:
            - code auditor: forbid editing files not in repo, forbid partial syntax
            - SRE: forbid kubectl delete, require kubectl get before act
            - clinical: require QC before DE analysis
        """
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
