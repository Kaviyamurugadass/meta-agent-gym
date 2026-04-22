"""Reward computer — three reward-combining modes.

    ADDITIVE       — R = Σ w_i × c_i
                     Smooth gradient, partial credit everywhere. Safest for
                     GRPO training. Gameable if domain doesn't enforce all
                     components via other mechanisms.

    MULTIPLICATIVE — R = 10 × ∏ c_i^w_i
                     All dims must succeed. Hard to game. Risk: collapses
                     to ~0 on any small drop → weak GRPO gradient. Use only
                     for binary-success / security-critical domains.

    HYBRID         — Gate with multiplicative logic, score with additive.
                     If any `gate_components` value < `gate_threshold`
                     → reward = 0.  Else: 10 × additive-weighted-sum.
                     Best for most domains. Strict on hard constraints,
                     smooth elsewhere. Default gate: ["safety"].

Recommendation: start with HYBRID (safe default). Switch to pure
MULTIPLICATIVE only if ALL components are equally non-negotiable.

TEMPLATE: override `_component_scores()` per domain to supply actual values.
Weights + penalties are already wired through RewardConfig.
"""

from __future__ import annotations

from models import (
    Action,
    RewardConfig,
    RewardMode,
    RuleViolation,
    State,
    TaskSpec,
)


class RewardComputer:
    """Compute per-step scalar reward + expose breakdown for observation."""

    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self._last_breakdown: dict[str, float] = {}

    def compute(
        self,
        action: Action,
        state: State,
        task: TaskSpec,
        violations: list[RuleViolation],
    ) -> float:
        """Return scalar reward. Breakdown stored in `self._last_breakdown`."""
        components = self._component_scores(action, state, task)

        # Normalize components to [0, 1] if caller gave raw values
        components = {k: max(0.0, min(1.0, v)) for k, v in components.items()}

        # Core score
        if self.config.mode == RewardMode.MULTIPLICATIVE:
            core = self._multiplicative(components)
        elif self.config.mode == RewardMode.HYBRID:
            core = self._hybrid(components)
        else:  # ADDITIVE
            core = self._additive(components)

        # Soft violation penalty — per-violation weight if provided, else flat fallback
        soft_violations = [v for v in violations if v.severity == "soft"]
        penalty, per_category_penalties = self._aggregate_soft_penalty(soft_violations)

        # Novelty bonus (only if zero soft violations)
        bonus = self.config.novelty_bonus if not soft_violations else 0.0

        # Regression penalty — DOMAIN: subclass should track pre/post passing checks
        regression = self._regression_penalty(action, state, task)

        total = core + bonus - penalty - regression

        self._last_breakdown = {
            **components,
            "core": core,
            "novelty_bonus": bonus,
            "soft_violation_penalty": -penalty,
            "regression_penalty": -regression,
            "total": total,
            # Per-category violation penalty breakdown (flattened for dict[str, float])
            **{f"violation_penalty_{cat}": -pen for cat, pen in per_category_penalties.items()},
        }
        return total

    def _aggregate_soft_penalty(
        self,
        soft_violations: list[RuleViolation],
    ) -> tuple[float, dict[str, float]]:
        """Sum soft-violation penalties, prefer per-violation weight over flat fallback.

        Returns (total_penalty, per_category_breakdown).

        Per-violation model: each violation carries its own `penalty` field.
        Flat fallback: if `penalty is None`, use `config.soft_violation_penalty`.
        Judges see both the total and the per-category breakdown in
        `reward_breakdown`.
        """
        total = 0.0
        per_category: dict[str, float] = {}
        for v in soft_violations:
            weight = v.penalty if v.penalty is not None else self.config.soft_violation_penalty
            total += weight
            per_category[v.category] = per_category.get(v.category, 0.0) + weight
        return total, per_category

    @property
    def last_breakdown(self) -> dict[str, float]:
        """Expose the last computed breakdown for observation building."""
        return dict(self._last_breakdown)

    # ------------------------------------------------------------------ Mode helpers

    def _multiplicative(self, components: dict[str, float]) -> float:
        """R = 10 × c1 × c2 × ... — zero in any dim → zero reward."""
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

    def _hybrid(self, components: dict[str, float]) -> float:
        """Gate with multiplicative, score with additive.

        1. If any `gate_components` value is below `gate_threshold` → reward = 0
           (the hard-constraint behavior of multiplicative)
        2. Otherwise: 10 × additive-weighted-sum of ALL components
           (smooth gradient for GRPO training)

        Best for most domains: strict on the few things that must hold,
        smooth on the rest. Won't collapse to zero from a small dip in
        non-critical dims.
        """
        for gate_key in self.config.gate_components:
            gate_value = components.get(gate_key, 0.0)
            if gate_value < self.config.gate_threshold:
                return 0.0
        # Scale by 10 to match multiplicative's magnitude — keeps the
        # reward-quality guardrail (expert >= 1.0 absolute) meaningful
        return 10.0 * self._additive(components)

    # ------------------------------------------------------------------ Fill-in hooks

    def _component_scores(
        self,
        action: Action,      # noqa: ARG002
        state: State,        # noqa: ARG002
        task: TaskSpec,      # noqa: ARG002
    ) -> dict[str, float]:
        """Return per-component score in [0, 1].

        DOMAIN: override to compute actual domain-specific values. The
        framework handles weighting + mode + penalties — you just supply
        the raw signal per component.

        Default returns zeros for the 4 default components — makes reward 0
        but keeps the structure visible for debugging.

        ----------------------------------------------------------------------
        JUDGE'S QUESTION: "How did you measure <X>?"
        ----------------------------------------------------------------------
        For every component you return, you need a defensible answer with:
            1. What it measures (one sentence)
            2. Exact formula
            3. Why this shape (linear / exponential / harmonic)
            4. A concrete example value
            5. How it interacts with other components (multiplicative mode)

        Pick from the formula menus below. Don't invent a formula mid-finale.
        Document your choice in README.md's "Reward Justification" section.

        ----------------------------------------------------------------------
        FORMULA MENU — efficiency
        ----------------------------------------------------------------------
            max(0, 1 - steps_used / max_steps)     → linear decay, simple
            remaining_budget / initial_budget      → when $ / compute is scarce
            exp(-steps / max_steps)                → exponential, sharp late-penalty

        ----------------------------------------------------------------------
        FORMULA MENU — correctness
        ----------------------------------------------------------------------
            passed_checks / total_checks           → discrete rubric
            1.0 if final == ground_truth else 0.0  → exact match (guess/classify)
            1 - edit_distance(out, tgt) / len(tgt) → continuous text/code similarity
            f1_score(predicted, expected)          → multi-label / multi-target

        ----------------------------------------------------------------------
        FORMULA MENU — quality
        ----------------------------------------------------------------------
            (readability + coverage + type_hints) / 3   → code-domain composite
            fidelity_score(output, reference)           → scientific/bio domain
            sharpe_ratio(returns) / max_sharpe          → finance domain
            1 - calibration_error(preds, outcomes)      → forecasting domain

        ----------------------------------------------------------------------
        FORMULA MENU — safety
        ----------------------------------------------------------------------
            1 - regressions_caused / baseline_passing   → anti-regression
            1 - destructive_actions / total_actions     → action-level safety
            constraint_satisfaction_rate                → rule-bounded domains

        ----------------------------------------------------------------------
        EXAMPLE — SRE/ops domain fill
        ----------------------------------------------------------------------
            return {
                "correctness": (
                    1.0 if state.hidden_truth["incident_resolved"] else 0.0
                ),
                "efficiency": max(0, 1 - state.step / task.max_steps),
                "quality":    state.progress_flags.get("triage_complete", 0) / 1.0,
                "safety":     1.0 - state.hidden_truth["destructive_calls"] / max(1, state.step),
            }

        Judge can challenge any of these and you have a 5-point answer ready.
        """
        return {
            "correctness": 0.0,
            "efficiency": 0.0,
            "quality": 0.0,
            "safety": 0.0,
        }

    def _regression_penalty(
        self,
        action: Action,      # noqa: ARG002
        state: State,        # noqa: ARG002
        task: TaskSpec,      # noqa: ARG002
    ) -> float:
        """Penalty for breaking previously-passing checks.

        DOMAIN: track which checks/tests were passing before the action; if
        fewer pass afterwards, return N × config.regression_penalty per
        regressed check.

        Default: 0.0 — no regression tracking yet.
        """
        return 0.0
