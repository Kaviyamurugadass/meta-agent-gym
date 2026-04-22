"""Scenario library — one TaskSpec per curated problem.

TEMPLATE: on finale day, replace `SCENARIOS` with 3-4 domain scenarios
grounded in real citations (papers, RFCs, style guides, benchmarks).

Difficulty scaling should be ARCHITECTURAL, not additive:
    easy   — single component, direct signal
    medium — 2-3 components with tradeoffs
    hard   — cross-file / cross-agent / cross-module dependencies
"""

from __future__ import annotations

from models import TaskSpec


# ---------------------------------------------------------------------------
# Placeholder scenarios — so reset() returns a valid TaskSpec during dry-run.
# DOMAIN: replace entirely with real scenarios on finale day.
# ---------------------------------------------------------------------------

SCENARIOS: list[TaskSpec] = [
    TaskSpec(
        task_id="placeholder_easy",
        difficulty="easy",
        problem_statement=(
            "PLACEHOLDER — template dry-run scenario. "
            "Replace with real domain easy scenario on finale day."
        ),
        max_steps=5,
        citations=["template://placeholder"],
        expected_findings={"placeholder": True},
    ),
    TaskSpec(
        task_id="placeholder_medium",
        difficulty="medium",
        problem_statement=(
            "PLACEHOLDER — template dry-run scenario. "
            "Replace with real domain medium scenario on finale day."
        ),
        max_steps=10,
        citations=["template://placeholder"],
        expected_findings={"placeholder": True},
    ),
    TaskSpec(
        task_id="placeholder_hard",
        difficulty="hard",
        problem_statement=(
            "PLACEHOLDER — template dry-run scenario. "
            "Replace with real domain hard scenario (cross-component, red herrings)."
        ),
        max_steps=20,
        citations=["template://placeholder"],
        expected_findings={"placeholder": True},
    ),
]


def get_scenario(name: str) -> TaskSpec | None:
    """Lookup by task_id. Returns None if not found."""
    for scenario in SCENARIOS:
        if scenario.task_id == name:
            return scenario
    return None
