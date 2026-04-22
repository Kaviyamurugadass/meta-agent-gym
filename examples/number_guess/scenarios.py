"""Example: number-guess domain — scenarios fill.

Three difficulty levels. Architectural scaling (range size), not additive.
"""

from __future__ import annotations

from models import TaskSpec

SCENARIOS: list[TaskSpec] = [
    TaskSpec(
        task_id="guess_easy",
        difficulty="easy",
        problem_statement=(
            "Guess an integer between 1 and 16 inclusive. "
            "After each guess, you'll be told 'higher' / 'lower' / 'correct'."
        ),
        max_steps=8,  # log2(16) + buffer
        citations=["https://en.wikipedia.org/wiki/Binary_search_algorithm"],
        expected_findings={"optimal_steps": 4, "range": [1, 16]},
    ),
    TaskSpec(
        task_id="guess_medium",
        difficulty="medium",
        problem_statement="Guess an integer between 1 and 256 inclusive.",
        max_steps=12,
        citations=["https://en.wikipedia.org/wiki/Binary_search_algorithm"],
        expected_findings={"optimal_steps": 8, "range": [1, 256]},
    ),
    TaskSpec(
        task_id="guess_hard",
        difficulty="hard",
        problem_statement=(
            "Guess an integer between 1 and 10000 inclusive. "
            "Watch the step budget — naive guessing will run out."
        ),
        max_steps=18,
        citations=["https://en.wikipedia.org/wiki/Binary_search_algorithm"],
        expected_findings={"optimal_steps": 14, "range": [1, 10000]},
    ),
]


def get_scenario(name: str) -> TaskSpec | None:
    for s in SCENARIOS:
        if s.task_id == name:
            return s
    return None
