"""Example: number-guess domain — expert benchmark.

The optimal strategy is binary search. Produces an expert trajectory
by computing the midpoint guesses for each scenario range.
"""

from __future__ import annotations

from models import Action


def binary_search_trajectory(low: int, high: int, target: int) -> list[Action]:
    """Emit the sequence of guesses an optimal binary searcher would make."""
    actions: list[Action] = []
    while low <= high:
        mid = (low + high) // 2
        actions.append(Action(
            command="guess",
            args={"value": mid},
            justification=f"binary search midpoint of [{low}, {high}]",
            confidence=1.0,
        ))
        if mid == target:
            actions.append(Action(command="submit", confidence=1.0))
            return actions
        elif mid < target:
            low = mid + 1
        else:
            high = mid - 1
    return actions  # shouldn't reach here for valid targets


# Precomputed expert trajectories (target sampled — a real fill would call
# binary_search_trajectory at runtime with the actual hidden target).
EXPERT_TRAJECTORIES = {
    "guess_easy":   binary_search_trajectory(1, 16, 13),      # target=13 example
    "guess_medium": binary_search_trajectory(1, 256, 137),    # target=137 example
    "guess_hard":   binary_search_trajectory(1, 10000, 4242), # target=4242 example
}
