"""Literature / expert benchmark — scripted "perfect agent" per scenario.

For each scenario, author the ideal action sequence a domain expert would
take. Benchmark runs this sequence against the env and reports match-ratio
against ground truth.

Why judges love this:
    - Proves the env isn't gameable (expert trajectory scores high)
    - Gives an upper bound for "Trained" column in README
    - Makes grading intent legible

TEMPLATE: `EXPERT_TRAJECTORIES` is a placeholder. On finale day, fill one
entry per scenario with the real action sequence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models import Action, ActionCommand
from training.reward_backend import LocalBackend


@dataclass
class BenchmarkResult:
    scenario_name: str
    total_reward: float
    steps_taken: int
    success: bool
    match_ratio: float  # fraction of expected_findings the trajectory discovered
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# Expert trajectories — DOMAIN: replace each entry with real action sequences
# ---------------------------------------------------------------------------

EXPERT_TRAJECTORIES: dict[str, list[Action]] = {
    # Placeholder expert trajectories — the template's default actions are all
    # no-ops ending in a submit. Domain code must override these per scenario.
    "placeholder_easy": [
        Action(command=ActionCommand.INSPECT, args={"target": "placeholder"}, confidence=0.9),
        Action(command=ActionCommand.NOOP),
        Action(command=ActionCommand.NOOP),
        Action(command=ActionCommand.SUBMIT, justification="placeholder", confidence=0.9),
    ],
    "placeholder_medium": [
        Action(command=ActionCommand.INSPECT, args={"target": "placeholder_a"}, confidence=0.9),
        Action(command=ActionCommand.INSPECT, args={"target": "placeholder_b"}, confidence=0.9),
        Action(command=ActionCommand.NOOP),
        Action(command=ActionCommand.NOOP),
        Action(command=ActionCommand.SUBMIT, justification="placeholder", confidence=0.9),
    ],
    "placeholder_hard": [
        Action(command=ActionCommand.INSPECT, args={"target": f"c{i}"}, confidence=0.9)
        for i in range(5)
    ] + [
        Action(command=ActionCommand.NOOP)
        for _ in range(5)
    ] + [
        Action(command=ActionCommand.SUBMIT, justification="placeholder", confidence=0.9),
    ],
}


def compute_match_ratio(observations, expected_findings: dict[str, Any]) -> float:  # type: ignore[no-untyped-def]
    """DOMAIN: override with keyword/semantic match.

    Default: checks whether each expected finding KEY appears anywhere in the
    observation history's summary text. Good enough for placeholder; real
    domains should check for specific values, not keys.
    """
    if not expected_findings:
        return 0.0
    corpus = " ".join(
        str(obs.summary) + " " + str(obs.latest_output or "")
        for obs in observations
    ).lower()
    hits = sum(1 for key in expected_findings if str(key).lower() in corpus)
    return hits / len(expected_findings)


def run_benchmark(
    scenario_name: str,
    trajectory: list[Action] | None = None,
    base_url: str | None = None,
) -> BenchmarkResult:
    """Run expert trajectory against the env, return result."""
    trajectory = trajectory or EXPERT_TRAJECTORIES.get(scenario_name)
    if trajectory is None:
        raise ValueError(f"No expert trajectory defined for {scenario_name}")

    backend = LocalBackend()
    total_reward, observations = backend.score(trajectory, scenario_name=scenario_name)

    # Pull expected_findings from the scenario library
    from server.tasks.scenarios import get_scenario
    scenario = get_scenario(scenario_name)
    expected = scenario.expected_findings if scenario else {}

    match = compute_match_ratio(observations, expected)
    final_obs = observations[-1]

    return BenchmarkResult(
        scenario_name=scenario_name,
        total_reward=total_reward,
        steps_taken=final_obs.step,
        success=final_obs.done and total_reward > 0,
        match_ratio=match,
        details={
            "truncated": final_obs.truncated,
            "final_reward_breakdown": dict(final_obs.reward_breakdown),
            "violations_seen": sum(
                len(obs.rule_violations) for obs in observations
            ),
        },
    )


def run_all(scenarios: list[str] | None = None) -> list[BenchmarkResult]:
    scenarios = scenarios or list(EXPERT_TRAJECTORIES.keys())
    return [run_benchmark(s) for s in scenarios]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run expert benchmark.")
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Specific scenarios to run. Default: all with expert trajectories.",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_all(args.scenarios)
    for r in results:
        print(
            f"{r.scenario_name:30s} "
            f"reward={r.total_reward:7.3f}  "
            f"steps={r.steps_taken:3d}  "
            f"match={r.match_ratio:4.2f}  "
            f"success={r.success}"
        )

    if args.output:
        payload = [vars(r) for r in results]
        Path(args.output).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\n==> {len(results)} results → {args.output}")


if __name__ == "__main__":
    main()
