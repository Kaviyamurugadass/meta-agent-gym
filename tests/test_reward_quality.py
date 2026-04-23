"""Reward-quality guardrails — fails the build if expert doesn't beat random.

Core invariant: your expert trajectory MUST score meaningfully higher than a
random policy on every non-placeholder scenario. If it doesn't, either:
  1. The reward function is broken (wrong formula, wrong weights)
  2. The expert trajectory is wrong (not actually optimal)
  3. The reward has no signal (all components return 0)

Any of these means GRPO training will produce garbage — agent can't learn
if there's no reward gradient between good and bad behavior.

Skip rule (same pattern as test_observation_quality.py):
  - `placeholder_*` scenarios skip — template filler, no expert trajectory
  - Domain-named scenarios activate the full check
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest

from server.tasks.scenarios import SCENARIOS
from training.benchmark import EXPERT_TRAJECTORIES, run_benchmark
from training.rollout_collection import collect


# Expert must score at least this multiple of random, with an absolute floor
# that prevents "0 vs 0" passing trivially.
EXPERT_RATIO_MIN = 3.0       # expert_reward >= 3.0 × random_reward
EXPERT_ABSOLUTE_MIN = 1.0    # expert_reward >= 1.0 regardless of random
N_RANDOM_EPISODES = 10       # sample size for random-policy baseline


def _is_placeholder(task_id: str) -> bool:
    return task_id.startswith("placeholder_")


def _scenarios_to_check() -> list:
    """Non-placeholder scenarios that ALSO have an expert trajectory defined."""
    return [
        s for s in SCENARIOS
        if not _is_placeholder(s.task_id) and s.task_id in EXPERT_TRAJECTORIES
    ]


@pytest.mark.parametrize(
    "scenario",
    [s for s in SCENARIOS],
    ids=lambda s: s.task_id,
)
def test_expert_beats_random(scenario, tmp_path):
    """Expert trajectory must produce rewards clearly above random policy.

    Enforces:
        expert_mean_reward >= EXPERT_RATIO_MIN × random_mean_reward
        AND
        expert_mean_reward >= EXPERT_ABSOLUTE_MIN
    """
    if _is_placeholder(scenario.task_id):
        pytest.skip(
            f"{scenario.task_id} is a placeholder — expert trajectory is "
            f"symbolic only, not a real optimal walkthrough."
        )

    if scenario.task_id not in EXPERT_TRAJECTORIES:
        pytest.skip(
            f"{scenario.task_id}: no expert trajectory defined in "
            f"training/benchmark.py. Add one before finale submission."
        )

    # 1. Random-policy baseline
    ds = collect(
        episodes=N_RANDOM_EPISODES,
        policy="random",
        output_dir=tmp_path,
        scenario_name=scenario.task_id,
        seed=42,
        domain_randomise=False,
    )
    random_mean = ds.summary().get("mean_reward", 0.0)

    # 2. Expert benchmark
    expert = run_benchmark(scenario.task_id)
    expert_reward = expert.total_reward

    # 3. Gate — absolute floor first, then ratio
    assert expert_reward >= EXPERT_ABSOLUTE_MIN, (
        f"\n{scenario.task_id}: expert reward is {expert_reward:.3f} "
        f"(< {EXPERT_ABSOLUTE_MIN}). "
        f"\nReward components are returning near-zero — domain fill of "
        f"_component_scores is incomplete. See server/rewards/reward.py "
        f"for formula menus per component type."
    )

    # Use max(random_mean, small_floor) to avoid divide-by-zero trivial pass
    effective_random = max(random_mean, 0.1)
    ratio = expert_reward / effective_random

    assert ratio >= EXPERT_RATIO_MIN, (
        f"\n{scenario.task_id}: expert {expert_reward:.3f} vs random {random_mean:.3f} "
        f"(ratio {ratio:.2f}x, need {EXPERT_RATIO_MIN}x). "
        f"\nSignal-to-noise too low — GRPO training won't produce improvement. "
        f"\nCheck: (a) reward formulas actually differentiate good/bad behavior, "
        f"(b) expert trajectory actually follows the optimal strategy, "
        f"(c) novelty_bonus isn't drowning out component signal (it's +0.1/step — "
        f"with 10 steps that's +1.0 that random can also pick up)."
    )


def test_report_reward_quality_status():
    """Informational — prints which scenarios will be reward-quality-enforced."""
    enforced = [
        s.task_id for s in SCENARIOS
        if not _is_placeholder(s.task_id) and s.task_id in EXPERT_TRAJECTORIES
    ]
    skipped_placeholder = [s.task_id for s in SCENARIOS if _is_placeholder(s.task_id)]
    skipped_no_expert = [
        s.task_id for s in SCENARIOS
        if not _is_placeholder(s.task_id) and s.task_id not in EXPERT_TRAJECTORIES
    ]

    print(f"\n[reward quality]")
    print(f"  enforced ({len(enforced)}):           {enforced or '(none yet)'}")
    print(f"  skipped: placeholder ({len(skipped_placeholder)}): {skipped_placeholder}")
    print(f"  skipped: no-expert  ({len(skipped_no_expert)}):  {skipped_no_expert}")
    print(
        f"  gates:  expert >= {EXPERT_RATIO_MIN}x random  AND  "
        f"expert >= {EXPERT_ABSOLUTE_MIN} absolute"
    )
    assert True


def test_expert_ratio_thresholds_are_sensible():
    """Sanity check on the guardrail thresholds themselves."""
    assert 1.0 < EXPERT_RATIO_MIN <= 10.0, "ratio threshold should be meaningful"
    assert 0.0 < EXPERT_ABSOLUTE_MIN <= 10.0, "absolute floor should be non-trivial"
    assert N_RANDOM_EPISODES >= 5, "need a reasonable sample for random baseline"
