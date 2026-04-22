"""Rollout collection — generate trajectories with random or heuristic policy.

Used by:
    - Pre-training baseline collection (fills README's Baseline column)
    - Data augmentation for imitation learning
    - Reward calibration sanity checks

CLI:
    uv run python training/rollout_collection.py \\
        --policy random --episodes 20 --output-dir data/baseline/random
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

from models import Action, ActionCommand
from server.environment import Environment
from training.trajectory import Trajectory, TrajectoryDataset, TrajectoryStep


# ---------------------------------------------------------------------------
# Policies — DOMAIN: extend with domain-aware heuristics on finale day
# ---------------------------------------------------------------------------


def random_policy(
    observation_dict: dict,  # noqa: ARG001
    rng: random.Random,
) -> Action:
    """Uniform random over ActionCommand values."""
    cmd = rng.choice(list(ActionCommand))
    return Action(command=cmd, args={}, confidence=0.5)


def heuristic_policy(
    observation_dict: dict,
    rng: random.Random,  # noqa: ARG001
) -> Action:
    """Simple rule-based policy — DOMAIN: override with domain knowledge.

    Default heuristic: SUBMIT only when at step >= max_steps - 1, else NOOP.
    A serviceable lower bound that respects the step budget.
    """
    step = observation_dict.get("step", 0)
    max_steps = observation_dict.get("max_steps", 10)
    if step >= max_steps - 1:
        return Action(command=ActionCommand.SUBMIT, confidence=0.7)
    return Action(command=ActionCommand.NOOP, confidence=0.5)


POLICIES = {
    "random": random_policy,
    "heuristic": heuristic_policy,
}


# ---------------------------------------------------------------------------
# Rollout runner
# ---------------------------------------------------------------------------


def run_episode(
    env: Environment,
    policy_name: str,
    scenario_name: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> Trajectory:
    """Run one episode to completion, return a Trajectory."""
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy: {policy_name}. Valid: {list(POLICIES)}")
    policy = POLICIES[policy_name]
    rng = rng or random.Random()

    obs = env.reset(scenario_name=scenario_name)
    traj = Trajectory(
        task_id=obs.task_id,
        scenario_name=scenario_name,
        difficulty=env._task.difficulty if env._task else None,
        metadata={"policy": policy_name},
    )

    while not (obs.done or obs.truncated):
        action = policy(obs.model_dump(), rng)
        obs = env.step(action)
        traj.append(TrajectoryStep(
            step=obs.step,
            action=action.model_dump(),
            observation=obs.model_dump(),
            reward=obs.reward,
            done=obs.done,
            reward_breakdown=obs.reward_breakdown,
        ))

    # DOMAIN: define success per theme. Default: positive final reward.
    traj.success = traj.total_reward > 0
    return traj


def collect(
    episodes: int,
    policy: str,
    output_dir: str | Path,
    scenario_name: Optional[str] = None,
    seed: Optional[int] = None,
    domain_randomise: bool = True,
) -> TrajectoryDataset:
    """Collect N episodes, save to output_dir as a TrajectoryDataset."""
    rng = random.Random(seed)
    dataset = TrajectoryDataset()

    for i in range(episodes):
        env = Environment(
            domain_randomise=domain_randomise,
            seed=rng.randint(0, 10_000_000),
        )
        traj = run_episode(env, policy, scenario_name=scenario_name, rng=rng)
        dataset.append(traj)
        print(
            f"[{i+1}/{episodes}] task={traj.task_id} "
            f"length={traj.length} reward={traj.total_reward:.3f} "
            f"success={traj.success}"
        )

    dataset.save_dir(output_dir)
    summary = dataset.summary()
    print(f"\n==> {episodes} episodes saved to {output_dir}")
    print(f"    mean_reward={summary['mean_reward']:.3f} "
          f"success_rate={summary['success_rate']:.1%} "
          f"mean_length={summary['mean_length']:.1f}")
    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect rollout trajectories.")
    parser.add_argument("--policy", choices=list(POLICIES), default="random")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--scenario-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-randomise", action="store_true")
    args = parser.parse_args()

    collect(
        episodes=args.episodes,
        policy=args.policy,
        output_dir=args.output_dir,
        scenario_name=args.scenario_name,
        seed=args.seed,
        domain_randomise=not args.no_randomise,
    )


if __name__ == "__main__":
    main()
