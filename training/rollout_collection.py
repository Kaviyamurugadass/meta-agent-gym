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

from models import Action, ActionCommand, RewardConfig, RewardMode
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
    """Competent rule-based baseline that actually builds a valid agent spec.

    Fills each required field once (name, description, skill, prompt, model) then
    submits. Purpose: prove the environment is reachable with >0 reward so GRPO
    has learning signal to bootstrap from. Not optimal — judge-scored components
    (description_quality, workflow_clarity, etc.) will be mediocre.
    """
    step = observation_dict.get("step", 0)
    max_steps = observation_dict.get("max_steps", 7)
    task_id = observation_dict.get("task_id", "task")
    available_skills = observation_dict.get("available_skills") or []
    current_spec = observation_dict.get("current_spec") or {}

    # Submit on the final step regardless
    if step >= max_steps - 1:
        return Action(command=ActionCommand.SUBMIT, confidence=0.7)

    # Fill missing required fields in priority order
    if not current_spec.get("name"):
        return Action(
            command=ActionCommand.SET_NAME,
            args={"name": task_id.replace("_", "-")},
            confidence=0.6,
        )
    if not current_spec.get("description"):
        return Action(
            command=ActionCommand.SET_DESCRIPTION,
            args={"description": f"Agent that handles {task_id} tasks end-to-end."},
            confidence=0.6,
        )
    if not current_spec.get("skills") and available_skills:
        return Action(
            command=ActionCommand.ADD_SKILL,
            args={"skill": available_skills[0]},
            confidence=0.6,
        )
    prompt = current_spec.get("system_prompt", "")
    if len(prompt) < 50:
        return Action(
            command=ActionCommand.WRITE_PROMPT,
            args={
                "prompt": (
                    "You are a specialist agent. Read the task carefully, plan the "
                    "steps, execute each one, then verify the result before submitting."
                ),
                "mode": "replace",
            },
            confidence=0.6,
        )
    if not current_spec.get("model"):
        return Action(
            command=ActionCommand.SET_MODEL,
            args={"model": "sonnet"},
            confidence=0.6,
        )
    # All required fields filled — coast until the final SUBMIT step
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
    curriculum_phase: Optional[int] = None,
    reward_config: Optional[RewardConfig] = None,
) -> TrajectoryDataset:
    """Collect N episodes, save to output_dir as a TrajectoryDataset."""
    rng = random.Random(seed)
    dataset = TrajectoryDataset()

    for i in range(episodes):
        env = Environment(
            domain_randomise=domain_randomise,
            seed=rng.randint(0, 10_000_000),
            curriculum_phase=curriculum_phase or 1,
            reward_config=reward_config,
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
    parser.add_argument("--curriculum-phase", type=int, default=None,
                        help="Curriculum phase (1-4) for task selection")
    parser.add_argument("--reward-mode",
                        choices=[m.value for m in RewardMode],
                        default=None,
                        help="Override reward mode (default: HYBRID). "
                             "Use 'additive' to expose anti-hack penalty surface "
                             "without gate masking.")
    args = parser.parse_args()

    reward_config = None
    if args.reward_mode is not None:
        reward_config = RewardConfig(mode=RewardMode(args.reward_mode))

    collect(
        episodes=args.episodes,
        policy=args.policy,
        output_dir=args.output_dir,
        scenario_name=args.scenario_name,
        seed=args.seed,
        domain_randomise=not args.no_randomise,
        curriculum_phase=args.curriculum_phase,
        reward_config=reward_config,
    )


if __name__ == "__main__":
    main()
