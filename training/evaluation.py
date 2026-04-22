"""Evaluation — metrics on trajectory datasets.

Populates the before/after training table in README. Three metric families:

    online_metrics     — return, length, success rate (standard RL eval)
    behavior_metrics   — action diversity, ordering score, invalid-action rate
    fidelity_metrics   — reward distribution match vs expert / baseline

CLI:
    uv run python training/evaluation.py \\
        --input-dirs data/baseline/random data/baseline/heuristic \\
        --output data/baseline/metrics.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Optional

from training.trajectory import Trajectory, TrajectoryDataset


class EvaluationSuite:
    """Static methods on trajectory datasets — compose to build custom eval."""

    # ---------------------------------------------------------------- Online RL

    @staticmethod
    def online_metrics(dataset: TrajectoryDataset) -> dict[str, float]:
        """Standard RL eval metrics."""
        if not dataset.trajectories:
            return {"n": 0}
        rewards = [t.total_reward for t in dataset.trajectories]
        lengths = [t.length for t in dataset.trajectories]
        successes = [float(t.success) for t in dataset.trajectories]
        return {
            "n": len(dataset.trajectories),
            "mean_reward": statistics.mean(rewards),
            "median_reward": statistics.median(rewards),
            "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "mean_length": statistics.mean(lengths),
            "success_rate": statistics.mean(successes),
        }

    # ---------------------------------------------------------------- Behavior

    @staticmethod
    def behavior_metrics(dataset: TrajectoryDataset) -> dict[str, float]:
        """Action-level metrics — catches mode collapse, random spam, etc."""
        if not dataset.trajectories:
            return {"n": 0}

        all_actions: list[str] = []
        all_confidences: list[float] = []
        violation_counts: list[int] = []

        for traj in dataset.trajectories:
            for s in traj.steps:
                cmd = s.action.get("command", "unknown")
                all_actions.append(cmd)
                conf = s.action.get("confidence")
                if conf is not None:
                    all_confidences.append(conf)
                violations = s.observation.get("rule_violations", [])
                violation_counts.append(len(violations))

        unique = len(set(all_actions))
        total = len(all_actions)
        return {
            "action_diversity": unique / total if total else 0.0,
            "mean_confidence": statistics.mean(all_confidences) if all_confidences else 0.0,
            "mean_violations_per_step": statistics.mean(violation_counts) if violation_counts else 0.0,
            "total_steps": total,
            "unique_actions": unique,
        }

    # ---------------------------------------------------------------- Fidelity

    @staticmethod
    def fidelity_metrics(
        trained: TrajectoryDataset,
        reference: TrajectoryDataset,
    ) -> dict[str, float]:
        """How well trained agent's reward distribution matches reference."""
        if not trained.trajectories or not reference.trajectories:
            return {}
        t_rewards = [t.total_reward for t in trained.trajectories]
        r_rewards = [t.total_reward for t in reference.trajectories]
        return {
            "reward_mean_gap": abs(statistics.mean(t_rewards) - statistics.mean(r_rewards)),
            "reward_median_gap": abs(statistics.median(t_rewards) - statistics.median(r_rewards)),
            "improvement_vs_reference": (
                (statistics.mean(t_rewards) - statistics.mean(r_rewards))
                / (abs(statistics.mean(r_rewards)) + 1e-6)
            ),
        }

    # ---------------------------------------------------------------- Composite

    @classmethod
    def full_report(
        cls,
        dataset: TrajectoryDataset,
        reference: Optional[TrajectoryDataset] = None,
        label: str = "dataset",
    ) -> dict[str, Any]:
        """One-call eval for a dataset."""
        out: dict[str, Any] = {
            "label": label,
            "online": cls.online_metrics(dataset),
            "behavior": cls.behavior_metrics(dataset),
        }
        if reference is not None:
            out["fidelity"] = cls.fidelity_metrics(dataset, reference)
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trajectory datasets.")
    parser.add_argument(
        "--input-dirs", nargs="+", required=True,
        help="One or more trajectory directories (saved by rollout_collection.py)",
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Write full report as JSON to this path (else stdout only)")
    parser.add_argument("--reference", type=str, default=None,
                        help="Optional reference dataset dir for fidelity metrics")
    args = parser.parse_args()

    reference = (
        TrajectoryDataset.load_dir(args.reference) if args.reference else None
    )

    reports: list[dict[str, Any]] = []
    for dir_path in args.input_dirs:
        label = Path(dir_path).name
        dataset = TrajectoryDataset.load_dir(dir_path)
        report = EvaluationSuite.full_report(dataset, reference, label=label)
        reports.append(report)
        print(f"\n=== {label} ===")
        print(json.dumps(report, indent=2, default=str))

    if args.output:
        Path(args.output).write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
        print(f"\n==> Report written to {args.output}")


if __name__ == "__main__":
    main()
