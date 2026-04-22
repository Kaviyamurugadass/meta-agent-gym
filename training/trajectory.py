"""Trajectory serialization — JSONL format for offline RL / imitation learning.

    TrajectoryStep  — one (action, observation, reward) tuple per step
    Trajectory      — one full episode
    TrajectoryDataset — batch of episodes, loadable with one call

Enables:
    - Pre-collected trajectories shipped with env (demo value)
    - Offline RL training (behavior cloning, DPO, etc.)
    - Before/after eval comparisons
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from pydantic import BaseModel, Field


class TrajectoryStep(BaseModel):
    """One step in an episode."""

    step: int
    action: dict[str, Any]
    observation: dict[str, Any]
    reward: float
    done: bool = False
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    # Optional: include hidden state snapshot for debugging / imitation learning
    latent_snapshot: Optional[dict[str, Any]] = None


class Trajectory(BaseModel):
    """One complete episode."""

    task_id: str
    scenario_name: Optional[str] = None
    difficulty: Optional[str] = None
    steps: list[TrajectoryStep] = Field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False  # DOMAIN: define what "success" means per theme
    metadata: dict[str, Any] = Field(default_factory=dict)

    def append(self, step: TrajectoryStep) -> None:
        self.steps.append(step)
        self.total_reward += step.reward

    @property
    def length(self) -> int:
        return len(self.steps)

    def save(self, path: str | Path) -> None:
        """Save as pretty JSON."""
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate_json(f.read())


class TrajectoryDataset(BaseModel):
    """Batch of episodes. Save/load as directory of JSONL files."""

    trajectories: list[Trajectory] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self) -> Iterator[Trajectory]:  # type: ignore[override]
        return iter(self.trajectories)

    def append(self, trajectory: Trajectory) -> None:
        self.trajectories.append(trajectory)

    def filter_successful(self) -> "TrajectoryDataset":
        return TrajectoryDataset(
            trajectories=[t for t in self.trajectories if t.success],
        )

    def save_dir(self, dir_path: str | Path) -> None:
        """Save one JSON file per trajectory (numbered), plus an index."""
        out = Path(dir_path)
        out.mkdir(parents=True, exist_ok=True)
        index: list[dict[str, Any]] = []
        for i, traj in enumerate(self.trajectories):
            fname = f"trajectory_{i:04d}.json"
            traj.save(out / fname)
            index.append({
                "file": fname,
                "task_id": traj.task_id,
                "difficulty": traj.difficulty,
                "length": traj.length,
                "total_reward": traj.total_reward,
                "success": traj.success,
            })
        (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    @classmethod
    def load_dir(cls, dir_path: str | Path) -> "TrajectoryDataset":
        """Load all trajectory_*.json files from a directory."""
        p = Path(dir_path)
        files = sorted(p.glob("trajectory_*.json"))
        return cls(trajectories=[Trajectory.load(f) for f in files])

    def summary(self) -> dict[str, Any]:
        """Quick stats for README / reports."""
        if not self.trajectories:
            return {"n": 0}
        rewards = [t.total_reward for t in self.trajectories]
        lengths = [t.length for t in self.trajectories]
        successes = [t.success for t in self.trajectories]
        return {
            "n": len(self.trajectories),
            "success_rate": sum(successes) / len(successes),
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "mean_length": sum(lengths) / len(lengths),
        }
