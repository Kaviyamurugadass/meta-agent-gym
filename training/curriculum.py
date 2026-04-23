"""Curriculum controller — manages phase progression during training.

Tracks success rate per phase over a rolling window. When the agent achieves
mastery (success rate > threshold), advances to the next phase. When
performance drops below regression threshold, goes back one phase.

Usage in training loop::

    controller = CurriculumController()
    for episode in range(num_episodes):
        phase = controller.current_phase
        obs = env.reset(curriculum_phase=phase)
        # ... run episode ...
        controller.record(trajectory.success, trajectory.task_id)

CLI:
    python -m training.curriculum --history data/trained --output curriculum_state.json
"""

from __future__ import annotations

import json
import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger("training.curriculum")

# Phase mastery thresholds from curriculum strategy
PHASE_MASTERY = {1: 0.50, 2: 0.30, 3: 0.10, 4: 1.00}
PHASE_REGRESSION = {1: 0.00, 2: 0.10, 3: 0.03, 4: 0.02}
PHASE_WINDOW = {1: 20, 2: 20, 3: 20, 4: 30}


@dataclass
class PhaseConfig:
    phase: int
    mastery_threshold: float
    regression_threshold: float
    window_size: int


DEFAULT_PHASE_CONFIGS: dict[int, PhaseConfig] = {
    p: PhaseConfig(
        phase=p,
        mastery_threshold=PHASE_MASTERY[p],
        regression_threshold=PHASE_REGRESSION[p],
        window_size=PHASE_WINDOW[p],
    )
    for p in range(1, 5)
}


@dataclass
class _CurriculumState:
    current_phase: int = 1
    episode_history: dict[int, deque] = field(default_factory=dict)
    phase_transitions: list[dict] = field(default_factory=list)
    total_episodes: int = 0


class CurriculumController:
    def __init__(
        self,
        phase_configs: Optional[dict[int, PhaseConfig]] = None,
        start_phase: int = 1,
    ) -> None:
        self._configs = phase_configs or DEFAULT_PHASE_CONFIGS
        self._state = _CurriculumState(current_phase=start_phase)
        for p in self._configs:
            if p not in self._state.episode_history:
                self._state.episode_history[p] = deque(
                    maxlen=self._configs[p].window_size,
                )

    @property
    def current_phase(self) -> int:
        return self._state.current_phase

    @property
    def total_episodes(self) -> int:
        return self._state.total_episodes

    def success_rate(self, phase: Optional[int] = None) -> float:
        phase = phase or self._state.current_phase
        history = self._state.episode_history.get(phase, deque())
        if not history:
            return 0.0
        return sum(history) / len(history)

    def record(self, success: bool, task_id: str = "") -> None:
        phase = self._state.current_phase
        self._state.episode_history.setdefault(
            phase, deque(maxlen=self._configs[phase].window_size),
        ).append(success)
        self._state.total_episodes += 1
        self._check_transition(task_id)

    def phase_summary(self) -> dict:
        return {
            "current_phase": self._state.current_phase,
            "total_episodes": self._state.total_episodes,
            "success_rates": {p: self.success_rate(p) for p in self._configs},
            "window_sizes": {
                p: len(self._state.episode_history.get(p, deque()))
                for p in self._configs
            },
            "transitions": self._state.phase_transitions,
        }

    def save(self, path: str | Path) -> None:
        state_dict = {
            "current_phase": self._state.current_phase,
            "total_episodes": self._state.total_episodes,
            "episode_history": {
                str(k): list(v) for k, v in self._state.episode_history.items()
            },
            "phase_transitions": self._state.phase_transitions,
        }
        Path(path).write_text(json.dumps(state_dict, indent=2), encoding="utf-8")
        logger.info("Curriculum state saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "CurriculumController":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ctrl = cls(start_phase=data["current_phase"])
        ctrl._state.total_episodes = data["total_episodes"]
        ctrl._state.episode_history = {
            int(k): deque(v, maxlen=ctrl._configs[int(k)].window_size)
            for k, v in data["episode_history"].items()
        }
        ctrl._state.phase_transitions = data.get("phase_transitions", [])
        logger.info(
            "Curriculum state loaded from %s (phase=%d, episodes=%d)",
            path, ctrl.current_phase, ctrl.total_episodes,
        )
        return ctrl

    def _check_transition(self, task_id: str) -> None:
        phase = self._state.current_phase
        config = self._configs.get(phase)
        if config is None:
            return
        history = self._state.episode_history.get(phase, deque())
        if len(history) < 5:
            return
        rate = self.success_rate(phase)
        if rate >= config.mastery_threshold and phase < 4:
            self._transition(phase + 1, "mastery", rate, task_id)
        elif rate < config.regression_threshold and phase > 1:
            self._transition(phase - 1, "regression", rate, task_id)

    def _transition(self, new_phase: int, reason: str, rate: float, task_id: str) -> None:
        old = self._state.current_phase
        self._state.current_phase = new_phase
        self._state.phase_transitions.append({
            "episode": self._state.total_episodes,
            "from_phase": old,
            "to_phase": new_phase,
            "reason": reason,
            "success_rate": rate,
            "task_id": task_id,
        })
        logger.info(
            "Curriculum transition: phase %d -> %d (%s) rate=%.2f at episode %d",
            old, new_phase, reason, rate, self._state.total_episodes,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Curriculum controller utilities.")
    parser.add_argument(
        "--history", type=str, default=None,
        help="Trajectory directory to compute phase placement from",
    )
    parser.add_argument(
        "--output", type=str, default="curriculum_state.json",
        help="Output path for curriculum state JSON",
    )
    parser.add_argument(
        "--start-phase", type=int, default=1,
        help="Starting curriculum phase (1-4)",
    )
    args = parser.parse_args()

    controller = CurriculumController(start_phase=args.start_phase)

    if args.history:
        from training.trajectory import TrajectoryDataset

        dataset = TrajectoryDataset.load_dir(args.history)
        for traj in dataset.trajectories:
            controller.record(traj.success, traj.task_id)

    summary = controller.phase_summary()
    print(json.dumps(summary, indent=2))
    controller.save(args.output)
    print(f"\nCurriculum state saved to {args.output}")


if __name__ == "__main__":
    main()
