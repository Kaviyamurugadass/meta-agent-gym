"""Training pipeline smoke tests.

Covers: trajectory serialization, rollout collection, evaluation,
reward backend abstraction, benchmark runner. Excludes TRL/Unsloth
actual training (needs GPU) — just verifies dry-runs import cleanly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from models import Action, ActionCommand


# ---------------------------------------------------------------------------
# Trajectory tests
# ---------------------------------------------------------------------------


def test_trajectory_step_roundtrip():
    from training.trajectory import TrajectoryStep
    s = TrajectoryStep(
        step=1,
        action={"command": "noop"},
        observation={"step": 1},
        reward=0.5,
    )
    loaded = TrajectoryStep.model_validate_json(s.model_dump_json())
    assert loaded.step == 1
    assert loaded.reward == 0.5


def test_trajectory_append_increments_total_reward():
    from training.trajectory import Trajectory, TrajectoryStep
    traj = Trajectory(task_id="t1")
    traj.append(TrajectoryStep(step=1, action={}, observation={}, reward=0.3))
    traj.append(TrajectoryStep(step=2, action={}, observation={}, reward=0.4))
    assert traj.length == 2
    assert traj.total_reward == pytest.approx(0.7)


def test_trajectory_dataset_save_load(tmp_path):
    from training.trajectory import Trajectory, TrajectoryDataset, TrajectoryStep
    ds = TrajectoryDataset()
    t = Trajectory(task_id="t1")
    t.append(TrajectoryStep(step=1, action={}, observation={}, reward=0.5))
    ds.append(t)
    ds.save_dir(tmp_path)
    loaded = TrajectoryDataset.load_dir(tmp_path)
    assert len(loaded) == 1
    assert loaded.trajectories[0].total_reward == 0.5


def test_trajectory_dataset_summary_on_empty():
    from training.trajectory import TrajectoryDataset
    assert TrajectoryDataset().summary() == {"n": 0}


def test_trajectory_dataset_filter_successful():
    from training.trajectory import Trajectory, TrajectoryDataset
    ds = TrajectoryDataset(trajectories=[
        Trajectory(task_id="a", success=True),
        Trajectory(task_id="b", success=False),
    ])
    assert len(ds.filter_successful()) == 1


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


def test_random_policy_returns_valid_action():
    import random
    from training.rollout_collection import random_policy
    action = random_policy({}, random.Random(42))
    assert isinstance(action, Action)
    assert action.command in ActionCommand


def test_heuristic_policy_submits_near_end():
    import random
    from training.rollout_collection import heuristic_policy
    obs = {"step": 4, "max_steps": 5}
    action = heuristic_policy(obs, random.Random(42))
    assert action.command == ActionCommand.SUBMIT


def test_heuristic_policy_fills_name_first():
    """With an empty spec, the heuristic's first action is SET_NAME."""
    import random
    from training.rollout_collection import heuristic_policy
    obs = {"step": 1, "max_steps": 10, "current_spec": {}}
    action = heuristic_policy(obs, random.Random(42))
    assert action.command == ActionCommand.SET_NAME


def test_heuristic_policy_noops_when_spec_complete():
    """Once every required field is set, the heuristic coasts on NOOP until SUBMIT."""
    import random
    from training.rollout_collection import heuristic_policy
    obs = {
        "step": 3,
        "max_steps": 10,
        "current_spec": {
            "name": "x",
            "description": "y",
            "skills": ["web-scraping"],
            "system_prompt": "You are a specialist agent. " * 3,
            "model": "sonnet",
        },
    }
    action = heuristic_policy(obs, random.Random(42))
    assert action.command == ActionCommand.NOOP


def test_collect_saves_trajectories(tmp_path):
    from training.rollout_collection import collect
    ds = collect(
        episodes=2,
        policy="random",
        output_dir=tmp_path,
        seed=7,
        domain_randomise=False,
    )
    assert len(ds) == 2
    assert (tmp_path / "trajectory_0000.json").exists()
    assert (tmp_path / "index.json").exists()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_evaluation_online_metrics_empty():
    from training.evaluation import EvaluationSuite
    from training.trajectory import TrajectoryDataset
    metrics = EvaluationSuite.online_metrics(TrajectoryDataset())
    assert metrics == {"n": 0}


def test_evaluation_full_report_populates_sections(tmp_path):
    from training.evaluation import EvaluationSuite
    from training.rollout_collection import collect
    ds = collect(episodes=3, policy="heuristic", output_dir=tmp_path, seed=1, domain_randomise=False)
    report = EvaluationSuite.full_report(ds, label="test")
    assert report["label"] == "test"
    assert "mean_reward" in report["online"]
    assert "action_diversity" in report["behavior"]


# ---------------------------------------------------------------------------
# Reward backend
# ---------------------------------------------------------------------------


def test_local_backend_scores_action_sequence():
    from training.reward_backend import LocalBackend
    backend = LocalBackend()
    actions = [Action(command=ActionCommand.NOOP) for _ in range(3)]
    total, obs = backend.score(actions, scenario_name="placeholder_easy")
    assert isinstance(total, float)
    assert len(obs) >= 1  # at least the reset observation


def test_make_backend_local():
    from training.reward_backend import LocalBackend, make_backend
    b = make_backend("local")
    assert isinstance(b, LocalBackend)


def test_make_backend_remote_requires_url():
    from training.reward_backend import make_backend
    with pytest.raises(ValueError):
        make_backend("remote")


def test_make_backend_unknown_mode_raises():
    from training.reward_backend import make_backend
    with pytest.raises(ValueError):
        make_backend("cloud")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def test_benchmark_runs_placeholder_scenarios():
    from training.benchmark import run_benchmark, EXPERT_TRAJECTORIES
    for name in EXPERT_TRAJECTORIES:
        result = run_benchmark(name)
        assert result.scenario_name == name
        assert result.steps_taken >= 1
        assert 0.0 <= result.match_ratio <= 1.0


def test_benchmark_unknown_scenario_raises():
    from training.benchmark import run_benchmark
    with pytest.raises(ValueError):
        run_benchmark("scenario_that_doesnt_exist")


# ---------------------------------------------------------------------------
# GRPO script CLI dry-run — verifies arg parsing + reward backend wiring
# ---------------------------------------------------------------------------


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.parametrize("script", ["training/grpo_trl.py", "training/grpo_unsloth.py"])
def test_grpo_script_dry_run(script):
    """Dry-run each training script — no GPU, no training, just verify imports + wiring."""
    if script.endswith("grpo_unsloth.py") and not _has_cuda():
        pytest.skip("unsloth requires an NVIDIA/Intel GPU at import time")
    result = subprocess.run(
        [sys.executable, script, "--dry-run"],
        capture_output=True,
        text=True,
        timeout=180,  # unsloth cold-import + torch + triton can take 60–90s on first run
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "Dry-run passed" in result.stdout
