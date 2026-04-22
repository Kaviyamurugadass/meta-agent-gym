# Sample Data

Pre-seeded trajectories and domain fixtures for offline RL / imitation learning / demos.

## Expected files (auto-generated once `training/rollout_collection.py` exists)

| File | Source | Purpose |
|---|---|---|
| `random_policy_20ep.jsonl` | `scripts/baseline_rollout.sh` | 20 episodes with random action policy — lower-bound baseline |
| `heuristic_policy_20ep.jsonl` | `scripts/baseline_rollout.sh` | 20 episodes with hand-crafted heuristic — informed baseline |
| `expert_trajectory.jsonl` | `training/benchmark.py` | Hand-authored "perfect walkthrough" per scenario |

## Generation

```bash
make baseline
# or
bash scripts/baseline_rollout.sh
```

## Loading in code

```python
from training.trajectory import TrajectoryDataset

dataset = TrajectoryDataset.load_dir("data/sample/")
print(dataset.summary())
```

## Fill-in on finale day

Replace these placeholder files with real domain trajectories:

1. Run `make baseline` to generate `random_*` and `heuristic_*` rollouts
2. Write `training/benchmark.py` to produce `expert_trajectory.jsonl`
3. Commit trajectories to the repo — judges can replay them without running training

## Why this matters

Pre-collected trajectories make the env immediately useful to researchers who want to try offline RL without spinning up compute. Scores "utility" / "reproducibility" points.
