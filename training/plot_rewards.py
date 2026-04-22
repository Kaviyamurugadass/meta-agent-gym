"""Plot reward curves from trajectory directories.

Produces a PNG per training run showing per-episode reward over time,
with optional trend line. Chain 2-3 runs together to tell a training
narrative (noisy → plateau → good-variance) in the README.

CLI:
    # Single run
    uv run python training/plot_rewards.py \\
        --input-dir data/trained \\
        --output training/plots/run_latest.png \\
        --title "Qwen3-0.6B + GRPO (Run 3)"

    # Compare multiple runs on one plot
    uv run python training/plot_rewards.py \\
        --compare data/baseline/heuristic data/trained \\
        --labels "Baseline" "Trained" \\
        --output training/plots/comparison.png

Requires: matplotlib (from `uv sync --extra train`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from training.trajectory import TrajectoryDataset


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "matplotlib not installed. Run: `uv sync --extra train`",
            file=sys.stderr,
        )
        sys.exit(1)


def _rewards_per_episode(dir_path: str | Path) -> list[float]:
    ds = TrajectoryDataset.load_dir(dir_path)
    return [t.total_reward for t in ds.trajectories]


def _trend_line(y: list[float]) -> tuple[list[float], float]:
    """Linear regression trend line. Returns (fitted_y, slope_per_episode)."""
    n = len(y)
    if n < 2:
        return list(y), 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(y) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(y))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den else 0.0
    intercept = y_mean - slope * x_mean
    return [slope * i + intercept for i in range(n)], slope


def plot_single(
    input_dir: str | Path,
    output_path: str | Path,
    title: Optional[str] = None,
    show_trend: bool = True,
) -> None:
    """Render a single reward curve with optional trend line."""
    _require_matplotlib()
    import matplotlib.pyplot as plt

    rewards = _rewards_per_episode(input_dir)
    if not rewards:
        print(f"no trajectories found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    episodes = list(range(1, len(rewards) + 1))
    mean = sum(rewards) / len(rewards)
    best = max(rewards)
    worst = min(rewards)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, rewards, marker="o", linewidth=1.5, markersize=5,
            color="#3b82f6", label="per-episode reward")
    ax.axhline(mean, color="#94a3b8", linestyle="--", linewidth=1,
               label=f"mean = {mean:.2f}")

    if show_trend and len(rewards) >= 2:
        trend, slope = _trend_line(rewards)
        ax.plot(episodes, trend, color="#ef4444", linestyle=":", linewidth=1.2,
                label=f"trend ({slope:+.3f}/ep)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(title or f"Reward curve — {len(rewards)} episodes")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    footer = f"best={best:.2f}  mean={mean:.2f}  worst={worst:.2f}  n={len(rewards)}"
    fig.text(0.5, -0.02, footer, ha="center", fontsize=8, color="#64748b")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def plot_compare(
    input_dirs: list[str | Path],
    labels: list[str],
    output_path: str | Path,
    title: Optional[str] = None,
) -> None:
    """Render multiple runs on one plot for side-by-side comparison."""
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if len(input_dirs) != len(labels):
        print("--compare and --labels must have the same length", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"]

    summary_lines = []
    for i, (dir_path, label) in enumerate(zip(input_dirs, labels)):
        rewards = _rewards_per_episode(dir_path)
        if not rewards:
            continue
        episodes = list(range(1, len(rewards) + 1))
        color = palette[i % len(palette)]
        mean = sum(rewards) / len(rewards)
        ax.plot(episodes, rewards, marker="o", linewidth=1.5, markersize=4,
                color=color, label=f"{label} (mean={mean:.2f})")
        summary_lines.append(f"{label}: n={len(rewards)} mean={mean:.2f} best={max(rewards):.2f}")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(title or "Training runs comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.axhline(0, color="#94a3b8", linewidth=0.5)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
    for line in summary_lines:
        print(f"  {line}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot reward curves from trajectory directories.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input-dir", help="Single run: directory of trajectory JSONs")
    mode.add_argument("--compare", nargs="+", help="Compare runs: multiple directories")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Labels for --compare (same order + count)")
    p.add_argument("--output", required=True, help="Output PNG path")
    p.add_argument("--title", default=None)
    p.add_argument("--no-trend", action="store_true",
                   help="Disable trend line (single-run mode only)")
    args = p.parse_args()

    if args.input_dir:
        plot_single(
            input_dir=args.input_dir,
            output_path=args.output,
            title=args.title,
            show_trend=not args.no_trend,
        )
    else:
        labels = args.labels or [Path(d).name for d in args.compare]
        plot_compare(
            input_dirs=args.compare,
            labels=labels,
            output_path=args.output,
            title=args.title,
        )


if __name__ == "__main__":
    main()
