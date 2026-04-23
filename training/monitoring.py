"""Training monitor — per-component reward tracking and reporting.

Reads TrajectoryDataset from disk, extracts per-component reward breakdowns
from each step, and produces JSON reports + optional matplotlib plots.

CLI:
    python training/monitoring.py \\
        --input-dirs data/baseline/random data/baseline/heuristic \\
        --output-dir monitoring

    # No plots (skip matplotlib):
    python training/monitoring.py \\
        --input-dirs data/baseline/random \\
        --output-dir monitoring --no-plot
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from training.trajectory import Trajectory, TrajectoryDataset

# Core judge components to show in component plots
CORE_COMPONENTS = [
    "skill_selection",
    "description_quality",
    "workflow_clarity",
    "model_appropriateness",
    "best_practices",
    "efficiency",
]


@dataclass
class ComponentStats:
    name: str
    count: int
    mean: float
    std: float
    min_val: float
    max_val: float
    last_10_mean: float
    trend: float


class TrainingMonitor:
    def __init__(self, window: int = 50) -> None:
        self._window = window
        self._episodes: list[dict] = []

    def ingest_dir(self, dir_path: str | Path) -> None:
        dataset = TrajectoryDataset.load_dir(dir_path)
        self.ingest_dataset(dataset)

    def ingest_dataset(self, dataset: TrajectoryDataset) -> None:
        for traj in dataset.trajectories:
            self._ingest_trajectory(traj)

    def ingest_trajectory(self, traj: Trajectory) -> None:
        self._ingest_trajectory(traj)

    def _ingest_trajectory(self, traj: Trajectory) -> None:
        if not traj.steps:
            return
        component_sums: dict[str, float] = {}
        component_counts: dict[str, int] = {}
        for step in traj.steps:
            for k, v in step.reward_breakdown.items():
                component_sums[k] = component_sums.get(k, 0.0) + v
                component_counts[k] = component_counts.get(k, 0) + 1
        component_means = {
            k: component_sums[k] / component_counts[k]
            for k in component_sums
        }
        self._episodes.append({
            "episode": len(self._episodes),
            "total_reward": traj.total_reward,
            "components": component_means,
            "success": traj.success,
            "task_id": traj.task_id,
        })

    def component_names(self) -> list[str]:
        names: set[str] = set()
        for ep in self._episodes:
            names.update(ep["components"].keys())
        return sorted(names)

    def component_stats(self, component_name: Optional[str] = None) -> dict[str, ComponentStats]:
        names = [component_name] if component_name else self.component_names()
        result: dict[str, ComponentStats] = {}
        for name in names:
            values = [
                ep["components"][name]
                for ep in self._episodes
                if name in ep["components"]
            ]
            if not values:
                continue
            n = len(values)
            mean = statistics.mean(values)
            std = statistics.pstdev(values) if n > 1 else 0.0
            last_10 = values[-10:]
            trend = self._trend(values)
            result[name] = ComponentStats(
                name=name,
                count=n,
                mean=mean,
                std=std,
                min_val=min(values),
                max_val=max(values),
                last_10_mean=statistics.mean(last_10),
                trend=trend,
            )
        return result

    def total_reward_stats(self) -> Optional[ComponentStats]:
        values = [ep["total_reward"] for ep in self._episodes]
        if not values:
            return None
        n = len(values)
        return ComponentStats(
            name="total_reward",
            count=n,
            mean=statistics.mean(values),
            std=statistics.pstdev(values) if n > 1 else 0.0,
            min_val=min(values),
            max_val=max(values),
            last_10_mean=statistics.mean(values[-10:]),
            trend=self._trend(values),
        )

    def success_rate_over_time(self, window: Optional[int] = None) -> list[float]:
        w = window or self._window
        rates = []
        successes = [ep["success"] for ep in self._episodes]
        for i in range(len(successes)):
            start = max(0, i - w + 1)
            window_vals = successes[start:i + 1]
            rates.append(sum(window_vals) / len(window_vals))
        return rates

    def report(self) -> dict:
        total_stats = self.total_reward_stats()
        comp_stats = self.component_stats()
        success_rates = self.success_rate_over_time()
        return {
            "total_episodes": len(self._episodes),
            "total_reward": {
                "mean": total_stats.mean,
                "std": total_stats.std,
                "min": total_stats.min_val,
                "max": total_stats.max_val,
                "trend": total_stats.trend,
            } if total_stats else None,
            "success_rate": success_rates[-1] if success_rates else 0.0,
            "components": {
                name: {
                    "mean": cs.mean,
                    "std": cs.std,
                    "min": cs.min_val,
                    "max": cs.max_val,
                    "last_10_mean": cs.last_10_mean,
                    "trend": cs.trend,
                }
                for name, cs in comp_stats.items()
            },
        }

    def print_summary(self) -> None:
        n = len(self._episodes)
        if n == 0:
            print("No episodes ingested.")
            return
        total = self.total_reward_stats()
        rates = self.success_rate_over_time()
        print(f"\n{'='*60}")
        print(f"Training Monitor — {n} episodes")
        print(f"{'='*60}")
        if total:
            print(f"  Total reward:  mean={total.mean:.3f}  std={total.std:.3f}")
            print(f"                 min={total.min_val:.3f}  max={total.max_val:.3f}")
            print(f"                 trend={total.trend:+.4f}/ep")
        print(f"  Success rate:  {rates[-1]:.1%}" if rates else "  Success rate:  N/A")
        print(f"\n  Component breakdown:")
        comp_stats = self.component_stats()
        for name in sorted(comp_stats):
            cs = comp_stats[name]
            marker = "*" if name in CORE_COMPONENTS else " "
            print(f"  {marker} {name:30s} mean={cs.mean:+.3f}  trend={cs.trend:+.4f}")
        print(f"{'='*60}\n")

    def save_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.report(), indent=2),
            encoding="utf-8",
        )
        print(f"Report saved to {out}")

    def plot(self, output_dir: str | Path, title: Optional[str] = None) -> list[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Run: `uv sync --extra train`")
            return []

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        generated: list[Path] = []

        episodes = list(range(1, len(self._episodes) + 1))
        rewards = [ep["total_reward"] for ep in self._episodes]

        # 1. Total reward curve
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(episodes, rewards, marker="o", linewidth=1.5, markersize=4,
                color="#3b82f6", label="per-episode reward")
        if rewards:
            mean = statistics.mean(rewards)
            ax.axhline(mean, color="#94a3b8", linestyle="--", linewidth=1,
                       label=f"mean = {mean:.2f}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.set_title(title or f"Total Reward Curve ({len(self._episodes)} episodes)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        p1 = out / "total_reward_curve.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(p1)

        # 2. Component curves (subplots for core components)
        available_core = [c for c in CORE_COMPONENTS if c in self.component_names()]
        if available_core:
            ncols = 2
            nrows = (len(available_core) + 1) // 2
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows))
            axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
            palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#06b6d4"]
            for idx, comp_name in enumerate(available_core):
                ax_i = axes_flat[idx]
                values = [
                    ep["components"].get(comp_name, 0.0) for ep in self._episodes
                ]
                color = palette[idx % len(palette)]
                ax_i.plot(episodes, values, marker="o", linewidth=1.2, markersize=3,
                          color=color)
                trend = self._trend(values)
                ax_i.set_title(f"{comp_name} (trend={trend:+.4f})", fontsize=9)
                ax_i.grid(True, alpha=0.3)
                ax_i.set_ylabel("Score", fontsize=8)
            for idx in range(len(available_core), len(axes_flat)):
                axes_flat[idx].set_visible(False)
            fig.suptitle(title or "Component Reward Curves", fontsize=11)
            fig.tight_layout()
            p2 = out / "component_curves.png"
            fig.savefig(p2, dpi=150, bbox_inches="tight")
            plt.close(fig)
            generated.append(p2)

        # 3. Success rate curve
        rates = self.success_rate_over_time()
        if rates:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(episodes, rates, linewidth=2, color="#10b981")
            ax.fill_between(episodes, rates, alpha=0.15, color="#10b981")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Success rate (rolling)")
            ax.set_ylim(0, 1.05)
            ax.set_title(title or "Success Rate Over Time")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p3 = out / "success_rate_curve.png"
            fig.savefig(p3, dpi=150, bbox_inches="tight")
            plt.close(fig)
            generated.append(p3)

        for p in generated:
            print(f"Plot saved: {p}")
        return generated

    @staticmethod
    def _trend(values: list[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Training monitoring dashboard.")
    parser.add_argument("--input-dirs", nargs="+", required=True,
                        help="Directories of trajectory JSON files")
    parser.add_argument("--output-dir", default="monitoring",
                        help="Output directory for reports and plots")
    parser.add_argument("--window", type=int, default=50,
                        help="Rolling window size for success rate")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation (no matplotlib needed)")
    parser.add_argument("--title", default=None,
                        help="Custom title for plots")
    args = parser.parse_args()

    monitor = TrainingMonitor(window=args.window)
    for d in args.input_dirs:
        print(f"Ingesting: {d}")
        monitor.ingest_dir(d)

    monitor.print_summary()

    out = Path(args.output_dir)
    monitor.save_json(out / "report.json")

    if not args.no_plot:
        monitor.plot(out, title=args.title)


if __name__ == "__main__":
    main()
