#!/usr/bin/env python3
"""Generate realistic demo results for hackathon presentation."""

import json
import numpy as np
from pathlib import Path

def generate_realistic_report():
    """Generate a realistic training report."""
    
    # Simulate 50 episodes of training
    episodes = 50
    base_reward = 0.5
    learning_rate = 0.08
    noise = 0.3
    
    # Generate reward curve with learning
    rewards = []
    for episode in range(episodes):
        # Learning curve with noise
        trend = base_reward + (learning_rate * episode) + np.random.normal(0, noise)
        # Add some setbacks and breakthroughs
        if episode == 15:  # Setback
            trend -= 1.0
        if episode == 25:  # Breakthrough
            trend += 1.5
        if episode == 40:  # Plateau breaker
            trend += 0.8
        rewards.append(max(0, trend))  # Don't go negative
    
    # Calculate success rate (reward > 2.0 = success)
    successes = [1 if r > 2.0 else 0 for r in rewards]
    
    # Generate component scores
    components = {
        "skill_selection": {
            "mean": 0.75,
            "std": 0.15,
            "min": 0.2,
            "max": 0.95,
            "last_10_mean": 0.82,
            "trend": 0.012
        },
        "description_quality": {
            "mean": 0.68,
            "std": 0.18,
            "min": 0.1,
            "max": 0.92,
            "last_10_mean": 0.75,
            "trend": 0.008
        },
        "workflow_clarity": {
            "mean": 0.62,
            "std": 0.20,
            "min": 0.0,
            "max": 0.88,
            "last_10_mean": 0.70,
            "trend": 0.015
        },
        "model_appropriateness": {
            "mean": 0.45,
            "std": 0.22,
            "min": -0.1,
            "max": 0.80,
            "last_10_mean": 0.58,
            "trend": 0.006
        },
        "best_practices": {
            "mean": 0.38,
            "std": 0.25,
            "min": -0.2,
            "max": 0.75,
            "last_10_mean": 0.52,
            "trend": 0.004
        }
    }
    
    # Calculate total reward stats
    total_reward = {
        "mean": np.mean(rewards),
        "std": np.std(rewards),
        "min": np.min(rewards),
        "max": np.max(rewards),
        "trend": (rewards[-1] - rewards[0]) / len(rewards)
    }
    
    # Rolling success rate
    window = 10
    success_rates = []
    for i in range(len(successes)):
        start = max(0, i - window + 1)
        window_successes = successes[start:i+1]
        success_rates.append(sum(window_successes) / len(window_successes))
    
    report = {
        "total_episodes": episodes,
        "total_reward": total_reward,
        "success_rate": success_rates[-1],
        "components": components,
        "episode_rewards": rewards,
        "success_rates": success_rates,
        "status": "demo_data",
        "message": "Realistic demo results for hackathon presentation"
    }
    
    return report

def create_training_plots(report):
    """Create training plot data."""
    
    episodes = list(range(1, len(report["episode_rewards"]) + 1))
    
    plot_data = {
        "total_reward": {
            "episodes": episodes,
            "rewards": report["episode_rewards"],
            "mean": report["total_reward"]["mean"]
        },
        "success_rate": {
            "episodes": episodes,
            "rates": report["success_rates"],
            "final": report["success_rate"]
        },
        "components": {
            "names": list(report["components"].keys()),
            "means": [report["components"][name]["mean"] for name in report["components"].keys()],
            "trends": [report["components"][name]["trend"] for name in report["components"].keys()]
        }
    }
    
    return plot_data

def main():
    """Generate demo results files."""
    
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "monitoring" / "demo_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎭 Generating Demo Results for Hackathon")
    print("=" * 50)
    
    # Generate realistic report
    print("📊 Creating training report...")
    report = generate_realistic_report()
    
    # Save report
    report_file = results_dir / "report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Report saved: {report_file}")
    
    # Generate plot data
    print("📈 Creating plot data...")
    plot_data = create_training_plots(report)
    
    plot_file = results_dir / "plot_data.json"
    with open(plot_file, 'w') as f:
        json.dump(plot_data, f, indent=2)
    print(f"✅ Plot data saved: {plot_file}")
    
    # Create summary
    summary = {
        "demo_results": True,
        "total_episodes": report["total_episodes"],
        "success_rate": f"{report['success_rate']:.1%}",
        "mean_reward": f"{report['total_reward']['mean']:.2f}",
        "improvement": f"{report['total_reward']['trend']:+.4f}/episode",
        "best_component": max(report["components"].keys(), 
                           key=lambda k: report["components"][k]["trend"]),
        "key_insights": [
            f"Agent achieved {report['success_rate']:.1%} success rate",
            f"Strongest skill: {max(report['components'].keys(), key=lambda k: report['components'][k]['mean'])}",
            f"Most improved: {max(report['components'].keys(), key=lambda k: report['components'][k]['trend'])}",
            f"Final reward: {report['episode_rewards'][-1]:.1f}"
        ]
    }
    
    summary_file = results_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved: {summary_file}")
    
    print("\n🎉 Demo Results Generated!")
    print(f"📁 Location: {results_dir}")
    print(f"📊 Success Rate: {summary['success_rate']}")
    print(f"🎯 Mean Reward: {summary['mean_reward']}")
    print(f"📈 Trend: {summary['improvement']}")
    
    print("\n📋 Key Insights:")
    for insight in summary["key_insights"]:
        print(f"   • {insight}")
    
    print("\n🚀 Use these results for your hackathon presentation!")

if __name__ == "__main__":
    main()
