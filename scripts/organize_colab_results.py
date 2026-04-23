#!/usr/bin/env python3
"""Organize Colab results into the local project structure."""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

def extract_results(zip_dir: str = "results") -> None:
    """Extract and organize Colab results."""
    
    project_root = Path(__file__).parent.parent
    zip_path = project_root / zip_dir
    
    if not zip_path.exists():
        print(f"❌ Results directory not found: {zip_path}")
        print("Please place your Colab zip files in the 'results/' directory")
        return
    
    print("🚀 Organizing Colab results...")
    
    # Create organized directories
    colab_results_dir = project_root / "monitoring" / "colab_results"
    colab_trained_dir = project_root / "data" / "colab_trained"
    colab_model_dir = project_root / "models" / "colab_model"
    
    # Process results zip
    results_zip = zip_path / "meta-agent-gym-results.zip"
    if results_zip.exists():
        print("📦 Extracting results...")
        colab_results_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(results_zip, colab_results_dir)
        print(f"✅ Results extracted to {colab_results_dir}")
    else:
        print("⚠️  meta-agent-gym-results.zip not found")
    
    # Process trained data zip
    trained_zip = zip_path / "meta-agent-gym-trained.zip"
    if trained_zip.exists():
        print("📦 Extracting trained data...")
        colab_trained_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(trained_zip, colab_trained_dir)
        print(f"✅ Trained data extracted to {colab_trained_dir}")
    else:
        print("⚠️  meta-agent-gym-trained.zip not found")
    
    # Process model zip
    model_zip = zip_path / "meta-agent-gym-model.zip"
    if model_zip.exists():
        print("📦 Extracting trained model...")
        colab_model_dir.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(model_zip, colab_model_dir)
        print(f"✅ Model extracted to {colab_model_dir}")
    else:
        print("⚠️  meta-agent-gym-model.zip not found")
    
    # Process summary
    summary_file = zip_path / "summary.json"
    if summary_file.exists():
        print("📋 Processing summary...")
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"✅ Summary: {summary.get('status', 'Unknown')}")
        print(f"   Fixes applied: {len(summary.get('fixes_applied', []))}")
    
    print("\n🎉 Organization complete!")
    print("\nNext steps:")
    print("1. Check the extracted files")
    print("2. Update README.md with new results")
    print("3. Test the model if available")
    print("4. Prepare demo materials")

def analyze_results() -> None:
    """Analyze the extracted results and provide insights."""
    
    project_root = Path(__file__).parent.parent
    report_file = project_root / "monitoring" / "colab_results" / "report.json"
    
    if not report_file.exists():
        print("❌ No report.json found - run extraction first")
        return
    
    print("\n📊 Analyzing Results...")
    
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    # Key metrics
    total_episodes = report.get('total_episodes', 0)
    success_rate = report.get('success_rate', 0.0)
    
    print(f"\n📈 Key Metrics:")
    print(f"   Total Episodes: {total_episodes}")
    print(f"   Success Rate: {success_rate:.1%}")
    
    # Reward analysis
    total_reward = report.get('total_reward')
    if total_reward:
        print(f"\n🎯 Reward Analysis:")
        print(f"   Mean: {total_reward['mean']:.3f}")
        print(f"   Std:  {total_reward['std']:.3f}")
        print(f"   Min:  {total_reward['min']:.3f}")
        print(f"   Max:  {total_reward['max']:.3f}")
        print(f"   Trend: {total_reward['trend']:+.4f}/ep")
    
    # Component analysis
    components = report.get('components', {})
    if components:
        print(f"\n🔧 Component Performance:")
        for name, stats in sorted(components.items()):
            print(f"   {name:20s}: {stats['mean']:+.3f} (trend: {stats['trend']:+.4f})")
    
    # Performance assessment
    print(f"\n🎭 Performance Assessment:")
    if success_rate > 0.7:
        print("   ✅ Excellent success rate!")
    elif success_rate > 0.5:
        print("   ✅ Good success rate")
    elif success_rate > 0.3:
        print("   ⚠️  Moderate success rate - room for improvement")
    else:
        print("   ❌ Low success rate - needs more training")
    
    if total_reward and total_reward['trend'] > 0.01:
        print("   ✅ Positive learning trend")
    elif total_reward and total_reward['trend'] < -0.01:
        print("   ⚠️  Negative learning trend - check training")
    else:
        print("   📊 Flat learning curve")

def main() -> None:
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_results()
    else:
        extract_results()
        analyze_results()

if __name__ == "__main__":
    main()
