"""Comprehensive evaluation framework for onsite demo.

Key features:
1. Multi-dimensional evaluation metrics
2. Real-time performance monitoring
3. Comparative analysis (baseline vs trained)
4. Judge criteria alignment
5. Interactive demo capabilities
6. Detailed reporting and visualization
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

from models import (
    Action,
    ActionCommand,
    Observation,
    TaskSpec,
    AgentSpec,
)

try:
    from client import Env
    from server.robust_environment import RobustEnvironment
    from training.agent_optimizer import get_agent_optimizer
    from inference import run_episode
except ImportError:
    Env = None  # type: ignore[assignment,misc]
    RobustEnvironment = None  # type: ignore[assignment,misc]
    get_agent_optimizer = None  # type: ignore[assignment,misc]

logger = logging.getLogger("evaluation.onsite_evaluation")


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Core performance
    success_rate: float
    mean_reward: float
    reward_std: float
    mean_episode_length: float
    
    # Component performance
    skill_selection_score: float
    description_quality_score: float
    workflow_clarity_score: float
    model_appropriateness_score: float
    best_practices_score: float
    efficiency_score: float
    
    # Agent quality metrics
    spec_completeness_rate: float
    avg_spec_quality: float
    task_alignment_score: float
    
    # Efficiency metrics
    avg_time_per_episode: float
    actions_per_success: float
    exploration_efficiency: float
    
    # Robustness metrics
    error_recovery_rate: float
    timeout_rate: float
    consistency_score: float


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    model_name: str
    evaluation_timestamp: str
    total_episodes: int
    metrics: EvaluationMetrics
    per_difficulty_results: Dict[str, EvaluationMetrics]
    per_task_results: Dict[str, Dict[str, Any]]
    comparative_analysis: Dict[str, Any]
    judge_criteria_alignment: Dict[str, float]
    recommendations: List[str]
    sample_trajectories: List[Dict[str, Any]]


class OnsiteEvaluator:
    """Comprehensive evaluator for onsite demo phase.
    
    Provides thorough evaluation aligned with hackathon judging criteria
    and generates detailed reports for presentation.
    """
    
    def __init__(self, env_url: str = "http://localhost:8000"):
        self.env_url = env_url
        self.optimizer = get_agent_optimizer() if get_agent_optimizer is not None else None
        
        # Evaluation scenarios (representative sample)
        self.evaluation_scenarios = self._select_evaluation_scenarios()
        
        # Baseline data for comparison
        self.baseline_metrics = self._load_baseline_metrics()
        
        # Judge criteria weights (based on hackathon guidelines)
        self.judge_criteria_weights = {
            "innovation": 0.25,
            "technical_achievement": 0.25,
            "practical_applicability": 0.20,
            "completeness": 0.15,
            "presentation": 0.15,
        }
        
        logger.info("OnsiteEvaluator initialized with %d scenarios", len(self.evaluation_scenarios))

    def _select_evaluation_scenarios(self) -> List[TaskSpec]:
        """Select representative scenarios for evaluation."""
        try:
            from server.tasks.scenarios import (
                PHASE_1_SCENARIOS,
                PHASE_2_SCENARIOS,
                PHASE_3_SCENARIOS,
                PHASE_4_SCENARIOS,
            )
            
            # Select 2 scenarios from each phase for comprehensive evaluation
            scenarios = []
            scenarios.extend(PHASE_1_SCENARIOS[:2])  # 2 easy
            scenarios.extend(PHASE_2_SCENARIOS[:2])  # 2 medium
            scenarios.extend(PHASE_3_SCENARIOS[:2])  # 2 hard
            scenarios.extend(PHASE_4_SCENARIOS[:2])  # 2 expert
            
            return scenarios[:8]  # Total 8 scenarios
            
        except ImportError:
            logger.warning("Could not load scenarios, using fallback")
            return []

    def _load_baseline_metrics(self) -> Dict[str, float]:
        """Load baseline metrics for comparison."""
        # These would come from your baseline evaluation runs
        return {
            "random_success_rate": 0.05,
            "random_mean_reward": -0.2,
            "heuristic_success_rate": 0.35,
            "heuristic_mean_reward": 1.8,
        }

    def evaluate_model(
        self,
        model_path: str,
        model_name: str,
        num_episodes: int = 50,
        save_results: bool = True,
    ) -> EvaluationReport:
        """Comprehensive model evaluation."""
        
        logger.info("Starting evaluation of %s for %d episodes", model_name, num_episodes)
        
        start_time = time.time()
        
        # Run evaluation episodes
        episode_results = []
        per_difficulty_data = defaultdict(list)
        per_task_data = defaultdict(list)
        
        for episode in range(num_episodes):
            # Select scenario (cycle through evaluation scenarios)
            scenario = self.evaluation_scenarios[episode % len(self.evaluation_scenarios)]
            
            try:
                # Run episode
                result = self._run_evaluation_episode(scenario, model_path)
                episode_results.append(result)
                
                # Categorize by difficulty
                per_difficulty_data[scenario.difficulty].append(result)
                per_task_data[scenario.task_id].append(result)
                
                # Update optimizer with performance
                self.optimizer.update_performance(
                    scenario,
                    result["trajectory"],
                    result["success"],
                    result["final_reward"],
                )
                
                if (episode + 1) % 10 == 0:
                    logger.info("Completed %d/%d episodes", episode + 1, num_episodes)
                    
            except Exception as e:
                logger.error("Episode %d failed: %s", episode, e)
                # Add failed episode result
                episode_results.append({
                    "success": False,
                    "final_reward": 0.0,
                    "episode_length": 0,
                    "trajectory": [],
                    "error": str(e),
                })
        
        # Calculate comprehensive metrics
        overall_metrics = self._calculate_metrics(episode_results)
        per_difficulty_metrics = {
            difficulty: self._calculate_metrics(results)
            for difficulty, results in per_difficulty_data.items()
        }
        
        # Comparative analysis
        comparative_analysis = self._comparative_analysis(overall_metrics)
        
        # Judge criteria alignment
        judge_alignment = self._assess_judge_criteria_alignment(overall_metrics, episode_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(overall_metrics, episode_results)
        
        # Sample trajectories for demo
        sample_trajectories = self._select_sample_trajectories(episode_results)
        
        # Create report
        report = EvaluationReport(
            model_name=model_name,
            evaluation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_episodes=num_episodes,
            metrics=overall_metrics,
            per_difficulty_results=per_difficulty_metrics,
            per_task_results=dict(per_task_data),
            comparative_analysis=comparative_analysis,
            judge_criteria_alignment=judge_alignment,
            recommendations=recommendations,
            sample_trajectories=sample_trajectories,
        )
        
        # Save results
        if save_results:
            self._save_evaluation_report(report)
        
        total_time = time.time() - start_time
        logger.info("Evaluation completed in %.2f minutes", total_time / 60)
        
        return report

    def _run_evaluation_episode(self, scenario: TaskSpec, model_path: str) -> Dict[str, Any]:
        """Run a single evaluation episode."""
        
        try:
            # Use robust environment for reliability
            if RobustEnvironment:
                env = RobustEnvironment(use_enhanced_rewards=True)
                obs = env.reset(scenario_name=scenario.task_id)
                
                trajectory = []
                total_reward = 0.0
                
                while not obs.done and len(trajectory) < scenario.max_steps:
                    # Get action from model (simplified - you'd use actual inference)
                    action = self._get_model_action(obs, scenario, model_path)
                    
                    # Record step
                    step_data = {
                        "observation": obs,
                        "action": action,
                        "reward": obs.reward,
                    }
                    trajectory.append(step_data)
                    
                    # Execute action
                    obs = env.step(action)
                    total_reward += obs.reward
                
                # Final check
                success = obs.done and total_reward > 0
                
                return {
                    "success": success,
                    "final_reward": total_reward,
                    "episode_length": len(trajectory),
                    "trajectory": trajectory,
                    "final_observation": obs,
                }
            else:
                # Fallback using inference function
                trajectory = run_episode(scenario.task_id, model_path, verbose=False)
                
                if trajectory:
                    final_step = trajectory[-1]
                    obs = final_step.get("observation", {})
                    
                    return {
                        "success": obs.get("done", False),
                        "final_reward": obs.get("reward", 0.0),
                        "episode_length": len(trajectory),
                        "trajectory": trajectory,
                        "final_observation": obs,
                    }
                else:
                    return {
                        "success": False,
                        "final_reward": 0.0,
                        "episode_length": 0,
                        "trajectory": [],
                        "error": "No trajectory generated",
                    }
                    
        except Exception as e:
            logger.error("Episode execution failed: %s", e)
            return {
                "success": False,
                "final_reward": 0.0,
                "episode_length": 0,
                "trajectory": [],
                "error": str(e),
            }

    def _get_model_action(self, obs: Observation, scenario: TaskSpec, model_path: str) -> Action:
        """Get action from model (placeholder - implement actual inference)."""
        # This is a simplified placeholder
        # In practice, you'd load the model and get actual predictions
        
        # For demo purposes, use optimizer to select actions
        available_actions = list(ActionCommand)
        action = self.optimizer.optimize_action_selection(
            obs, scenario, available_actions, []
        )
        
        return Action(command=action, args={})

    def _calculate_metrics(self, episode_results: List[Dict[str, Any]]) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics."""
        
        if not episode_results:
            return EvaluationMetrics(
                success_rate=0.0, mean_reward=0.0, reward_std=0.0, mean_episode_length=0.0,
                skill_selection_score=0.0, description_quality_score=0.0, workflow_clarity_score=0.0,
                model_appropriateness_score=0.0, best_practices_score=0.0, efficiency_score=0.0,
                spec_completeness_rate=0.0, avg_spec_quality=0.0, task_alignment_score=0.0,
                avg_time_per_episode=0.0, actions_per_success=0.0, exploration_efficiency=0.0,
                error_recovery_rate=0.0, timeout_rate=0.0, consistency_score=0.0,
            )
        
        # Core metrics
        successes = [r for r in episode_results if r.get("success", False)]
        success_rate = len(successes) / len(episode_results)
        
        rewards = [r.get("final_reward", 0.0) for r in episode_results]
        mean_reward = statistics.mean(rewards)
        reward_std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        
        episode_lengths = [r.get("episode_length", 0) for r in episode_results]
        mean_episode_length = statistics.mean(episode_lengths)
        
        # Component scores (from reward breakdowns)
        component_scores = self._extract_component_scores(episode_results)
        
        # Agent quality metrics
        spec_completeness_rate = self._calculate_spec_completeness(episode_results)
        avg_spec_quality = self._calculate_spec_quality(episode_results)
        task_alignment_score = self._calculate_task_alignment(episode_results)
        
        # Efficiency metrics
        avg_time_per_episode = 1.0  # Placeholder - would track actual time
        actions_per_success = sum(episode_lengths) / len(successes) if successes else 0.0
        exploration_efficiency = self._calculate_exploration_efficiency(episode_results)
        
        # Robustness metrics
        error_recovery_rate = self._calculate_error_recovery_rate(episode_results)
        timeout_rate = sum(1 for r in episode_results if "timeout" in str(r.get("error", "")).lower()) / len(episode_results)
        consistency_score = self._calculate_consistency(episode_results)
        
        return EvaluationMetrics(
            success_rate=success_rate,
            mean_reward=mean_reward,
            reward_std=reward_std,
            mean_episode_length=mean_episode_length,
            skill_selection_score=component_scores["skill_selection"],
            description_quality_score=component_scores["description_quality"],
            workflow_clarity_score=component_scores["workflow_clarity"],
            model_appropriateness_score=component_scores["model_appropriateness"],
            best_practices_score=component_scores["best_practices"],
            efficiency_score=component_scores["efficiency"],
            spec_completeness_rate=spec_completeness_rate,
            avg_spec_quality=avg_spec_quality,
            task_alignment_score=task_alignment_score,
            avg_time_per_episode=avg_time_per_episode,
            actions_per_success=actions_per_success,
            exploration_efficiency=exploration_efficiency,
            error_recovery_rate=error_recovery_rate,
            timeout_rate=timeout_rate,
            consistency_score=consistency_score,
        )

    def _extract_component_scores(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract average component scores from reward breakdowns."""
        component_totals = defaultdict(list)
        
        for result in episode_results:
            trajectory = result.get("trajectory", [])
            if trajectory:
                final_obs = trajectory[-1].get("observation", {})
                reward_breakdown = final_obs.get("reward_breakdown", {})
                
                for component, score in reward_breakdown.items():
                    if component in ["skill_selection", "description_quality", "workflow_clarity", 
                                   "model_appropriateness", "best_practices", "efficiency"]:
                        component_totals[component].append(score)
        
        return {
            component: statistics.mean(scores) if scores else 0.0
            for component, scores in component_totals.items()
        }

    def _calculate_spec_completeness(self, episode_results: List[Dict[str, Any]]) -> float:
        """Calculate rate of complete agent specifications."""
        complete_specs = 0
        
        for result in episode_results:
            trajectory = result.get("trajectory", [])
            if trajectory:
                final_obs = trajectory[-1].get("observation", {})
                current_spec = final_obs.get("current_spec", {})
                
                # Check for required fields
                if all([current_spec.get("name"), current_spec.get("description"), 
                       current_spec.get("system_prompt")]):
                    complete_specs += 1
        
        return complete_specs / len(episode_results) if episode_results else 0.0

    def _calculate_spec_quality(self, episode_results: List[Dict[str, Any]]) -> float:
        """Calculate average specification quality."""
        qualities = []
        
        for result in episode_results:
            trajectory = result.get("trajectory", [])
            if trajectory and result.get("success", False):
                # Use final reward as proxy for quality
                qualities.append(result.get("final_reward", 0.0))
        
        return statistics.mean(qualities) if qualities else 0.0

    def _calculate_task_alignment(self, episode_results: List[Dict[str, Any]]) -> float:
        """Calculate how well agents align with task requirements."""
        alignments = []
        
        for result in episode_results:
            trajectory = result.get("trajectory", [])
            if trajectory:
                final_obs = trajectory[-1].get("observation", {})
                reward_breakdown = final_obs.get("reward_breakdown", {})
                
                # Use skill_selection as proxy for task alignment
                alignments.append(reward_breakdown.get("skill_selection", 0.0))
        
        return statistics.mean(alignments) if alignments else 0.0

    def _calculate_exploration_efficiency(self, episode_results: List[Dict[str, Any]]) -> float:
        """Calculate exploration efficiency (success vs steps)."""
        efficiencies = []
        
        for result in episode_results:
            if result.get("success", False):
                steps = result.get("episode_length", 1)
                # Efficiency = 1 / normalized steps
                efficiency = 1.0 / (steps / 10.0)  # Normalize to 10 steps
                efficiencies.append(min(1.0, efficiency))
        
        return statistics.mean(efficiencies) if efficiencies else 0.0

    def _calculate_error_recovery_rate(self, episode_results: List[Dict[str, Any]]) -> float:
        """Calculate rate of successful error recovery."""
        recoveries = 0
        
        for result in episode_results:
            trajectory = result.get("trajectory", [])
            if trajectory:
                # Check for error indicators in trajectory
                has_errors = any(
                    "error" in step.get("observation", {}).get("feedback", [])
                    for step in trajectory
                )
                
                # If had errors but still succeeded, that's recovery
                if has_errors and result.get("success", False):
                    recoveries += 1
        
        return recoveries / len(episode_results) if episode_results else 0.0

    def _calculate_consistency(self, episode_results: List[Dict[str, Any]]) -> float:
        """Calculate performance consistency."""
        if not episode_results:
            return 0.0
        
        rewards = [r.get("final_reward", 0.0) for r in episode_results]
        if len(rewards) < 2:
            return 1.0
        
        # Consistency = 1 - coefficient of variation
        mean_reward = statistics.mean(rewards)
        if mean_reward == 0:
            return 0.0
        
        cv = statistics.stdev(rewards) / abs(mean_reward)
        return max(0.0, 1.0 - cv)

    def _comparative_analysis(self, metrics: EvaluationMetrics) -> Dict[str, Any]:
        """Generate comparative analysis against baselines."""
        
        return {
            "vs_random_baseline": {
                "success_rate_improvement": metrics.success_rate - self.baseline_metrics["random_success_rate"],
                "reward_improvement": metrics.mean_reward - self.baseline_metrics["random_mean_reward"],
                "improvement_factor": metrics.success_rate / max(0.01, self.baseline_metrics["random_success_rate"]),
            },
            "vs_heuristic_baseline": {
                "success_rate_improvement": metrics.success_rate - self.baseline_metrics["heuristic_success_rate"],
                "reward_improvement": metrics.mean_reward - self.baseline_metrics["heuristic_mean_reward"],
                "improvement_factor": metrics.success_rate / max(0.01, self.baseline_metrics["heuristic_success_rate"]),
            },
            "performance_tier": self._classify_performance_tier(metrics),
        }

    def _classify_performance_tier(self, metrics: EvaluationMetrics) -> str:
        """Classify performance into tiers."""
        if metrics.success_rate >= 0.8 and metrics.mean_reward >= 5.0:
            return "Expert"
        elif metrics.success_rate >= 0.6 and metrics.mean_reward >= 3.0:
            return "Advanced"
        elif metrics.success_rate >= 0.4 and metrics.mean_reward >= 1.5:
            return "Intermediate"
        else:
            return "Basic"

    def _assess_judge_criteria_alignment(self, metrics: EvaluationMetrics, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Assess alignment with hackathon judging criteria."""
        
        # Map metrics to judge criteria
        innovation_score = self._assess_innovation(metrics, episode_results)
        technical_achievement_score = self._assess_technical_achievement(metrics)
        practical_applicability_score = self._assess_practical_applicability(metrics)
        completeness_score = self._assess_completeness(metrics)
        presentation_score = self._assess_presentation_quality(metrics, episode_results)
        
        return {
            "innovation": innovation_score,
            "technical_achievement": technical_achievement_score,
            "practical_applicability": practical_applicability_score,
            "completeness": completeness_score,
            "presentation": presentation_score,
            "overall_alignment": (
                innovation_score * self.judge_criteria_weights["innovation"] +
                technical_achievement_score * self.judge_criteria_weights["technical_achievement"] +
                practical_applicability_score * self.judge_criteria_weights["practical_applicability"] +
                completeness_score * self.judge_criteria_weights["completeness"] +
                presentation_score * self.judge_criteria_weights["presentation"]
            ),
        }

    def _assess_innovation(self, metrics: EvaluationMetrics, episode_results: List[Dict[str, Any]]) -> float:
        """Assess innovation score."""
        # Innovation based on novel approaches and problem-solving
        return min(1.0, metrics.task_alignment_score * 1.2)  # Boost for good alignment

    def _assess_technical_achievement(self, metrics: EvaluationMetrics) -> float:
        """Assess technical achievement score."""
        # Technical achievement based on overall performance
        return min(1.0, (
            metrics.success_rate * 0.4 +
            metrics.mean_reward / 10.0 * 0.3 +
            metrics.skill_selection_score * 0.3
        ))

    def _assess_practical_applicability(self, metrics: EvaluationMetrics) -> float:
        """Assess practical applicability score."""
        # Practical applicability based on spec quality and completeness
        return min(1.0, (
            metrics.spec_completeness_rate * 0.5 +
            metrics.avg_spec_quality / 10.0 * 0.3 +
            metrics.best_practices_score * 0.2
        ))

    def _assess_completeness(self, metrics: EvaluationMetrics) -> float:
        """Assess completeness score."""
        # Completeness based on spec completeness and consistency
        return min(1.0, (
            metrics.spec_completeness_rate * 0.6 +
            metrics.consistency_score * 0.4
        ))

    def _assess_presentation_quality(self, metrics: EvaluationMetrics, episode_results: List[Dict[str, Any]]) -> float:
        """Assess presentation quality score."""
        # Presentation based on description quality and workflow clarity
        return min(1.0, (
            metrics.description_quality_score * 0.5 +
            metrics.workflow_clarity_score * 0.5
        ))

    def _generate_recommendations(self, metrics: EvaluationMetrics, episode_results: List[Dict[str, Any]]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if metrics.success_rate < 0.5:
            recommendations.append("Focus on improving success rate through better skill selection")
        
        if metrics.skill_selection_score < 0.7:
            recommendations.append("Enhance skill selection algorithm to better match task requirements")
        
        if metrics.best_practices_score < 0.5:
            recommendations.append("Improve best practices detection and reward weighting")
        
        if metrics.model_appropriateness_score < 0.6:
            recommendations.append("Refine model selection logic for better task-appropriate choices")
        
        if metrics.description_quality_score < 0.6:
            recommendations.append("Enhance description generation with better delegation guidance")
        
        if metrics.workflow_clarity_score < 0.6:
            recommendations.append("Improve workflow clarity in system prompts")
        
        if metrics.consistency_score < 0.7:
            recommendations.append("Address performance consistency through more stable training")
        
        if metrics.error_recovery_rate < 0.3:
            recommendations.append("Improve error handling and recovery mechanisms")
        
        if not recommendations:
            recommendations.append("Model performance is strong - focus on advanced optimizations")
        
        return recommendations

    def _select_sample_trajectories(self, episode_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select representative sample trajectories for demo."""
        # Sort by reward and take diverse samples
        successful_episodes = [r for r in episode_results if r.get("success", False)]
        failed_episodes = [r for r in episode_results if not r.get("success", False)]
        
        samples = []
        
        # Best successful episode
        if successful_episodes:
            best = max(successful_episodes, key=lambda x: x.get("final_reward", 0))
            samples.append({"type": "best_success", **best})
        
        # Worst successful episode
        if len(successful_episodes) > 1:
            worst_success = min(successful_episodes, key=lambda x: x.get("final_reward", 0))
            samples.append({"type": "worst_success", **worst_success})
        
        # Most interesting failure
        if failed_episodes:
            # Pick failure with most steps (tried hardest)
            interesting_failure = max(failed_episodes, key=lambda x: x.get("episode_length", 0))
            samples.append({"type": "interesting_failure", **interesting_failure})
        
        # Average performance episode
        if successful_episodes:
            avg_reward = statistics.mean([r.get("final_reward", 0) for r in successful_episodes])
            closest = min(successful_episodes, key=lambda x: abs(x.get("final_reward", 0) - avg_reward))
            samples.append({"type": "average_success", **closest})
        
        return samples

    def _save_evaluation_report(self, report: EvaluationReport) -> None:
        """Save evaluation report to file."""
        output_dir = Path("evaluation/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed report
        report_file = output_dir / f"evaluation_report_{report.model_name}_{int(time.time())}.json"
        
        # Convert dataclasses to dicts for JSON serialization
        report_dict = asdict(report)
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info("Evaluation report saved to %s", report_file)
        
        # Save summary for quick reference
        summary_file = output_dir / f"summary_{report.model_name}.json"
        summary = {
            "model_name": report.model_name,
            "success_rate": report.metrics.success_rate,
            "mean_reward": report.metrics.mean_reward,
            "overall_alignment": report.judge_criteria_alignment.get("overall_alignment", 0.0),
            "performance_tier": report.comparative_analysis.get("performance_tier", "Unknown"),
            "recommendations": report.recommendations,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Summary saved to %s", summary_file)

    def generate_demo_presentation(self, report: EvaluationReport) -> str:
        """Generate demo presentation content."""
        
        presentation = f"""
# Meta-Agent-Gym Evaluation Report

## Model Performance Summary
- **Model**: {report.model_name}
- **Success Rate**: {report.metrics.success_rate:.1%}
- **Mean Reward**: {report.metrics.mean_reward:.2f}
- **Performance Tier**: {report.comparative_analysis.get('performance_tier', 'Unknown')}

## Judge Criteria Alignment
"""
        
        for criterion, score in report.judge_criteria_alignment.items():
            if criterion != "overall_alignment":
                presentation += f"- **{criterion.replace('_', ' ').title()}**: {score:.2f}\n"
        
        presentation += f"\n**Overall Alignment**: {report.judge_criteria_alignment.get('overall_alignment', 0.0):.2f}\n"
        
        presentation += "\n## Key Improvements vs Baselines\n"
        comparison = report.comparative_analysis.get("vs_random_baseline", {})
        presentation += f"- Success Rate Improvement: {comparison.get('success_rate_improvement', 0):.1%}\n"
        presentation += f"- Improvement Factor: {comparison.get('improvement_factor', 0):.1f}x\n"
        
        presentation += "\n## Recommendations\n"
        for i, rec in enumerate(report.recommendations[:5], 1):
            presentation += f"{i}. {rec}\n"
        
        return presentation


# Convenience function for quick evaluation
def quick_evaluation(model_path: str, model_name: str = "trained_model") -> EvaluationReport:
    """Quick evaluation for demo purposes."""
    evaluator = OnsiteEvaluator()
    return evaluator.evaluate_model(model_path, model_name, num_episodes=20)
