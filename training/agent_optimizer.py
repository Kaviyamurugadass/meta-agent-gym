"""Agent behavior optimization for onsite phase.

Key improvements:
1. Intelligent action sequencing
2. Adaptive exploration strategies
3. Failure pattern detection and recovery
4. Curriculum-aware behavior tuning
5. Performance-based parameter adaptation
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass

from models import (
    Action,
    ActionCommand,
    AgentSpec,
    TaskSpec,
    State,
)

logger = logging.getLogger("training.agent_optimizer")


@dataclass
class BehaviorProfile:
    """Profile for agent behavior patterns."""
    success_rate: float
    avg_steps_to_complete: float
    common_failure_points: List[int]
    preferred_action_sequence: List[ActionCommand]
    skill_selection_accuracy: float
    description_quality_score: float
    workflow_clarity_score: float


class AgentOptimizer:
    """Optimize agent behaviors for better task completion rates.
    
    Uses historical performance data to adapt behavior strategies
    and improve success rates across different task types.
    """
    
    def __init__(self):
        # Performance tracking
        self.task_profiles: Dict[str, BehaviorProfile] = {}
        self.global_stats = defaultdict(list)
        self.failure_patterns: Dict[str, List[Dict]] = defaultdict(list)
        
        # Adaptive parameters
        self.exploration_rates = {
            "easy": 0.1,
            "medium": 0.2,
            "hard": 0.3,
            "expert": 0.4,
        }
        
        # Action sequence templates
        self.sequence_templates = self._initialize_sequence_templates()
        
        # Recent performance window
        self.recent_performance = deque(maxlen=50)
        
        logger.info("AgentOptimizer initialized")

    def _initialize_sequence_templates(self) -> Dict[str, List[ActionCommand]]:
        """Initialize optimal action sequences by difficulty."""
        return {
            "easy": [
                ActionCommand.SET_NAME,
                ActionCommand.SET_DESCRIPTION,
                ActionCommand.ADD_SKILL,
                ActionCommand.SET_MODEL,
                ActionCommand.WRITE_PROMPT,
                ActionCommand.SUBMIT,
            ],
            "medium": [
                ActionCommand.SET_NAME,
                ActionCommand.SET_DESCRIPTION,
                ActionCommand.ADD_SKILL,
                ActionCommand.ADD_SKILL,  # Multiple skills
                ActionCommand.SET_MODEL,
                ActionCommand.WRITE_PROMPT,
                ActionCommand.SUBMIT,
            ],
            "hard": [
                ActionCommand.SET_NAME,
                ActionCommand.SET_DESCRIPTION,
                ActionCommand.ADD_SKILL,
                ActionCommand.ADD_SKILL,
                ActionCommand.ADD_SKILL,  # Multiple skills
                ActionCommand.SET_MODEL,
                ActionCommand.WRITE_PROMPT,
                ActionCommand.ADD_TOOLS,  # Additional complexity
                ActionCommand.SUBMIT,
            ],
            "expert": [
                ActionCommand.SET_NAME,
                ActionCommand.SET_DESCRIPTION,
                ActionCommand.ADD_SKILL,
                ActionCommand.ADD_SKILL,
                ActionCommand.ADD_SKILL,
                ActionCommand.ADD_SKILL,  # Many skills
                ActionCommand.SET_MODEL,
                ActionCommand.WRITE_PROMPT,
                ActionCommand.ADD_TOOLS,
                ActionCommand.SET_MEMORY,
                ActionCommand.SET_MAX_TURNS,
                ActionCommand.SUBMIT,
            ],
        }

    def optimize_action_selection(
        self,
        current_state: State,
        task: TaskSpec,
        available_actions: List[ActionCommand],
        performance_history: List[Dict],
    ) -> ActionCommand:
        """Select optimal action based on context and performance."""
        
        difficulty = task.difficulty
        current_step = current_state.step
        
        # Check for failure recovery patterns
        recovery_action = self._check_failure_recovery(current_state, task, performance_history)
        if recovery_action:
            logger.info("Using recovery action: %s", recovery_action)
            return recovery_action
        
        # Use template-based sequence for structured approach
        template_action = self._get_template_action(difficulty, current_step, current_state)
        if template_action and template_action in available_actions:
            return template_action
        
        # Adaptive exploration vs exploitation
        if self._should_explore(difficulty, current_step, performance_history):
            return self._exploration_action(available_actions, current_state, task)
        else:
            return self._exploitation_action(available_actions, current_state, task)

    def _check_failure_recovery(
        self,
        state: State,
        task: TaskSpec,
        history: List[Dict],
    ) -> Optional[ActionCommand]:
        """Check if we need recovery action based on failure patterns."""
        
        # Check for repeated failures
        recent_failures = [h for h in history[-5:] if not h.get("success", True)]
        if len(recent_failures) >= 3:
            # Pattern: agents getting stuck - try investigation
            if ActionCommand.CHECK_SCORE in [ActionCommand.NOOP, ActionCommand.CHECK_SCORE]:
                return ActionCommand.CHECK_SCORE
        
        # Check for incomplete specs near deadline
        if state.step >= state.max_steps - 2:
            spec = state.current_spec or {}
            missing_fields = []
            if not spec.get("name"):
                missing_fields.append(ActionCommand.SET_NAME)
            if not spec.get("description"):
                missing_fields.append(ActionCommand.SET_DESCRIPTION)
            if not spec.get("system_prompt"):
                missing_fields.append(ActionCommand.WRITE_PROMPT)
            
            if missing_fields:
                # Prioritize most critical missing field
                if ActionCommand.SET_NAME in missing_fields:
                    return ActionCommand.SET_NAME
                elif ActionCommand.WRITE_PROMPT in missing_fields:
                    return ActionCommand.WRITE_PROMPT
                else:
                    return missing_fields[0]
        
        return None

    def _get_template_action(
        self,
        difficulty: str,
        current_step: int,
        state: State,
    ) -> Optional[ActionCommand]:
        """Get action from optimal sequence template."""
        
        template = self.sequence_templates.get(difficulty, [])
        if current_step < len(template):
            template_action = template[current_step]
            
            # Check if action is still needed
            if template_action == ActionCommand.SET_NAME and state.current_spec.get("name"):
                return None  # Skip if already done
            elif template_action == ActionCommand.SET_DESCRIPTION and state.current_spec.get("description"):
                return None
            elif template_action == ActionCommand.ADD_SKILL and state.current_spec.get("skills"):
                # Check if we have enough skills
                required_skills = len(getattr(state, 'task', TaskSpec()).required_skills or [])
                current_skills = len(state.current_spec.get("skills", []))
                if current_skills >= required_skills:
                    return None
            
            return template_action
        
        return None

    def _should_explore(
        self,
        difficulty: str,
        current_step: int,
        history: List[Dict],
    ) -> bool:
        """Decide whether to explore or exploit."""
        
        # Early episodes: more exploration
        total_episodes = len(history)
        if total_episodes < 10:
            return random.random() < 0.3
        
        # Recent performance: adapt exploration rate
        recent_success_rate = sum(1 for h in history[-10:] if h.get("success", False)) / 10
        base_exploration = self.exploration_rates.get(difficulty, 0.2)
        
        # Reduce exploration if performing well
        if recent_success_rate > 0.8:
            return random.random() < (base_exploration * 0.5)
        elif recent_success_rate < 0.3:
            return random.random() < (base_exploration * 1.5)
        else:
            return random.random() < base_exploration

    def _exploration_action(
        self,
        available_actions: List[ActionCommand],
        state: State,
        task: TaskSpec,
    ) -> ActionCommand:
        """Select exploration action."""
        
        # Prefer investigation actions during exploration
        investigation_actions = [ActionCommand.CHECK_SCORE, ActionCommand.INSPECT_EXAMPLE]
        available_investigation = [a for a in investigation_actions if a in available_actions]
        
        if available_investigation and random.random() < 0.4:
            return random.choice(available_investigation)
        
        # Try different skill combinations
        if ActionCommand.ADD_SKILL in available_actions:
            if random.random() < 0.3:
                return ActionCommand.ADD_SKILL
        
        # Random valid action
        return random.choice(available_actions)

    def _exploitation_action(
        self,
        available_actions: List[ActionCommand],
        state: State,
        task: TaskSpec,
    ) -> ActionCommand:
        """Select exploitation action based on learned patterns."""
        
        # Get task profile
        task_id = task.task_id
        profile = self.task_profiles.get(task_id)
        
        if profile and profile.preferred_action_sequence:
            # Use learned preferred sequence
            for action in profile.preferred_action_sequence:
                if action in available_actions:
                    # Check if action makes sense in current context
                    if self._is_action_appropriate(action, state, task):
                        return action
        
        # Fallback to template-based exploitation
        template_action = self._get_template_action(task.difficulty, state.step, state)
        if template_action and template_action in available_actions:
            return template_action
        
        # Default to most critical needed action
        return self._get_critical_action(available_actions, state, task)

    def _is_action_appropriate(
        self,
        action: ActionCommand,
        state: State,
        task: TaskSpec,
    ) -> bool:
        """Check if action is appropriate in current context."""
        
        spec = state.current_spec or {}
        
        if action == ActionCommand.SET_NAME:
            return not spec.get("name")
        elif action == ActionCommand.SET_DESCRIPTION:
            return not spec.get("description")
        elif action == ActionCommand.ADD_SKILL:
            required_skills = task.required_skills or []
            current_skills = spec.get("skills", [])
            return len(current_skills) < len(required_skills)
        elif action == ActionCommand.WRITE_PROMPT:
            return not spec.get("system_prompt")
        elif action == ActionCommand.SUBMIT:
            # Only submit if we have required fields
            return all([spec.get("name"), spec.get("description"), spec.get("system_prompt")])
        
        return True

    def _get_critical_action(
        self,
        available_actions: List[ActionCommand],
        state: State,
        task: TaskSpec,
    ) -> ActionCommand:
        """Get most critical action needed."""
        
        spec = state.current_spec or {}
        
        # Priority order for missing critical components
        critical_actions = [
            (ActionCommand.SET_NAME, not spec.get("name")),
            (ActionCommand.WRITE_PROMPT, not spec.get("system_prompt") or len(spec.get("system_prompt", "")) < 50),
            (ActionCommand.SET_DESCRIPTION, not spec.get("description")),
            (ActionCommand.ADD_SKILL, len(spec.get("skills", [])) < len(task.required_skills or [])),
            (ActionCommand.SET_MODEL, not spec.get("model")),
            (ActionCommand.SUBMIT, all([spec.get("name"), spec.get("description"), spec.get("system_prompt")])),
        ]
        
        for action, condition in critical_actions:
            if condition and action in available_actions:
                return action
        
        # Fallback
        return available_actions[0] if available_actions else ActionCommand.NOOP

    def update_performance(
        self,
        task: TaskSpec,
        trajectory: List[Dict],
        success: bool,
        final_reward: float,
    ) -> None:
        """Update performance metrics and behavior profiles."""
        
        # Record performance
        performance_data = {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "success": success,
            "reward": final_reward,
            "steps": len(trajectory),
            "timestamp": len(self.global_stats["total_episodes"]),
        }
        
        self.recent_performance.append(performance_data)
        self.global_stats["total_episodes"].append(performance_data)
        
        # Update task-specific profile
        self._update_task_profile(task, trajectory, success, final_reward)
        
        # Update failure patterns
        if not success:
            self._record_failure_pattern(task, trajectory)
        
        # Adapt exploration rates based on performance
        self._adapt_exploration_rates()

    def _update_task_profile(
        self,
        task: TaskSpec,
        trajectory: List[Dict],
        success: bool,
        final_reward: float,
    ) -> None:
        """Update behavior profile for specific task."""
        
        task_id = task.task_id
        existing_profile = self.task_profiles.get(task_id)
        
        # Extract action sequence
        action_sequence = [step["action"]["command"] for step in trajectory if "action" in step]
        
        # Calculate metrics
        steps_to_complete = len(trajectory)
        skill_selection_accuracy = self._calculate_skill_selection_accuracy(task, trajectory)
        description_quality = self._calculate_description_quality(trajectory)
        workflow_clarity = self._calculate_workflow_clarity(trajectory)
        
        # Update or create profile
        if existing_profile:
            # Exponential moving average update
            alpha = 0.1
            existing_profile.success_rate = alpha * success + (1 - alpha) * existing_profile.success_rate
            existing_profile.avg_steps_to_complete = alpha * steps_to_complete + (1 - alpha) * existing_profile.avg_steps_to_complete
            existing_profile.skill_selection_accuracy = alpha * skill_selection_accuracy + (1 - alpha) * existing_profile.skill_selection_accuracy
            existing_profile.description_quality_score = alpha * description_quality + (1 - alpha) * existing_profile.description_quality_score
            existing_profile.workflow_clarity_score = alpha * workflow_clarity + (1 - alpha) * existing_profile.workflow_clarity_score
        else:
            self.task_profiles[task_id] = BehaviorProfile(
                success_rate=success,
                avg_steps_to_complete=steps_to_complete,
                common_failure_points=[],
                preferred_action_sequence=action_sequence,
                skill_selection_accuracy=skill_selection_accuracy,
                description_quality_score=description_quality,
                workflow_clarity_score=workflow_clarity,
            )

    def _calculate_skill_selection_accuracy(self, task: TaskSpec, trajectory: List[Dict]) -> float:
        """Calculate how well skills were selected for the task."""
        required_skills = set(task.required_skills or [])
        
        # Find final spec
        final_spec = None
        for step in reversed(trajectory):
            if "observation" in step and step["observation"].get("current_spec"):
                final_spec = step["observation"]["current_spec"]
                break
        
        if not final_spec:
            return 0.0
        
        selected_skills = set(final_spec.get("skills", []))
        
        # Calculate precision and recall
        if not required_skills:
            return 1.0  # No requirements = full credit
        
        precision = len(required_skills & selected_skills) / len(selected_skills) if selected_skills else 0.0
        recall = len(required_skills & selected_skills) / len(required_skills)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _calculate_description_quality(self, trajectory: List[Dict]) -> float:
        """Estimate description quality from trajectory."""
        # This is a simplified heuristic - in practice, you'd use the judge scores
        for step in reversed(trajectory):
            if "observation" in step and "reward_breakdown" in step["observation"]:
                breakdown = step["observation"]["reward_breakdown"]
                return breakdown.get("description_quality", 0.0)
        return 0.0

    def _calculate_workflow_clarity(self, trajectory: List[Dict]) -> float:
        """Estimate workflow clarity from trajectory."""
        for step in reversed(trajectory):
            if "observation" in step and "reward_breakdown" in step["observation"]:
                breakdown = step["observation"]["reward_breakdown"]
                return breakdown.get("workflow_clarity", 0.0)
        return 0.0

    def _record_failure_pattern(self, task: TaskSpec, trajectory: List[Dict]) -> None:
        """Record failure patterns for analysis."""
        pattern = {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "failure_step": len(trajectory),
            "final_spec": trajectory[-1].get("observation", {}).get("current_spec", {}),
            "missing_fields": [],
        }
        
        # Identify missing fields
        final_spec = pattern["final_spec"]
        if not final_spec.get("name"):
            pattern["missing_fields"].append("name")
        if not final_spec.get("description"):
            pattern["missing_fields"].append("description")
        if not final_spec.get("system_prompt"):
            pattern["missing_fields"].append("system_prompt")
        
        self.failure_patterns[task.task_id].append(pattern)

    def _adapt_exploration_rates(self) -> None:
        """Adapt exploration rates based on recent performance."""
        if len(self.recent_performance) < 10:
            return
        
        recent_success_rate = sum(1 for p in list(self.recent_performance)[-10:] if p["success"]) / 10
        
        # Adjust exploration rates
        for difficulty in self.exploration_rates:
            if recent_success_rate > 0.8:
                # Reduce exploration when doing well
                self.exploration_rates[difficulty] *= 0.95
            elif recent_success_rate < 0.3:
                # Increase exploration when struggling
                self.exploration_rates[difficulty] *= 1.05
            
            # Keep within reasonable bounds
            self.exploration_rates[difficulty] = max(0.05, min(0.5, self.exploration_rates[difficulty]))

    def get_optimization_report(self) -> Dict[str, any]:
        """Generate optimization performance report."""
        if not self.recent_performance:
            return {"status": "no_data"}
        
        recent_episodes = list(self.recent_performance)[-20:]
        success_rate = sum(1 for e in recent_episodes if e["success"]) / len(recent_episodes)
        avg_reward = sum(e["reward"] for e in recent_episodes) / len(recent_episodes)
        avg_steps = sum(e["steps"] for e in recent_episodes) / len(recent_episodes)
        
        return {
            "recent_episodes": len(recent_episodes),
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "task_profiles": len(self.task_profiles),
            "exploration_rates": dict(self.exploration_rates),
            "failure_patterns": sum(len(patterns) for patterns in self.failure_patterns.values()),
        }


# Singleton instance for easy access
_optimizer_instance: Optional[AgentOptimizer] = None


def get_agent_optimizer() -> AgentOptimizer:
    """Get global agent optimizer instance."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = AgentOptimizer()
    return _optimizer_instance
