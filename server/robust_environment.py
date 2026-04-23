"""Robust environment wrapper for onsite phase.

Key improvements:
1. Better error handling and recovery
2. Edge case detection and graceful degradation
3. State validation and consistency checks
4. Timeout and resource management
5. Comprehensive logging for debugging
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set
from contextlib import contextmanager

from models import (
    Action,
    ActionCommand,
    Observation,
    State,
    TaskSpec,
    AgentSpec,
    RuleViolation,
    ViolationSeverity,
)

try:
    from server.environment import Environment
    from server.rewards.reward import MetaAgentRewardComputer
    from server.rewards.enhanced_reward import EnhancedRewardComputer
except ImportError:
    Environment = None  # type: ignore[assignment,misc]
    MetaAgentRewardComputer = None  # type: ignore[assignment,misc]
    EnhancedRewardComputer = None  # type: ignore[assignment,misc]

logger = logging.getLogger("server.robust_environment")


class RobustEnvironment:
    """Robust wrapper around Environment for onsite phase.
    
    Provides:
    - Error recovery and graceful degradation
    - State validation and consistency
    - Timeout protection
    - Enhanced debugging
    """
    
    def __init__(self, use_enhanced_rewards: bool = True):
        if Environment is None:
            raise RuntimeError("Environment not available")
            
        self.env = Environment()
        self.use_enhanced_rewards = use_enhanced_rewards
        
        # State tracking for validation
        self._episode_history: List[Dict[str, Any]] = []
        self._last_valid_state: Optional[State] = None
        self._error_counts: Dict[str, int] = {}
        self._start_time: Optional[float] = None
        
        # Timeout protection
        self.max_episode_time = 300  # 5 minutes
        self.max_step_time = 30      # 30 seconds per step
        
        # Enhanced reward computer
        if use_enhanced_rewards and EnhancedRewardComputer:
            from models import RewardConfig
            config = RewardConfig()
            self.reward_computer = EnhancedRewardComputer(config)
        elif MetaAgentRewardComputer:
            from models import RewardConfig
            config = RewardConfig()
            self.reward_computer = MetaAgentRewardComputer(config)
        else:
            self.reward_computer = None
            
        logger.info("RobustEnvironment initialized with enhanced_rewards=%s", use_enhanced_rewards)

    def reset(self, scenario_name: Optional[str] = None) -> Observation:
        """Robust reset with error handling and validation."""
        try:
            with self._timeout_context("reset"):
                # Reset episode tracking
                self._episode_history = []
                self._start_time = time.time()
                
                # Call underlying environment
                obs = self.env.reset(scenario_name=scenario_name)
                
                # Validate observation
                self._validate_observation(obs, "reset")
                
                # Record state
                self._last_valid_state = self.env.state
                self._record_step("reset", None, obs)
                
                logger.info("Successfully reset scenario: %s", scenario_name or "default")
                return obs
                
        except Exception as e:
            logger.error("Reset failed for scenario %s: %s", scenario_name, e)
            return self._create_fallback_observation("reset", str(e))

    def step(self, action: Action) -> Observation:
        """Robust step with comprehensive error handling."""
        try:
            with self._timeout_context("step"):
                # Validate action
                self._validate_action(action)
                
                # Record pre-step state
                pre_step_state = self.env.state
                
                # Execute step
                obs = self.env.step(action)
                
                # Validate observation
                self._validate_observation(obs, "step")
                
                # Check for state consistency
                self._validate_state_consistency(pre_step_state, self.env.state, action)
                
                # Record step
                self._record_step("step", action, obs)
                
                # Update last valid state
                self._last_valid_state = self.env.state
                
                return obs
                
        except TimeoutError:
            logger.error("Step timeout for action: %s", action.command)
            self._increment_error_count("timeout")
            return self._create_fallback_observation("step", "Step timeout")
            
        except Exception as e:
            logger.error("Step failed for action %s: %s", action.command, e)
            self._increment_error_count("step_error")
            return self._create_fallback_observation("step", str(e))

    def _validate_action(self, action: Action) -> None:
        """Validate action before execution."""
        if not isinstance(action.command, ActionCommand):
            raise ValueError(f"Invalid command: {action.command}")
            
        # Check for suspicious patterns
        if action.command == ActionCommand.NOOP:
            noop_count = sum(1 for step in self._episode_history 
                           if step.get("action") and step["action"].command == ActionCommand.NOOP)
            if noop_count > 3:  # Too many noops
                logger.warning("Excessive noop actions detected: %d", noop_count)
                
        # Validate skill additions
        if action.command == ActionCommand.ADD_SKILL:
            skill = action.args.get("skill", "")
            if not skill or not isinstance(skill, str):
                raise ValueError("Invalid skill argument")
                
        # Validate prompt writing
        if action.command == ActionCommand.WRITE_PROMPT:
            prompt = action.args.get("prompt", "")
            if not isinstance(prompt, str):
                raise ValueError("Invalid prompt argument")
            if len(prompt) > 10000:  # Reasonable limit
                raise ValueError("Prompt too long")

    def _validate_observation(self, obs: Observation, context: str) -> None:
        """Validate observation structure and content."""
        if not isinstance(obs, Observation):
            raise ValueError(f"Invalid observation type: {type(obs)}")
            
        # Check required fields
        if not hasattr(obs, 'reward') or obs.reward is None:
            raise ValueError("Observation missing reward")
            
        if not hasattr(obs, 'done') or obs.done is None:
            raise ValueError("Observation missing done flag")
            
        # Validate reward is numeric
        try:
            float(obs.reward)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid reward value: {obs.reward}")
            
        # Check for negative reward spikes (potential errors)
        if obs.reward < -10:
            logger.warning("Large negative reward detected: %s", obs.reward)
            
        # Validate current_spec if present
        if hasattr(obs, 'current_spec') and obs.current_spec:
            self._validate_agent_spec(obs.current_spec)

    def _validate_agent_spec(self, spec: Dict[str, Any]) -> None:
        """Validate agent specification structure."""
        if not isinstance(spec, dict):
            raise ValueError("Agent spec must be a dictionary")
            
        # Check for reasonable field values
        if "name" in spec and spec["name"]:
            if not isinstance(spec["name"], str) or len(spec["name"]) > 100:
                raise ValueError("Invalid agent name")
                
        if "description" in spec and spec["description"]:
            if not isinstance(spec["description"], str) or len(spec["description"]) > 500:
                raise ValueError("Invalid agent description")
                
        if "skills" in spec and spec["skills"]:
            if not isinstance(spec["skills"], list):
                raise ValueError("Skills must be a list")
            if len(spec["skills"]) > 20:  # Reasonable limit
                raise ValueError("Too many skills")

    def _validate_state_consistency(
        self, 
        before: State, 
        after: State, 
        action: Action
    ) -> None:
        """Check state transition consistency."""
        # Basic consistency checks
        if after.step != before.step + 1:
            logger.warning("Step count inconsistency: %d -> %d", before.step, after.step)
            
        # Check for unexpected spec changes
        if action.command not in [ActionCommand.SET_NAME, ActionCommand.SET_DESCRIPTION, 
                                 ActionCommand.ADD_SKILL, ActionCommand.REMOVE_SKILL,
                                 ActionCommand.WRITE_PROMPT, ActionCommand.SET_MODEL]:
            # Spec shouldn't change for other actions
            if before.current_spec != after.current_spec:
                logger.warning("Unexpected spec change for action: %s", action.command)

    def _record_step(self, context: str, action: Optional[Action], obs: Observation) -> None:
        """Record step for debugging and analysis."""
        step_data = {
            "context": context,
            "timestamp": time.time(),
            "observation": {
                "reward": obs.reward,
                "done": obs.done,
                "step": getattr(obs, 'step', None),
                "summary": getattr(obs, 'summary', None),
            }
        }
        
        if action:
            step_data["action"] = {
                "command": action.command,
                "args": action.args,
            }
            
        self._episode_history.append(step_data)
        
        # Limit history size
        if len(self._episode_history) > 1000:
            self._episode_history = self._episode_history[-500:]

    def _create_fallback_observation(self, context: str, error_msg: str) -> Observation:
        """Create safe fallback observation on errors."""
        logger.info("Creating fallback observation for %s: %s", context, error_msg)
        
        # Use last valid state if available
        if self._last_valid_state:
            fallback_state = self._last_valid_state
        else:
            # Create minimal valid state
            fallback_state = State(
                task_id="fallback",
                step=0,
                max_steps=7,
                current_spec={},
                available_skills=[],
                score=0.0,
                feedback=[f"System error: {error_msg}"],
                summary=f"Fallback mode: {error_msg}",
                done=True,  # End episode on error
                reward_breakdown={"total": 0.0},
                rule_violations=[],
            )
            
        return Observation(
            done=fallback_state.done,
            reward=0.0,  # Neutral reward on error
            metadata={"error": error_msg, "fallback": True},
            **fallback_state.model_dump()
        )

    @contextmanager
    def _timeout_context(self, operation: str):
        """Context manager for operation timeout."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if duration > self.max_step_time:
                logger.warning("Operation %s took %.2fs (limit: %.2fs)", 
                             operation, duration, self.max_step_time)

    def _increment_error_count(self, error_type: str) -> None:
        """Track error occurrences."""
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        
        # Log if errors are frequent
        if self._error_counts[error_type] > 5:
            logger.error("High error count for %s: %d", error_type, self._error_counts[error_type])

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for the current episode."""
        if not self._episode_history:
            return {"status": "no_episode"}
            
        total_reward = sum(step["observation"]["reward"] for step in self._episode_history 
                          if step["context"] == "step")
        
        action_counts = {}
        for step in self._episode_history:
            if step.get("action"):
                cmd = step["action"]["command"]
                action_counts[cmd] = action_counts.get(cmd, 0) + 1
                
        return {
            "total_steps": len([s for s in self._episode_history if s["context"] == "step"]),
            "total_reward": total_reward,
            "action_distribution": action_counts,
            "error_counts": dict(self._error_counts),
            "episode_duration": time.time() - self._start_time if self._start_time else 0,
        }

    @property
    def state(self) -> State:
        """Get current environment state."""
        return self.env.state


# Factory function for easy integration
def create_robust_environment(use_enhanced_rewards: bool = True) -> RobustEnvironment:
    """Create robust environment instance."""
    return RobustEnvironment(use_enhanced_rewards=use_enhanced_rewards)
