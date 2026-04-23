"""Core OpenEnv environment — meta-agent generation.

META-AGENT EXTENSION:
    - Three-tier verification (hard verifiers → judge → real execution)
    - Command-based action space for agent generation
    - POMDP structure with hidden "optimal spec" state
    - Curriculum progression (1-skill → 5+ skills)

Structured logging tags [START]/[STEP]/[END] follow Round 1 requirement.
"""

from __future__ import annotations

import logging
import logging.config
import random
from pathlib import Path
from typing import Optional

import yaml

from models import (
    Action,
    ActionCommand,
    AgentSpec,
    Observation,
    RewardConfig,
    RuleViolation,
    State,
    TaskSpec,
)

# OpenEnv ABC
try:
    from openenv.core import Environment as _OpenEnvBase
except ImportError:
    _OpenEnvBase = object  # type: ignore[assignment,misc]

# Sibling-module imports
try:
    from server.rules.engine import RuleEngine
except ImportError:
    RuleEngine = None  # type: ignore[assignment,misc]

try:
    from server.rewards.reward import MetaAgentRewardComputer
except ImportError:
    MetaAgentRewardComputer = None  # type: ignore[assignment,misc]

try:
    from server.tasks.generator import TaskGenerator
except ImportError:
    TaskGenerator = None  # type: ignore[assignment,misc]

try:
    from server.tasks.scenarios import SCENARIOS, get_scenario
except ImportError:
    SCENARIOS = None  # type: ignore[assignment,misc]
    get_scenario = None  # type: ignore[assignment,misc]

try:
    from server.skills import AVAILABLE_SKILLS, get_curriculum_skills
except ImportError:
    AVAILABLE_SKILLS = {}
    get_curriculum_skills = None  # type: ignore[assignment,misc]

try:
    from server.verifiers import HardVerifiers
except ImportError:
    HardVerifiers = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOGGING_CONFIGURED = False


def _configure_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    config_path = Path(__file__).parent.parent / "training" / "logging.yaml"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                logging.config.dictConfig(yaml.safe_load(f))
        except Exception:
            logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)
    _LOGGING_CONFIGURED = True


_configure_logging()
logger = logging.getLogger("server.environment")


# ---------------------------------------------------------------------------
# Core environment class
# ---------------------------------------------------------------------------


class Environment(_OpenEnvBase):
    """Meta-agent generation environment.

    The environment trains a policy to generate AGENT.md files from task
    descriptions. Uses three-tier verification:
        1. Hard verifiers (100% of steps, free)
        2. Fast judge (90% of steps, ~$0.01)
        3. Real execution (steps 3, 6, 9, ~$1-10)

    Inherits from openenv.core.Environment. Required abstract methods:
        reset(seed, episode_id, **kwargs) -> Observation
        step(action, timeout_s, **kwargs) -> Observation
        state -> State
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    # META-AGENT: Steps to run real execution (10% of steps)
    REAL_EXEC_STEPS = [3, 6, 9]

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        domain_randomise: bool = True,
        seed: Optional[int] = None,
        curriculum_phase: int = 1,  # META-AGENT: Curriculum phase (1-4)
    ) -> None:
        if _OpenEnvBase is not object:
            try:
                super().__init__()
            except TypeError:
                pass

        self.reward_config = reward_config or RewardConfig()
        self.domain_randomise = domain_randomise
        self.seed = seed
        self._rng = random.Random(seed)
        self._state: Optional[State] = None
        self._task: Optional[TaskSpec] = None
        self._curriculum_phase = curriculum_phase

        # Plugins
        self._rules = RuleEngine() if RuleEngine is not None else None
        self._reward = (
            MetaAgentRewardComputer(self.reward_config)
            if MetaAgentRewardComputer is not None
            else None
        )
        self._tasks = TaskGenerator(seed=seed) if TaskGenerator is not None else None

        self._last_breakdown: dict[str, float] = {}

        # META-AGENT: Judge state (90% frequency)
        self._judge_step_count = 0
        self._calibration_data: list[dict] = []

    # ------------------------------------------------------------------ OpenEnv interface

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,  # noqa: ARG002
        **kwargs,
    ) -> Observation:
        """Start a new episode. Extra kwargs accepted: `scenario_name`, `curriculum_phase`."""
        if seed is not None:
            self._rng = random.Random(seed)

        scenario_name = kwargs.get("scenario_name")
        curriculum_phase = kwargs.get("curriculum_phase", self._curriculum_phase)
        self._curriculum_phase = curriculum_phase

        task = self._pick_task(scenario_name)
        self._task = task

        # META-AGENT: Initialize state with empty spec
        self._state = State(
            task_id=task.task_id,
            step=0,
            max_steps=task.max_steps,
            current_spec={},  # Start with empty spec
            hidden_truth=self._generate_hidden_truth(task),
        )

        # Reset judge state
        self._judge_step_count = 0

        logger.info(
            "reset task=%s difficulty=%s max_steps=%d phase=%d",
            task.task_id, task.difficulty, task.max_steps, self._curriculum_phase,
            extra={"hackathon_tag": "START"},
        )

        return self._build_observation(reward=0.0, violations={})

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> Observation:
        """Advance one step."""
        if self._state is None or self._task is None:
            self.reset()

        # 1. Rule engine checks
        violations = self._check_rules(action)
        hard = [v for v in violations if v.severity == "hard"]

        # 2. Apply action (unless blocked)
        if not hard:
            self._execute_action(action)
        else:
            logger.warning(
                "hard violation blocked action: %s",
                hard[0].message,
                extra={"hackathon_tag": "STEP"},
            )

        self._state.step += 1

        # 3. Compute reward
        reward = self._compute_reward(action, violations)

        # 4. Handle truncation wipe
        if (
            self._state.step >= self._state.max_steps
            and self.reward_config.truncation_reward_total is not None
            and not self._check_termination()
        ):
            target = self.reward_config.truncation_reward_total
            wipe_adjustment = target - (self._state.cumulative_reward + reward)
            reward += wipe_adjustment
            self._last_breakdown["truncation_wipe"] = wipe_adjustment

        self._state.cumulative_reward += reward
        self._state.previous_score = self._last_breakdown.get("total", 0.0)

        # 5. Record step
        self._state.step_history.append({
            "step": self._state.step,
            "action": action.model_dump(),
            "reward": reward,
            "violations": [v.model_dump() for v in violations],
        })

        logger.info(
            "step=%d action=%s reward=%.3f violations=%d",
            self._state.step, action.command.value, reward, len(violations),
            extra={"hackathon_tag": "STEP"},
        )

        obs = self._build_observation(reward=reward, violations=violations)

        if obs.done or obs.truncated:
            logger.info(
                "episode end cumulative=%.3f steps=%d done=%s truncated=%s",
                self._state.cumulative_reward, self._state.step, obs.done, obs.truncated,
                extra={"hackathon_tag": "END"},
            )

        return obs

    @property
    def state(self) -> State:
        """Full state (hidden + visible)."""
        if self._state is None:
            self.reset()
        assert self._state is not None
        return self._state

    # ------------------------------------------------------------------ Fill-in hooks

    def _execute_action(self, action: Action) -> None:
        """Apply meta-agent command to current spec."""
        assert self._state is not None

        cmd = action.command
        args = action.args
        spec = self._state.current_spec

        # Investigation commands (don't modify spec)
        if cmd == ActionCommand.CHECK_SCORE:
            # Will be handled in _build_observation
            return
        elif cmd == ActionCommand.INSPECT_EXAMPLE:
            # Will be handled in _build_observation
            return

        # Spec building commands
        if cmd == ActionCommand.SET_NAME:
            spec["name"] = args.get("name", "")
        elif cmd == ActionCommand.SET_DESCRIPTION:
            spec["description"] = args.get("description", "")
        elif cmd == ActionCommand.ADD_SKILL:
            skill = args.get("skill")
            if skill:
                spec.setdefault("skills", []).append(skill)
                # Dedupe
                spec["skills"] = list(set(spec["skills"]))
        elif cmd == ActionCommand.REMOVE_SKILL:
            skill = args.get("skill")
            if skill and "skills" in spec:
                spec["skills"] = [s for s in spec["skills"] if s != skill]
        elif cmd == ActionCommand.SET_MODEL:
            spec["model"] = args.get("model", "sonnet")
        elif cmd == ActionCommand.ADD_TOOLS:
            tool = args.get("tool")
            if tool:
                spec.setdefault("allowed_tools", []).append(tool)
        elif cmd == ActionCommand.WRITE_PROMPT:
            # Can append or replace
            prompt = args.get("prompt", "")
            mode = args.get("mode", "replace")  # "replace" or "append"
            if mode == "append":
                spec["system_prompt"] = spec.get("system_prompt", "") + "\n" + prompt
            else:
                spec["system_prompt"] = prompt
        elif cmd == ActionCommand.SET_MEMORY:
            spec["memory"] = args.get("memory")
        elif cmd == ActionCommand.SET_MAX_TURNS:
            spec["max_turns"] = args.get("max_turns")

    def _build_observation(
        self,
        reward: float,
        violations: dict | list[RuleViolation],
    ) -> Observation:
        """Project state → agent-visible view (POMDP)."""
        assert self._state is not None and self._task is not None

        done = self._check_termination()
        truncated = (
            self._state.step >= self._state.max_steps and not done
        )

        # META-AGENT: Build investigation result if relevant
        investigation_result = None
        latest_output: dict | None = None
        summary_parts = [f"Step {self._state.step}/{self._state.max_steps}"]

        last_action = (
            self._state.step_history[-1]["action"]
            if self._state.step_history
            else None
        )
        if last_action:
            cmd = last_action.get("command")
            if cmd == ActionCommand.CHECK_SCORE.value:
                investigation_result = {
                    "current_score": self._last_breakdown.get("total", 0.0),
                    "breakdown": dict(self._last_breakdown),
                }
                latest_output = {"score_check": self._last_breakdown.get("total", 0.0)}
                summary_parts.append("checked")
            elif cmd == ActionCommand.INSPECT_EXAMPLE.value:
                investigation_result = self._get_example_agent(self._task.domain)
                latest_output = {"example_inspected": self._task.domain}
                summary_parts.append("inspected")
            elif cmd == ActionCommand.INSPECT.value:
                # INSPECT command provides task info
                latest_output = {
                    "domain": self._task.domain,
                    "difficulty": self._task.difficulty,
                    "required_skills": self._task.required_skills,
                }
                summary_parts.append(f"inspected {self._task.domain}")
            elif cmd == ActionCommand.NOOP.value:
                latest_output = {"noop": True}
                summary_parts.append("noop")
            else:
                # Other commands modify spec
                spec_keys = list(self._state.current_spec.keys())
                latest_output = {"spec_keys": spec_keys}
                summary_parts.append(f"spec({len(spec_keys)} fields)")

        summary = ", ".join(summary_parts)

        # META-AGENT: Build feedback list
        feedback = []
        if violations:
            if isinstance(violations, list):
                for v in violations:
                    feedback.append(f"{v.category}: {v.message}")
            else:
                feedback = [str(violations)]

        # Add reward-based feedback
        if self._last_breakdown.get("total", 0) < 0:
            feedback.append("Current score is negative. Consider revising your approach.")

        return Observation(
            task_id=self._state.task_id,
            step=self._state.step,
            max_steps=self._state.max_steps,
            # META-AGENT fields
            current_spec=dict(self._state.current_spec),
            investigation_result=investigation_result,
            available_skills=list(AVAILABLE_SKILLS.keys()),
            score=self._last_breakdown.get("total", 0.0),
            feedback=feedback,
            # Standard fields
            summary=summary,
            latest_output=latest_output,
            history=self._state.step_history[-5:],
            reward=reward,
            reward_breakdown=dict(self._last_breakdown),
            rule_violations=violations if isinstance(violations, list) else [],
            budget_remaining=self._task.budget,
            time_remaining=self._task.time_limit,
            done=done,
            truncated=truncated,
        )

    def _check_termination(self) -> bool:
        """Is the task complete? META-AGENT: Submit command triggers completion."""
        if not self._state or not self._state.step_history:
            return False

        last_action = self._state.step_history[-1]["action"]
        if last_action.get("command") == ActionCommand.SUBMIT.value:
            # Validate spec is complete before accepting submit
            if HardVerifiers is not None:
                passed, _ = HardVerifiers.get_gate_results(self._state.current_spec)
                return passed
            return True

        return False

    # ------------------------------------------------------------------ Internal helpers

    def _pick_task(self, scenario_name: Optional[str]) -> TaskSpec:
        """Pick a task based on curriculum phase."""
        if SCENARIOS is not None and get_scenario is not None:
            # Try specific scenario first
            if scenario_name:
                scenario = get_scenario(scenario_name)
                if scenario:
                    return scenario

            # Filter by curriculum phase
            from server.tasks.scenarios import get_scenarios_by_phase
            phase_tasks = get_scenarios_by_phase(self._curriculum_phase)
            if phase_tasks:
                return self._rng.choice(phase_tasks)

            # Fallback to all scenarios
            return self._rng.choice(SCENARIOS)

        return TaskSpec(
            task_id=scenario_name or "placeholder",
            difficulty="easy",
            problem_statement="Placeholder task",
            max_steps=5,
        )

    def _generate_hidden_truth(self, task: TaskSpec) -> dict:
        """Generate hidden ground truth for the task."""
        # This is the "optimal" spec that the policy should discover
        # In practice, this could come from expert demonstrations
        return {
            "optimal_skills": set(task.required_skills + task.recommended_skills),
            "optimal_model": "sonnet" if task.difficulty != "expert" else "opus",
            "min_prompt_length": 100,
        }

    def _get_example_agent(self, domain: str) -> dict:
        """Get an example good agent for this domain."""
        from server.skills import get_template_for_domain

        template = get_template_for_domain(domain)
        return {
            "domain": domain,
            "example_template": template or "No template available",
            "hint": f"Focus on {domain}-specific skills and best practices",
        }

    def _check_rules(self, action: Action) -> list[RuleViolation]:
        if self._rules is None:
            return []
        assert self._state is not None and self._task is not None
        return self._rules.check(action, self._state, self._task)

    def _compute_reward(
        self,
        action: Action,
        violations: list[RuleViolation],
    ) -> float:
        if self._reward is None:
            self._last_breakdown = {}
            return 0.0
        assert self._state is not None and self._task is not None
        reward = self._reward.compute(action, self._state, self._task, violations)
        self._last_breakdown = self._reward.last_breakdown
        return reward
