"""Core OpenEnv environment — reset / step / state lifecycle.

TEMPLATE: Override `_execute_action` and `_build_observation` on finale day
for domain-specific behavior. The framework itself (rule checks, reward
computation, structured logging, trajectory recording) stays untouched.

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
    Observation,
    RewardConfig,
    RuleViolation,
    State,
    TaskSpec,
)

# OpenEnv ABC — our Environment subclasses this so create_app accepts it.
# Fall back to `object` if openenv-core isn't installed (local dev only).
try:
    from openenv.core import Environment as _OpenEnvBase
except ImportError:
    _OpenEnvBase = object  # type: ignore[assignment,misc]

# Sibling-module imports are defensive: during scaffolding, engine/reward/tasks
# may not exist yet. Environment falls back to safe no-op behavior.
try:
    from server.rules.engine import RuleEngine
except ImportError:
    RuleEngine = None  # type: ignore[assignment,misc]

try:
    from server.rewards.reward import RewardComputer
except ImportError:
    RewardComputer = None  # type: ignore[assignment,misc]

try:
    from server.tasks.generator import TaskGenerator
except ImportError:
    TaskGenerator = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Logging setup — load training/logging.yaml once at import time
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
    """Domain-agnostic OpenEnv environment skeleton.

    Inherits from openenv.core.Environment. Required abstract methods:
        reset(seed, episode_id, **kwargs) -> Observation
        step(action, timeout_s, **kwargs) -> Observation
        state -> State   # NOTE: property, not method

    Fill-in points (finale day):
        _execute_action      — mutate hidden state based on action
        _build_observation   — project state → what agent sees (POMDP-aware)
        _check_termination   — custom end conditions (default: step >= max_steps)
    """

    # All session state lives on `self` (self._state, self._task, self._rng).
    # OpenEnv creates a fresh Environment instance per session, so state is
    # naturally isolated. Required for max_concurrent_envs > 1.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        domain_randomise: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        # Call super if we inherit from OpenEnv (not object fallback)
        if _OpenEnvBase is not object:
            try:
                super().__init__()
            except TypeError:
                pass  # OpenEnv base may not have a compatible __init__

        self.reward_config = reward_config or RewardConfig()
        self.domain_randomise = domain_randomise
        self.seed = seed
        self._rng = random.Random(seed)
        self._state: Optional[State] = None
        self._task: Optional[TaskSpec] = None

        # Plugins — lazy-init so absent modules don't break scaffolding
        self._rules = RuleEngine() if RuleEngine is not None else None
        self._reward = (
            RewardComputer(self.reward_config) if RewardComputer is not None else None
        )
        self._tasks = TaskGenerator(seed=seed) if TaskGenerator is not None else None

        # Cached breakdown from the last reward computation (populates Observation)
        self._last_breakdown: dict[str, float] = {}

    # ------------------------------------------------------------------ OpenEnv interface

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,  # noqa: ARG002
        **kwargs,
    ) -> Observation:
        """Start a new episode. Extra kwargs accepted: `scenario_name`."""
        if seed is not None:
            self._rng = random.Random(seed)
        scenario_name = kwargs.get("scenario_name")
        task = self._pick_task(scenario_name)
        self._task = task
        self._state = State(
            task_id=task.task_id,
            step=0,
            max_steps=task.max_steps,
        )
        logger.info(
            "reset task=%s difficulty=%s max_steps=%d",
            task.task_id, task.difficulty, task.max_steps,
            extra={"hackathon_tag": "START"},
        )
        return self._build_observation(reward=0.0, violations=[])

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> Observation:
        """Advance one step.

        OpenEnv HTTP is stateless (fresh Environment per request). WebSocket
        sessions preserve state. Auto-reset here so HTTP /step still works
        standalone for debugging/smoke tests — WebSocket clients control
        their own reset/step cadence.
        """
        if self._state is None or self._task is None:
            self.reset()

        # 1. Rule engine checks — hard violations block, soft violations pass through
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

        # 3b. Truncation wipe — if this step hits max_steps and config sets
        # truncation_reward_total, force the cumulative episode reward to
        # exactly that value. Guarantees GRPO advantage variance between
        # success episodes (naturally high) and truncated episodes
        # (deterministically wiped).
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

        # 4. Record step in history (trajectory replay support)
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
        """Full state (hidden + visible). Debug/eval only — not for agent.

        NOTE: This is a @property (not method) per openenv.core.Environment ABC.
        Auto-resets on first access to support stateless HTTP GET /state.
        """
        if self._state is None:
            self.reset()
        assert self._state is not None
        return self._state

    # ------------------------------------------------------------------ Fill-in hooks

    def _execute_action(self, action: Action) -> None:  # noqa: ARG002
        """Domain-specific state transition.

        DOMAIN: override to mutate ``self._state.hidden_truth`` and
        ``self._state.progress_flags`` based on the action. The base class
        just tracks step count.
        """
        return

    def _build_observation(
        self,
        reward: float,
        violations: list[RuleViolation],
    ) -> Observation:
        """Project state → agent-visible view.

        DOMAIN: extend with domain-specific ``summary``, ``latest_output``.
        Keep hidden_truth out of the observation for POMDP integrity.
        """
        assert self._state is not None and self._task is not None

        done = self._check_termination()
        truncated = (
            self._state.step >= self._state.max_steps and not done
        )
        return Observation(
            task_id=self._state.task_id,
            step=self._state.step,
            max_steps=self._state.max_steps,
            summary=f"Step {self._state.step}/{self._state.max_steps}",
            history=self._state.step_history[-5:],
            reward=reward,
            reward_breakdown=dict(self._last_breakdown),
            rule_violations=violations,
            budget_remaining=self._task.budget,
            time_remaining=self._task.time_limit,
            done=done,
            truncated=truncated,
        )

    def _check_termination(self) -> bool:
        """Is the task complete (natural success)? Override per domain.

        Default: False. The base class produces `truncated=True` when
        step >= max_steps (via _build_observation) but never `done=True`
        on its own — completion is domain-specific (e.g., correct guess,
        successful fix, submitted answer matching ground truth).

        Domains override this to return True when task success conditions
        are met. This distinction matters for:
          - truncation_reward_total wipe (only applies when truncated, not done)
          - GRPO advantage computation (success vs timeout rewards differ)
        """
        return False

    # ------------------------------------------------------------------ Internal helpers

    def _pick_task(self, scenario_name: Optional[str]) -> TaskSpec:
        if self._tasks is not None:
            return self._tasks.generate(
                scenario_name=scenario_name,
                domain_randomise=self.domain_randomise,
            )
        return TaskSpec(
            task_id=scenario_name or "placeholder",
            difficulty="easy",
            problem_statement="Placeholder task — TaskGenerator not yet wired.",
            max_steps=5,
        )

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
