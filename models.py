"""Pydantic schemas for Action, Observation, State, and Task.

TEMPLATE: Slots marked `# DOMAIN:` are filled in per theme on finale day.
The framework itself (hard/soft violations, reward breakdown, sub-agent hook)
is domain-agnostic.

Action / Observation / State inherit from openenv.core base classes so that
openenv's create_app recognizes them. If openenv-core isn't installed (e.g.
during local development without deps), fall back to plain Pydantic BaseModel.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core import Action as _OpenEnvAction
    from openenv.core import Observation as _OpenEnvObservation
    from openenv.core import State as _OpenEnvState
except ImportError:
    _OpenEnvAction = BaseModel  # type: ignore[assignment,misc]
    _OpenEnvObservation = BaseModel  # type: ignore[assignment,misc]
    _OpenEnvState = BaseModel  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Action model — command-based pattern
# ---------------------------------------------------------------------------

class ActionCommand(str, Enum):
    """Base command set. Extend per domain.

    DOMAIN: Add domain-specific commands here.
    Example (code auditor): EDIT_FILE, ADD_FILE, DELETE_FILE, INSPECT, SUBMIT
    Example (SRE): KUBECTL, DESCRIBE, FIX, RESOLVE
    """

    INSPECT = "inspect"
    SUBMIT = "submit"
    NOOP = "noop"


class Action(_OpenEnvAction):
    """Agent's action at one step.

    Command-based action model beats free-form text: token-efficient,
    grader-friendly, easier to validate.

    Inherits `metadata: dict` from openenv.core.Action.
    """

    command: ActionCommand
    args: dict[str, Any] = Field(default_factory=dict)

    # Optional delegation hook for hierarchical / multi-agent designs.
    # Leave None for single-agent; set when sub-agents exist.
    invoked_subagent: Optional[str] = None

    # Optional rationale the agent provides — useful for post-hoc analysis
    # and for rewarding calibrated confidence.
    justification: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Observation model — what the agent sees
# ---------------------------------------------------------------------------

class RuleViolation(BaseModel):
    """Single rule-engine violation.

    The `penalty` field is optional per-violation weight. When present, the
    reward computer uses it verbatim (per-violation model); when absent, it
    falls back to the flat `RewardConfig.soft_violation_penalty × count`.

    Per-violation weights are the advanced pattern:
      - Transparent: judges see why each violation costs what it costs
      - Debuggable: easy to attribute reward changes to specific violations
      - Fine-grained: a "causal error" can cost more than "minor redundancy"
    """

    severity: str  # "hard" | "soft"
    category: str  # e.g. "prerequisite", "resource", "redundancy", "causal"
    message: str
    penalty: Optional[float] = None  # DOMAIN: set per-violation weight (else flat fallback)


class Observation(_OpenEnvObservation):
    """What the agent sees after each step.

    POMDP-friendly by default: exposes summaries, not ground truth.
    Hidden state lives in `State` below.

    Inherits from openenv.core.Observation:
        done: bool    (inherited, default False)
        reward: float (inherited, default None — we set 0.0)
        metadata: dict (inherited)
    """

    # Task context
    task_id: str
    step: int
    max_steps: int

    # Override inherited reward to default 0.0 (not None)
    reward: float = 0.0

    # Visible state summary — DOMAIN: fill these
    summary: str = ""
    latest_output: Optional[dict[str, Any]] = None
    history: list[dict[str, Any]] = Field(default_factory=list)

    # Decomposed reward breakdown + rule violations
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    rule_violations: list[RuleViolation] = Field(default_factory=list)

    # Resources (budget/time). None if domain has no budget.
    budget_remaining: Optional[float] = None
    time_remaining: Optional[float] = None

    # Truncation flag (complements inherited `done`)
    truncated: bool = False


# ---------------------------------------------------------------------------
# State model — hidden + visible ground truth
# ---------------------------------------------------------------------------

class State(_OpenEnvState):
    """Full environment state.

    Split into visible/hidden sections for POMDP support.
    The agent receives `Observation`, NOT `State`.

    Inherits from openenv.core.State:
        episode_id: Optional[str] = None
        step_count: int = 0
    """

    # Visible — mirrors Observation. `step` is our name; `step_count` is inherited.
    task_id: str
    step: int = 0
    max_steps: int

    # Hidden ground truth — DOMAIN: fill per theme
    # Examples: true DE genes (Bio), correct code fix (auditor), true faults (SRE)
    hidden_truth: dict[str, Any] = Field(default_factory=dict)

    # Progress flags — incremented as pipeline milestones hit
    progress_flags: dict[str, bool] = Field(default_factory=dict)

    # Cumulative reward so far
    cumulative_reward: float = 0.0

    # Full step history (for trajectory serialization)
    step_history: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Task model — a single scenario
# ---------------------------------------------------------------------------

class TaskSpec(BaseModel):
    """One scenario presented to the agent.

    Difficulty-scaled. Real-world-grounded with citations.
    """

    task_id: str
    difficulty: str  # "easy" | "medium" | "hard"
    problem_statement: str
    max_steps: int

    # Real-world grounding — DOMAIN: cite a paper/RFC/style-guide/benchmark
    citations: list[str] = Field(default_factory=list)

    # Budget/time caps — None if domain has no budget
    budget: Optional[float] = None
    time_limit: Optional[float] = None

    # Expected findings / ground truth summary for grading
    expected_findings: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward config — decomposed reward modes
# ---------------------------------------------------------------------------

class RewardMode(str, Enum):
    ADDITIVE = "additive"            # weighted sum of components
    MULTIPLICATIVE = "multiplicative" # product of components × K; prevents fake wins
    HYBRID = "hybrid"                # gate + additive score


class RewardConfig(BaseModel):
    """Reward function configuration.

    Three modes, pick by domain:
      ADDITIVE       — smooth gradient, partial credit everywhere. Safest for
                       GRPO training. Weakness: gameable if any one dim can
                       compensate for a broken other dim.
      MULTIPLICATIVE — all dims must succeed simultaneously. Prevents fake
                       wins. Weakness: near-zero on any low component →
                       gradient collapses. Use only for binary-success domains.
      HYBRID         — gate critical dims multiplicatively, score the rest
                       additively. Best of both: strict constraints + smooth
                       signal. Recommended for most domains.

    HYBRID uses `gate_components`: any listed component must exceed
    `gate_threshold` or the entire reward collapses to 0. Other components
    are combined additively with their weights.
    """

    mode: RewardMode = RewardMode.MULTIPLICATIVE
    component_weights: dict[str, float] = Field(default_factory=dict)

    # HYBRID-only: which components are hard gates (e.g., ["safety"])
    gate_components: list[str] = Field(default_factory=lambda: ["safety"])
    gate_threshold: float = 0.01  # below this → gate fails → reward = 0

    regression_penalty: float = 0.2   # explicit "don't break working stuff"
    soft_violation_penalty: float = 0.15
    novelty_bonus: float = 0.1
    shaping_gamma: float = 0.99

    # Force truncated episodes to a fixed total reward.
    # When None (default) truncated episodes sum naturally. When set (e.g. -2.0)
    # the last step's reward is adjusted so that cumulative_reward == this value.
    # Guarantees GRPO advantage variance: success → high positive,
    # truncation → deterministic negative → clear gradient signal between them.
    truncation_reward_total: Optional[float] = None
