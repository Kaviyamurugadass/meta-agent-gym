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
# Meta-Agent specific models
# ---------------------------------------------------------------------------


class ModelType(str, Enum):
    """Allowed model types for AGENT.md files."""
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"
    INHERIT = "inherit"


class AgentSpec(BaseModel):
    """Complete agent specification following Agent Skills Open Standard.

    This is the OUTPUT of our meta-agent training — a fully formed AGENT.md
    that can be used in Claude Code, Goose, Copilot, etc.
    """

    # Required fields
    name: str = Field(..., description="Agent name (lowercase, hyphens)")
    description: str = Field(..., description="What agent does + when to use it")

    # Skills (capabilities) — list of skill identifiers
    skills: list[str] = Field(default_factory=list, description="Skills from registry")

    # Model selection
    model: ModelType = Field(default=ModelType.SONNET, description="Which model to use")

    # System prompt — the core instructions
    system_prompt: str = Field(..., description="Agent instructions and workflow")

    # Optional metadata
    user_invocable: bool = Field(default=True, description="Can users invoke directly?")
    allowed_tools: Optional[list[str]] = Field(default=None, description="Tool restrictions")
    memory: Optional[str] = Field(default=None, description="Memory scope: user/project/local")
    max_turns: Optional[int] = Field(default=None, description="Max agent turns")

    def to_markdown(self) -> str:
        """Convert to AGENT.md format (frontmatter + prompt)."""
        frontmatter = {
            "name": self.name,
            "description": self.description,
            "user-invocable": self.user_invocable,
        }
        if self.allowed_tools:
            frontmatter["allowed-tools"] = ", ".join(self.allowed_tools)
        if self.skills:
            frontmatter["skills"] = self.skills
        if self.model != ModelType.INHERIT:
            frontmatter["model"] = self.model.value
        if self.memory:
            frontmatter["memory"] = self.memory
        if self.max_turns:
            frontmatter["max-turns"] = self.max_turns

        # Build YAML frontmatter
        yaml_lines = ["---"]
        for key, value in frontmatter.items():
            if isinstance(value, bool):
                yaml_lines.append(f"{key}: {str(value).lower()}")
            elif isinstance(value, list):
                yaml_lines.append(f"{key}:")
                for item in value:
                    yaml_lines.append(f"  - {item}")
            else:
                yaml_lines.append(f"{key}: {value}")
        yaml_lines.append("---")

        return "\n".join(yaml_lines) + "\n\n" + self.system_prompt

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict for verification."""
        return {
            "name": self.name,
            "description": self.description,
            "skills": self.skills,
            "model": self.model.value,
            "system_prompt": self.system_prompt,
            "user_invocable": self.user_invocable,
            "allowed_tools": self.allowed_tools,
            "memory": self.memory,
            "max_turns": self.max_turns,
        }


# ---------------------------------------------------------------------------
# Action model — command-based pattern
# ---------------------------------------------------------------------------

class ActionCommand(str, Enum):
    """Command set for meta-agent environment.

    Uses command-based actions for token efficiency and better GRPO performance.
    The agent builds an AGENT.md incrementally through discrete commands.
    """

    # Core agent building commands
    SET_NAME = "set_name"
    SET_DESCRIPTION = "set_description"
    ADD_SKILL = "add_skill"
    REMOVE_SKILL = "remove_skill"
    SET_MODEL = "set_model"
    ADD_TOOLS = "add_tools"
    WRITE_PROMPT = "write_prompt"
    SET_MEMORY = "set_memory"
    SET_MAX_TURNS = "set_max_turns"

    # Investigation commands (peek at state without submitting)
    CHECK_SCORE = "check_score"
    INSPECT_EXAMPLE = "inspect_example"

    # Terminal command
    SUBMIT = "submit"

    # Fallback
    NOOP = "noop"
    INSPECT = "inspect"  # Alias for CHECK_SCORE


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

    META-AGENT EXTENSIONS:
        - current_spec: Partial view of agent being built
        - investigation_result: Result from CHECK_SCORE/INSPECT_EXAMPLE
        - available_skills: Skills the agent can choose from
    """

    # Task context
    task_id: str
    step: int
    max_steps: int

    # Override inherited reward to default 0.0 (not None)
    reward: float = 0.0

    # META-AGENT: Current agent spec state (partial observability)
    current_spec: dict[str, Any] = Field(
        default_factory=dict,
        description="Partial view of current agent state being built"
    )

    # META-AGENT: Investigation command results
    investigation_result: Optional[dict[str, Any]] = Field(
        default=None,
        description="Result from CHECK_SCORE or INSPECT_EXAMPLE commands"
    )

    # META-AGENT: Available skills for this task
    available_skills: list[str] = Field(
        default_factory=list,
        description="Skills the agent can choose from"
    )

    # META-AGENT: Current score and feedback
    score: float = Field(default=0.0, description="Current total score")
    feedback: list[str] = Field(
        default_factory=list,
        description="Feedback from judge/verifiers"
    )

    # Visible state summary
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

    META-AGENT EXTENSIONS:
        - current_spec: The agent spec being built
        - passing_checks: Track which checks have passed (for regression detection)
        - previous_score: Track score for delta calculation
    """

    # Visible — mirrors Observation. `step` is our name; `step_count` is inherited.
    task_id: str
    step: int = 0
    max_steps: int

    # Hidden ground truth — META-AGENT: the "optimal" spec (hidden from policy)
    hidden_truth: dict[str, Any] = Field(default_factory=dict)

    # META-AGENT: Current agent spec being built
    current_spec: dict[str, Any] = Field(
        default_factory=dict,
        description="Current state of the agent being built"
    )

    # META-AGENT: Track passing checks for regression detection
    passing_checks: set[str] = Field(
        default_factory=set,
        description="Checks that have passed (for regression penalty)"
    )

    # Progress flags — incremented as pipeline milestones hit
    progress_flags: dict[str, bool] = Field(default_factory=dict)

    # Cumulative reward so far
    cumulative_reward: float = 0.0

    # META-AGENT: Previous score for delta calculation
    previous_score: float = 0.0

    # Full step history (for trajectory serialization)
    step_history: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Task model — a single scenario
# ---------------------------------------------------------------------------

class TaskSpec(BaseModel):
    """One scenario presented to the agent.

    Difficulty-scaled. Real-world-grounded with citations.

    META-AGENT EXTENSIONS:
        - required_skills: Skills the agent MUST include
        - domain: Task domain (web, data, code, etc.)
        - user_preferences: User constraints/preferences
    """

    task_id: str
    difficulty: str  # "easy" | "medium" | "hard" | "expert"
    problem_statement: str
    max_steps: int

    # META-AGENT: Domain and skill requirements
    domain: str = Field(default="general", description="Task domain")
    required_skills: list[str] = Field(
        default_factory=list,
        description="Skills the generated agent MUST include"
    )
    recommended_skills: list[str] = Field(
        default_factory=list,
        description="Optional but relevant skills"
    )
    user_preferences: dict[str, Any] = Field(
        default_factory=dict,
        description="User constraints (language, cost limits, etc.)"
    )

    # Real-world grounding — cite a paper/RFC/style-guide/benchmark
    citations: list[str] = Field(default_factory=list)

    # Budget/time caps — None if domain has no budget
    budget: Optional[float] = None
    time_limit: Optional[float] = None

    # Expected findings / ground truth summary for grading
    expected_findings: dict[str, Any] = Field(default_factory=dict)

    # META-AGENT: Red herrings (patterns that look wrong but are correct)
    red_herrings: list[str] = Field(
        default_factory=list,
        description="Patterns that look wrong but shouldn't be 'fixed'"
    )


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

    META-AGENT: Uses HYBRID mode with hard verifiers as gates.
    """

    mode: RewardMode = RewardMode.HYBRID

    # META-AGENT: Component weights for judge-based rewards
    component_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "skill_selection": 0.25,
            "description_quality": 0.20,
            "workflow_clarity": 0.20,
            "model_appropriateness": 0.15,
            "best_practices": 0.10,
            "efficiency": 0.10,
        }
    )

    # HYBRID-only: which components are hard gates
    # META-AGENT: Hard verifiers are gates — they must pass
    gate_components: list[str] = Field(
        default_factory=lambda: ["yaml_valid", "has_required_fields", "prompt_length_ok"]  # META-AGENT: Gate on format + completeness
    )
    gate_threshold: float = 0.99  # below this → gate fails → reward = 0

    # Standard penalties
    regression_penalty: float = 0.15
    soft_violation_penalty: float = 0.05
    novelty_bonus: float = 0.1
    shaping_gamma: float = 0.99

    # Force truncated episodes to a fixed total reward
    truncation_reward_total: Optional[float] = None

    # META-AGENT: Anti-hacking penalties (v4 RLVR approach)
    anti_hack_empty_spec: float = -5.0
    anti_hack_over_engineered: float = -0.5
    anti_hack_repetitive: float = -0.3

    # Thresholds for anti-hacking
    max_skills_limit: int = 10  # Penalty if > this many skills
    min_prompt_length: int = 50  # Penalty if < this many chars
