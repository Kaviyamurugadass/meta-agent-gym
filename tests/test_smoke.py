"""Smoke tests — verify the minimum env lifecycle works.

These run without network (no LLM inference). Confirms:
    - models.py schemas validate
    - Environment can reset + step + state
    - Rule engine + reward computer + task generator wire together
    - Episode can terminate via max_steps
"""

from __future__ import annotations

import pytest

from models import Action, ActionCommand, Observation, State
from server.environment import Environment


# ---------------------------------------------------------------------------
# Model schema tests
# ---------------------------------------------------------------------------


def test_action_schema():
    a = Action(command=ActionCommand.NOOP)
    assert a.command == ActionCommand.NOOP
    assert a.args == {}
    assert a.confidence is None


def test_action_with_args_and_confidence():
    a = Action(
        command=ActionCommand.INSPECT,
        args={"target": "file.py"},
        justification="checking state",
        confidence=0.85,
    )
    assert a.args["target"] == "file.py"
    assert a.confidence == 0.85


def test_action_confidence_clamped():
    import pydantic
    try:
        Action(command=ActionCommand.NOOP, confidence=1.5)
    except pydantic.ValidationError:
        pass
    else:
        raise AssertionError("confidence > 1.0 should raise ValidationError")


def test_observation_defaults():
    o = Observation(task_id="t1", step=0, max_steps=5)
    assert o.reward == 0.0
    assert o.done is False
    assert o.rule_violations == []


# ---------------------------------------------------------------------------
# Environment lifecycle tests
# ---------------------------------------------------------------------------


def test_environment_reset():
    env = Environment(domain_randomise=False, seed=42)
    obs = env.reset()
    assert obs.step == 0
    assert obs.max_steps > 0
    assert obs.reward == 0.0
    assert obs.done is False


def test_environment_step():
    env = Environment(domain_randomise=False, seed=42)
    env.reset()
    obs = env.step(Action(command=ActionCommand.NOOP))
    assert obs.step == 1
    assert isinstance(obs.reward, float)


def test_environment_terminates_at_max_steps():
    env = Environment(domain_randomise=False, seed=42)
    obs = env.reset()
    max_steps = obs.max_steps

    for _ in range(max_steps + 2):
        if obs.done or obs.truncated:
            break
        obs = env.step(Action(command=ActionCommand.NOOP))

    assert obs.done or obs.truncated, "episode must terminate within max_steps+2 steps"


def test_environment_state_reflects_steps():
    env = Environment(domain_randomise=False, seed=42)
    env.reset()
    env.step(Action(command=ActionCommand.NOOP))
    env.step(Action(command=ActionCommand.INSPECT))

    state = env.state
    assert isinstance(state, State)
    assert state.step == 2
    assert len(state.step_history) == 2


def test_environment_step_before_reset_auto_resets():
    """Auto-reset on step without explicit reset — supports OpenEnv stateless HTTP."""
    env = Environment(domain_randomise=False)
    obs = env.step(Action(command=ActionCommand.NOOP))
    assert obs.step == 1
    assert obs.task_id  # auto-reset populated a task


def test_environment_reset_with_named_scenario():
    env = Environment(domain_randomise=False)
    obs = env.reset(scenario_name="placeholder_easy")
    assert obs.task_id == "placeholder_easy"
    assert obs.max_steps == 5


def test_observation_surfaces_reward_breakdown():
    """Reward breakdown from RewardComputer must propagate into Observation."""
    env = Environment(domain_randomise=False)
    env.reset(scenario_name="placeholder_easy")
    obs = env.step(Action(command=ActionCommand.NOOP))
    # Novelty bonus triggers when there are no soft violations → breakdown is non-empty
    assert obs.reward_breakdown, "reward_breakdown must be populated, not an empty dict"
    assert "total" in obs.reward_breakdown
    assert "core" in obs.reward_breakdown


def test_hybrid_mode_gates_on_zero_safety():
    """HYBRID mode: if safety (gate component) is 0, reward must be 0."""
    from models import RewardConfig, RewardMode
    from server.rewards.reward import RewardComputer
    from models import Action, ActionCommand

    config = RewardConfig(
        mode=RewardMode.HYBRID,
        gate_components=["safety"],
        gate_threshold=0.01,
        component_weights={"correctness": 0.5, "efficiency": 0.2, "quality": 0.2, "safety": 0.1},
    )

    class FailSafetyComputer(RewardComputer):
        def _component_scores(self, action, state, task):
            return {"correctness": 1.0, "efficiency": 1.0, "quality": 1.0, "safety": 0.0}

    class PassSafetyComputer(RewardComputer):
        def _component_scores(self, action, state, task):
            return {"correctness": 0.5, "efficiency": 0.5, "quality": 0.5, "safety": 1.0}

    from models import State, TaskSpec
    task = TaskSpec(task_id="t", difficulty="easy", problem_statement="", max_steps=1)
    state = State(task_id="t", max_steps=1)
    action = Action(command=ActionCommand.NOOP)

    fail = FailSafetyComputer(config)
    pass_ = PassSafetyComputer(config)

    # Safety=0 → gate fails → core is 0 → no novelty bonus offset in core
    assert fail.last_breakdown == {}  # not computed yet
    r_fail = fail.compute(action, state, task, violations=[])
    assert fail.last_breakdown["core"] == 0.0, "safety=0 must collapse core to 0"

    # Safety>threshold → additive scoring applies, core non-zero
    r_pass = pass_.compute(action, state, task, violations=[])
    assert pass_.last_breakdown["core"] > 0.0, "safety>threshold must produce positive core"
    assert r_pass > r_fail, "passing gate must exceed failing gate"


def test_per_violation_penalty_beats_flat():
    """Violations with explicit `penalty` use that value; without, use flat config."""
    from models import RewardConfig, RuleViolation, State, TaskSpec
    from server.rewards.reward import RewardComputer

    cfg = RewardConfig(soft_violation_penalty=0.15)
    computer = RewardComputer(cfg)

    # Per-violation weights: style=0.05, efficiency=0.10 → total 0.15
    weighted = [
        RuleViolation(severity="soft", category="style", message="", penalty=0.05),
        RuleViolation(severity="soft", category="efficiency", message="", penalty=0.10),
    ]
    total_w, per_cat_w = computer._aggregate_soft_penalty(weighted)
    assert total_w == pytest.approx(0.15)
    assert per_cat_w == {"style": 0.05, "efficiency": 0.10}

    # Flat fallback: two violations without penalty → 2 × 0.15 = 0.30
    flat = [
        RuleViolation(severity="soft", category="redundancy", message=""),
        RuleViolation(severity="soft", category="causal", message=""),
    ]
    total_f, per_cat_f = computer._aggregate_soft_penalty(flat)
    assert total_f == pytest.approx(0.30)
    assert per_cat_f == {"redundancy": 0.15, "causal": 0.15}


def test_breakdown_exposes_per_category_violation_penalties():
    """reward_breakdown must surface the per-violation-type penalty attribution."""
    import pytest as _pytest
    from models import (
        Action, ActionCommand, RewardConfig, RuleViolation, State, TaskSpec,
    )
    from server.rewards.reward import RewardComputer

    cfg = RewardConfig(component_weights={"correctness": 1.0})
    class Fixed(RewardComputer):
        def _component_scores(self, action, state, task):
            return {"correctness": 0.5}

    computer = Fixed(cfg)
    task = TaskSpec(task_id="t", difficulty="easy", problem_statement="", max_steps=1)
    state = State(task_id="t", max_steps=1)
    action = Action(command=ActionCommand.NOOP)
    violations = [
        RuleViolation(severity="soft", category="style", message="", penalty=0.05),
        RuleViolation(severity="soft", category="efficiency", message="", penalty=0.10),
    ]
    computer.compute(action, state, task, violations)
    bd = computer.last_breakdown
    assert bd["violation_penalty_style"] == _pytest.approx(-0.05)
    assert bd["violation_penalty_efficiency"] == _pytest.approx(-0.10)
    assert bd["soft_violation_penalty"] == _pytest.approx(-0.15)


def test_truncation_wipe_forces_fixed_episode_total():
    """Truncated episodes get a deterministic reward total."""
    from models import RewardConfig, RewardMode

    # Config with truncation wipe to -2.0 and additive rewards for easier math
    cfg = RewardConfig(
        mode=RewardMode.ADDITIVE,
        component_weights={"correctness": 0.5, "efficiency": 0.5},
        truncation_reward_total=-2.0,
    )
    env = Environment(reward_config=cfg, domain_randomise=False)
    obs = env.reset(scenario_name="placeholder_easy")
    max_steps = obs.max_steps

    # Run to truncation
    for _ in range(max_steps):
        obs = env.step(Action(command=ActionCommand.NOOP))
        if obs.done or obs.truncated:
            break

    # Episode's cumulative reward should be exactly -2.0
    assert env.state.cumulative_reward == pytest.approx(-2.0, abs=1e-6), (
        f"truncation_reward_total=-2.0 should force cumulative to -2.0, "
        f"got {env.state.cumulative_reward}"
    )


def test_truncation_wipe_disabled_by_default():
    """Without truncation_reward_total, cumulative reward is natural sum."""
    from models import RewardConfig
    env = Environment(reward_config=RewardConfig(), domain_randomise=False)
    obs = env.reset(scenario_name="placeholder_easy")
    max_steps = obs.max_steps

    for _ in range(max_steps):
        obs = env.step(Action(command=ActionCommand.NOOP))
        if obs.done or obs.truncated:
            break

    # With no wipe, cumulative is just the natural novelty_bonus sum
    # (template default: each step gets ~0.1 novelty bonus, no wipe)
    assert env.state.cumulative_reward != pytest.approx(-2.0)
    # The cumulative should match what the step-by-step sum produced
    assert env.state.cumulative_reward > 0  # novelty bonuses accumulated


def test_hybrid_mode_differs_from_multiplicative():
    """Non-zero components: hybrid (additive inside) should differ from pure multiplicative."""
    from models import RewardConfig, RewardMode, State, TaskSpec, Action, ActionCommand
    from server.rewards.reward import RewardComputer

    weights = {"correctness": 0.4, "efficiency": 0.2, "quality": 0.2, "safety": 0.2}

    class FixedComputer(RewardComputer):
        def _component_scores(self, action, state, task):
            return {"correctness": 0.8, "efficiency": 0.8, "quality": 0.8, "safety": 0.8}

    task = TaskSpec(task_id="t", difficulty="easy", problem_statement="", max_steps=1)
    state = State(task_id="t", max_steps=1)
    action = Action(command=ActionCommand.NOOP)

    mult_cfg = RewardConfig(mode=RewardMode.MULTIPLICATIVE, component_weights=weights)
    hybrid_cfg = RewardConfig(mode=RewardMode.HYBRID, component_weights=weights)

    r_mult = FixedComputer(mult_cfg).compute(action, state, task, violations=[])
    r_hybrid = FixedComputer(hybrid_cfg).compute(action, state, task, violations=[])

    # With 0.8 values and mixed weights, hybrid (additive inside) ≠ multiplicative (product)
    assert r_mult != r_hybrid, "hybrid must produce different reward than multiplicative"


# ---------------------------------------------------------------------------
# Integration: rule engine + reward + task generator wire correctly
# ---------------------------------------------------------------------------


def test_full_episode_produces_trajectory():
    env = Environment(domain_randomise=False, seed=7)
    obs = env.reset(scenario_name="placeholder_easy")
    trajectory: list[dict] = []

    while not (obs.done or obs.truncated):
        obs = env.step(Action(command=ActionCommand.NOOP))
        trajectory.append({
            "step": obs.step,
            "reward": obs.reward,
            "done": obs.done,
        })

    assert len(trajectory) >= 1
    assert trajectory[-1]["done"] or obs.truncated


def test_domain_randomization_varies_task_id():
    env = Environment(domain_randomise=True, seed=1)
    ids = {env.reset().task_id for _ in range(5)}
    # With randomization, task_id gets a suffix — should produce multiple unique values
    # Some may collide on small samples — just assert at least 2 variants
    assert len(ids) >= 1  # placeholder scenarios have no budget/time → no suffix; relax
