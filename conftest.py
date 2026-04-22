"""Shared pytest fixtures for the OpenEnv R2 Kit test suite."""

from __future__ import annotations

import pytest

from models import (
    Action,
    ActionCommand,
    Observation,
    RewardConfig,
    RewardMode,
    RuleViolation,
    TaskSpec,
)


# ---------------------------------------------------------------------------
# Task / scenario fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_task() -> TaskSpec:
    """Minimal task spec for env/reward tests."""
    return TaskSpec(
        task_id="test_task_001",
        difficulty="easy",
        problem_statement="Test task for unit tests.",
        max_steps=5,
        citations=["test://citation"],
        expected_findings={"key": "value"},
    )


@pytest.fixture
def hard_task() -> TaskSpec:
    """Harder task with budget + time constraints."""
    return TaskSpec(
        task_id="test_task_hard",
        difficulty="hard",
        problem_statement="Constrained task with budget and time limit.",
        max_steps=15,
        citations=["test://hard"],
        expected_findings={"milestones": 5, "critical": True},
        budget=100.0,
        time_limit=50.0,
    )


# ---------------------------------------------------------------------------
# Action fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def noop_action() -> Action:
    """NOOP action — safe default for step tests."""
    return Action(command=ActionCommand.NOOP, args={})


@pytest.fixture
def submit_action() -> Action:
    """SUBMIT action — terminates episode."""
    return Action(command=ActionCommand.SUBMIT, args={})


@pytest.fixture
def inspect_action() -> Action:
    """INSPECT action — investigation-tool pattern."""
    return Action(
        command=ActionCommand.INSPECT,
        args={"target": "example"},
        justification="Investigating before acting.",
        confidence=0.7,
    )


# ---------------------------------------------------------------------------
# Observation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def initial_observation() -> Observation:
    """Observation at step 0 after reset."""
    return Observation(
        task_id="test_task_001",
        step=0,
        max_steps=5,
        summary="Initial observation",
    )


@pytest.fixture
def mid_episode_observation() -> Observation:
    """Observation mid-episode with history."""
    return Observation(
        task_id="test_task_001",
        step=3,
        max_steps=5,
        summary="Mid-episode observation",
        history=[
            {"action": "noop", "reward": 0.0},
            {"action": "inspect", "reward": 0.1},
            {"action": "noop", "reward": 0.05},
        ],
        reward_breakdown={"correctness": 0.5, "efficiency": 0.2},
    )


# ---------------------------------------------------------------------------
# Reward config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def multiplicative_reward_config() -> RewardConfig:
    """Default multiplicative reward config."""
    return RewardConfig(
        mode=RewardMode.MULTIPLICATIVE,
        component_weights={
            "correctness": 0.4,
            "efficiency": 0.2,
            "quality": 0.2,
            "safety": 0.2,
        },
    )


@pytest.fixture
def additive_reward_config() -> RewardConfig:
    """Additive reward config."""
    return RewardConfig(
        mode=RewardMode.ADDITIVE,
        component_weights={
            "validity": 0.3,
            "ordering": 0.2,
            "info_gain": 0.4,
            "efficiency": 0.3,
        },
    )


# ---------------------------------------------------------------------------
# Rule violation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hard_violation() -> RuleViolation:
    return RuleViolation(
        severity="hard",
        category="prerequisite",
        message="Action blocked: prerequisite not met.",
    )


@pytest.fixture
def soft_violation() -> RuleViolation:
    return RuleViolation(
        severity="soft",
        category="redundancy",
        message="Redundant action — quality degraded.",
    )
