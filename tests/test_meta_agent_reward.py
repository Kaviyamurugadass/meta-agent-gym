"""Tests for MetaAgentRewardComputer."""

import pytest

from models import Action, ActionCommand, RewardConfig, RewardMode, State, TaskSpec
from server.rewards.reward import MetaAgentRewardComputer


@pytest.fixture
def reward_config():
    """Create a test reward config."""
    return RewardConfig(
        mode=RewardMode.HYBRID,
        component_weights={
            "skill_selection": 0.25,
            "description_quality": 0.20,
            "workflow_clarity": 0.20,
            "model_appropriateness": 0.15,
            "best_practices": 0.10,
            "efficiency": 0.10,
        },
    )


@pytest.fixture
def easy_task():
    """Create an easy test task."""
    return TaskSpec(
        task_id="ws_easy_001",
        domain="web",
        difficulty="easy",
        problem_statement="Scrape a web page",
        max_steps=7,
        required_skills=["web-scraping"],
        recommended_skills=[],
    )


@pytest.fixture
def empty_state(easy_task):
    """Create an empty state for testing."""
    return State(
        task_id=easy_task.task_id,
        step=0,
        max_steps=easy_task.max_steps,
        current_spec={},
    )


def test_reward_computer_init(reward_config):
    """Test reward computer initialization."""
    computer = MetaAgentRewardComputer(reward_config)

    assert computer.config == reward_config
    assert computer._last_breakdown == {}


def test_compute_empty_spec(reward_config, empty_state, easy_task):
    """Test computing reward for empty spec (should trigger gates)."""
    computer = MetaAgentRewardComputer(reward_config)

    action = Action(command=ActionCommand.SET_NAME, args={"name": "test"})
    empty_state.current_spec = {"name": "test"}

    reward = computer.compute(action, empty_state, easy_task, [])

    # Empty spec should fail gates → reward = 0
    assert reward == 0.0
    assert computer._last_breakdown.get("gated") == 1.0


def test_compute_valid_spec_passes_gates(reward_config, empty_state, easy_task):
    """Test computing reward for valid spec (passes gates)."""
    computer = MetaAgentRewardComputer(reward_config)

    # Create a minimal valid spec
    empty_state.current_spec = {
        "name": "web-scraper",
        "description": "A web scraping agent that extracts data",
        "system_prompt": "You are a web scraping specialist. " + "A" * 100,
        "skills": ["web-scraping"],
        "model": "sonnet",
    }

    action = Action(command=ActionCommand.SUBMIT)
    reward = computer.compute(action, empty_state, easy_task, [])

    # Should pass gates and get positive reward
    assert reward > 0.0
    assert computer._last_breakdown.get("yaml_valid") == 1.0
    assert computer._last_breakdown.get("has_required_fields") == 1.0


def test_skill_selection_score(reward_config, empty_state, easy_task):
    """Test skill selection scoring."""
    computer = MetaAgentRewardComputer(reward_config)

    # Perfect match
    empty_state.current_spec = {
        "name": "test",
        "description": "Test",
        "system_prompt": "A" * 100,
        "skills": ["web-scraping"],  # Exactly what's required
        "model": "sonnet",
    }

    action = Action(command=ActionCommand.SUBMIT)
    computer.compute(action, empty_state, easy_task, [])

    skill_score = computer._last_breakdown.get("skill_selection", 0)
    assert skill_score >= 0.8  # High score for perfect match


def test_skill_selection_missing_required(reward_config, empty_state, easy_task):
    """Test skill selection with missing required skills."""
    computer = MetaAgentRewardComputer(reward_config)

    empty_state.current_spec = {
        "name": "test",
        "description": "Test",
        "system_prompt": "A" * 100,
        "skills": [],  # Missing required skill
        "model": "sonnet",
    }

    action = Action(command=ActionCommand.SUBMIT)
    computer.compute(action, empty_state, easy_task, [])

    skill_score = computer._last_breakdown.get("skill_selection", 0)
    assert skill_score < 0.5  # Low score for missing required skill


def test_anti_hack_empty_spec_penalty(reward_config, empty_state, easy_task):
    """Test anti-hacking penalty for empty spec."""
    config = RewardConfig(
        mode=RewardMode.ADDITIVE,  # Use ADDITIVE to avoid gate blocking
        min_prompt_length=50,
        anti_hack_empty_spec=-5.0,
    )
    computer = MetaAgentRewardComputer(config)

    # Spec with prompt that's too short
    empty_state.current_spec = {
        "name": "test-agent",
        "description": "A proper description that passes gates",
        "system_prompt": "Short",  # Too short
        "model": "sonnet",
    }

    action = Action(command=ActionCommand.SUBMIT)
    computer.compute(action, empty_state, easy_task, [])

    # Check for anti-hack penalty
    anti_hack = computer._last_breakdown.get("anti_hack_empty_spec", 0)
    assert anti_hack == -5.0


def test_anti_hack_over_engineered(reward_config, empty_state, easy_task):
    """Test anti-hacking penalty for over-engineered spec."""
    config = RewardConfig(
        mode=RewardMode.HYBRID,
        max_skills_limit=10,
        anti_hack_over_engineered=-0.5,
    )
    computer = MetaAgentRewardComputer(config)

    # Spec with too many skills
    empty_state.current_spec = {
        "name": "test",
        "description": "Test",
        "system_prompt": "A" * 100,
        "skills": [f"skill-{i}" for i in range(15)],  # Over limit
        "model": "sonnet",
    }

    action = Action(command=ActionCommand.SUBMIT)
    computer.compute(action, empty_state, easy_task, [])

    # Check for anti-hack penalty
    anti_hack = computer._last_breakdown.get("anti_hack_over_engineered", 0)
    assert anti_hack == -0.5


def test_investigation_command_no_reward(reward_config, empty_state, easy_task):
    """Test that investigation commands don't generate judge rewards."""
    computer = MetaAgentRewardComputer(reward_config)

    action = Action(command=ActionCommand.CHECK_SCORE)
    empty_state.current_spec = {}

    reward = computer.compute(action, empty_state, easy_task, [])

    # Investigation commands should have minimal/no judge components
    # Just hard verifiers run
    assert "skill_selection" not in computer._last_breakdown or \
           computer._last_breakdown.get("skill_selection") == 0


def test_reward_breakdown_contains_all_components(reward_config, empty_state, easy_task):
    """Test that reward breakdown contains all expected components."""
    computer = MetaAgentRewardComputer(reward_config)

    empty_state.current_spec = {
        "name": "test",
        "description": "A test agent that handles tasks proactively",
        "system_prompt": "You are a specialist. " + "A" * 100,
        "skills": ["web-scraping"],
        "model": "sonnet",
    }

    action = Action(command=ActionCommand.SUBMIT)
    computer.compute(action, empty_state, easy_task, [])

    breakdown = computer._last_breakdown

    # Check hard verifier components
    assert "yaml_valid" in breakdown
    assert "has_required_fields" in breakdown
    assert "prompt_length_ok" in breakdown

    # Check judge components
    assert "skill_selection" in breakdown
    assert "description_quality" in breakdown
    assert "workflow_clarity" in breakdown
    assert "model_appropriateness" in breakdown
    assert "best_practices" in breakdown
    assert "efficiency" in breakdown

    # Check total exists
    assert "total" in breakdown
