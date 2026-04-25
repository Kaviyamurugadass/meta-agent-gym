"""Tests for meta-agent environment."""

import pytest

from models import Action, ActionCommand, RewardConfig, RewardMode
from server.environment import Environment


@pytest.fixture
def env():
    """Create a test environment."""
    config = RewardConfig(mode=RewardMode.HYBRID)
    return Environment(reward_config=config, curriculum_phase=1)


def test_environment_reset(env):
    """Test environment reset."""
    obs = env.reset(scenario_name="ws_easy_001")

    assert obs.task_id == "ws_easy_001"
    assert obs.step == 0
    assert obs.max_steps > 0
    assert obs.done is False
    assert obs.truncated is False
    assert obs.reward == 0.0


def test_environment_set_name(env):
    """Test setting agent name."""
    env.reset()

    action = Action(command=ActionCommand.SET_NAME, args={"name": "test-agent"})
    obs = env.step(action)

    assert obs.current_spec.get("name") == "test-agent"
    assert obs.step == 1


def test_environment_add_skill(env):
    """Test adding skills."""
    env.reset()

    action = Action(command=ActionCommand.ADD_SKILL, args={"skill": "web-scraping"})
    obs = env.step(action)

    assert "web-scraping" in obs.current_spec.get("skills", [])


def test_environment_add_multiple_skills(env):
    """Test adding multiple skills."""
    env.reset()

    env.step(Action(command=ActionCommand.ADD_SKILL, args={"skill": "web-scraping"}))
    env.step(Action(command=ActionCommand.ADD_SKILL, args={"skill": "http-client"}))
    obs = env.step(Action(command=ActionCommand.ADD_SKILL, args={"skill": "web-scraping"}))

    # Should dedupe
    skills = obs.current_spec.get("skills", [])
    assert "web-scraping" in skills
    assert "http-client" in skills
    assert len(skills) == 2  # Deduped


def test_environment_remove_skill(env):
    """Test removing skills."""
    env.reset()

    env.step(Action(command=ActionCommand.ADD_SKILL, args={"skill": "web-scraping"}))
    obs = env.step(Action(command=ActionCommand.REMOVE_SKILL, args={"skill": "web-scraping"}))

    assert "web-scraping" not in obs.current_spec.get("skills", [])


def test_environment_set_description(env):
    """Test setting description."""
    env.reset()

    action = Action(
        command=ActionCommand.SET_DESCRIPTION,
        args={"description": "A web scraping agent"}
    )
    obs = env.step(action)

    assert obs.current_spec.get("description") == "A web scraping agent"


def test_environment_write_prompt_replace(env):
    """Test writing prompt (replace mode)."""
    env.reset()

    action = Action(
        command=ActionCommand.WRITE_PROMPT,
        args={"prompt": "You are a specialist.", "mode": "replace"}
    )
    obs = env.step(action)

    assert obs.current_spec.get("system_prompt") == "You are a specialist."


def test_environment_write_prompt_append(env):
    """Test writing prompt (append mode)."""
    env.reset()

    env.step(Action(
        command=ActionCommand.WRITE_PROMPT,
        args={"prompt": "Line 1", "mode": "replace"}
    ))
    obs = env.step(Action(
        command=ActionCommand.WRITE_PROMPT,
        args={"prompt": "Line 2", "mode": "append"}
    ))

    prompt = obs.current_spec.get("system_prompt", "")
    assert "Line 1" in prompt
    assert "Line 2" in prompt


def test_environment_set_model(env):
    """Test setting model."""
    env.reset()

    action = Action(command=ActionCommand.SET_MODEL, args={"model": "opus"})
    obs = env.step(action)

    assert obs.current_spec.get("model") == "opus"


def test_environment_check_score_investigation(env):
    """Test CHECK_SCORE investigation command."""
    env.reset()

    # Build a minimal spec first
    env.step(Action(command=ActionCommand.SET_NAME, args={"name": "test"}))
    env.step(Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Test"}))
    env.step(Action(command=ActionCommand.WRITE_PROMPT, args={"prompt": "A" * 100}))

    obs = env.step(Action(command=ActionCommand.CHECK_SCORE))

    # Should have investigation result
    assert obs.investigation_result is not None
    assert "current_score" in obs.investigation_result


def test_environment_inspect_example_investigation(env):
    """Test INSPECT_EXAMPLE investigation command."""
    env.reset()

    obs = env.step(Action(command=ActionCommand.INSPECT_EXAMPLE))

    # Should have investigation result with template
    assert obs.investigation_result is not None
    assert "example_template" in obs.investigation_result


def test_environment_submit_incomplete(env):
    """Test SUBMIT with incomplete spec (should fail gate)."""
    env.reset()

    # Incomplete spec - just name
    env.step(Action(command=ActionCommand.SET_NAME, args={"name": "test"}))

    obs = env.step(Action(command=ActionCommand.SUBMIT))

    # Should not be done (failed gate)
    assert obs.done is False


def test_environment_submit_complete(env):
    """Test SUBMIT with complete spec."""
    env.reset()

    # Build complete spec
    env.step(Action(command=ActionCommand.SET_NAME, args={"name": "web-scraper"}))
    env.step(Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Scrapes web data"}))
    env.step(Action(command=ActionCommand.ADD_SKILL, args={"skill": "web-scraping"}))
    env.step(Action(command=ActionCommand.WRITE_PROMPT, args={"prompt": "A" * 100}))
    env.step(Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}))

    obs = env.step(Action(command=ActionCommand.SUBMIT))

    # Should be done (passed gates)
    assert obs.done is True


def test_environment_truncation(env):
    """Test episode truncation at max_steps."""
    # Use a task with small max_steps
    obs = env.reset(scenario_name="ws_easy_001")

    # Run until max_steps
    for _ in range(obs.max_steps + 5):
        if obs.done:
            break
        obs = env.step(Action(command=ActionCommand.NOOP))

    assert obs.truncated is True or obs.done is True


def test_environment_available_skills(env):
    """Test that available skills are populated and resolve to known skill IDs."""
    from server.skills import AVAILABLE_SKILLS

    obs = env.reset()

    assert len(obs.available_skills) > 0
    # Every advertised skill should be a valid entry in the skills.sh registry
    for skill_id in obs.available_skills:
        assert skill_id in AVAILABLE_SKILLS, f"unknown skill in observation: {skill_id}"


def test_environment_feedback(env):
    """Test feedback generation."""
    env.reset()

    # Create an empty spec (should generate negative feedback)
    obs = env.step(Action(command=ActionCommand.SUBMIT))

    # Should have feedback
    if obs.reward < 0:
        assert len(obs.feedback) > 0


def test_environment_state_persistence(env):
    """Test that state persists across steps."""
    env.reset()

    env.step(Action(command=ActionCommand.SET_NAME, args={"name": "test"}))
    obs1 = env.step(Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Test"}))

    # State should have both values
    assert env.state.current_spec.get("name") == "test"
    assert env.state.current_spec.get("description") == "Test"
    assert obs1.current_spec.get("name") == "test"


def test_environment_reward_breakdown(env):
    """Test that reward breakdown is populated."""
    env.reset()

    env.step(Action(command=ActionCommand.SET_NAME, args={"name": "test"}))
    env.step(Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Test agent"}))
    env.step(Action(command=ActionCommand.WRITE_PROMPT, args={"prompt": "A" * 100}))
    obs = env.step(Action(command=ActionCommand.SUBMIT))

    # Should have reward breakdown
    assert len(obs.reward_breakdown) > 0
    assert "total" in obs.reward_breakdown
