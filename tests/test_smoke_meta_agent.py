"""Smoke test for meta-agent environment — end-to-end flow."""

from models import Action, ActionCommand, RewardConfig, RewardMode
from server.environment import Environment


def test_end_to_end_agent_generation():
    """Test complete agent generation workflow."""
    env = Environment(reward_config=RewardConfig(mode=RewardMode.HYBRID))

    # Reset with easy task
    obs = env.reset(scenario_name="ws_easy_001")
    assert obs.step == 0
    assert obs.task_id == "ws_easy_001"

    # Step 1: Set name
    obs = env.step(Action(command=ActionCommand.SET_NAME, args={"name": "price-scraper"}))
    assert obs.current_spec["name"] == "price-scraper"
    assert obs.step == 1

    # Step 2: Set description
    obs = env.step(Action(
        command=ActionCommand.SET_DESCRIPTION,
        args={"description": "Extracts product prices from e-commerce pages"}
    ))
    assert obs.current_spec["description"] == "Extracts product prices from e-commerce pages"
    assert obs.step == 2

    # Step 3: Add required skill
    obs = env.step(Action(command=ActionCommand.ADD_SKILL, args={"skill": "web-scraping"}))
    assert "web-scraping" in obs.current_spec["skills"]
    assert obs.step == 3

    # Step 4: Check score (investigation)
    obs = env.step(Action(command=ActionCommand.CHECK_SCORE))
    assert obs.investigation_result is not None
    assert "current_score" in obs.investigation_result
    assert obs.step == 4

    # Step 5: Write prompt
    obs = env.step(Action(
        command=ActionCommand.WRITE_PROMPT,
        args={
            "prompt": """You are a web scraping specialist.

When scraping:
1. Identify the target structure
2. Extract product prices
3. Return structured JSON data
"""
        }
    ))
    assert "web scraping" in obs.current_spec["system_prompt"].lower()
    assert obs.step == 5

    # Step 6: Set model
    obs = env.step(Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}))
    assert obs.current_spec["model"] == "sonnet"
    assert obs.step == 6

    # Step 7: Submit
    obs = env.step(Action(command=ActionCommand.SUBMIT))

    # Should be done (complete spec with all required fields)
    assert obs.done is True

    # Check reward breakdown
    assert "total" in obs.reward_breakdown
    assert obs.reward_breakdown["total"] >= 0  # Should have positive score

    # Check final spec can be converted to AgentSpec
    from models import AgentSpec
    spec = AgentSpec(
        name=obs.current_spec["name"],
        description=obs.current_spec["description"],
        skills=obs.current_spec.get("skills", []),
        model=obs.current_spec.get("model", "sonnet"),
        system_prompt=obs.current_spec.get("system_prompt", ""),
    )

    # Can convert to markdown
    markdown = spec.to_markdown()
    assert "---" in markdown
    assert "price-scraper" in markdown

    print(f"\n[PASS] End-to-end test passed!")
    print(f"Final score: {obs.reward_breakdown['total']:.2f}")
    print(f"Steps taken: {obs.step}")
    print(f"Generated AGENT.md preview:\n{markdown[:200]}...")


if __name__ == "__main__":
    test_end_to_end_agent_generation()
