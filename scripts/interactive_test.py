"""Interactive test script for meta-agent environment."""

from models import Action, ActionCommand, RewardConfig, RewardMode
from server.environment import Environment


def main():
    print("=" * 60)
    print("Meta-Agent Environment - Interactive Test")
    print("=" * 60)

    # Create environment
    env = Environment(
        reward_config=RewardConfig(mode=RewardMode.HYBRID),
        curriculum_phase=1
    )

    # Reset with easy task
    obs = env.reset(scenario_name="ws_easy_001")
    print(f"\n[RESET] Task: {obs.task_id}")
    print(f"        Description: {env._task.problem_statement}")
    print(f"        Required Skills: {env._task.required_skills}")
    print(f"        Max Steps: {obs.max_steps}")

    # Build agent step by step
    steps = [
        (ActionCommand.SET_NAME, {"name": "price-scraper"}, "Set name to 'price-scraper'"),
        (ActionCommand.SET_DESCRIPTION, {"description": "Extract product prices from e-commerce pages"}, "Set description"),
        (ActionCommand.ADD_SKILL, {"skill": "web-scraping"}, "Add web-scraping skill"),
        (ActionCommand.CHECK_SCORE, {}, "Check current score"),
        (ActionCommand.WRITE_PROMPT, {"prompt": "You are a web scraping specialist. Extract prices from product pages."}, "Write system prompt"),
        (ActionCommand.SET_MODEL, {"model": "sonnet"}, "Set model to sonnet"),
        (ActionCommand.SUBMIT, {}, "Submit agent"),
    ]

    for i, (cmd, args, desc) in enumerate(steps, 1):
        print(f"\n[STEP {i}] {desc}")
        obs = env.step(Action(command=cmd, args=args))
        print(f"  -> Reward: {obs.reward:.3f}")
        print(f"  -> Score: {obs.score:.3f}")
        print(f"  -> Current Spec: {obs.current_spec}")

        if obs.investigation_result:
            print(f"  -> Investigation: {obs.investigation_result.get('current_score', 'N/A')}")

        if obs.feedback:
            print(f"  -> Feedback: {obs.feedback}")

        if obs.done:
            print(f"\n[DONE] Episode complete!")
            break

    # Show final agent
    print("\n" + "=" * 60)
    print("FINAL AGENT SPECIFICATION")
    print("=" * 60)
    print(f"Name: {obs.current_spec.get('name')}")
    print(f"Description: {obs.current_spec.get('description')}")
    print(f"Skills: {obs.current_spec.get('skills', [])}")
    print(f"Model: {obs.current_spec.get('model')}")
    print(f"System Prompt: {obs.current_spec.get('system_prompt', '')[:100]}...")

    # Convert to AGENT.md format
    from models import AgentSpec
    spec = AgentSpec(
        name=obs.current_spec["name"],
        description=obs.current_spec["description"],
        skills=obs.current_spec.get("skills", []),
        model=obs.current_spec.get("model", "sonnet"),
        system_prompt=obs.current_spec.get("system_prompt", ""),
    )

    print("\n" + "=" * 60)
    print("GENERATED AGENT.md")
    print("=" * 60)
    print(spec.to_markdown())

    print("\n" + "=" * 60)
    print("REWARD BREAKDOWN")
    print("=" * 60)
    for key, value in obs.reward_breakdown.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
