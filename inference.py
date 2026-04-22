"""LLM inference — run an episode with a base model, collect trajectory.

Triple-fallback JSON parser:
    1. Direct json.loads
    2. Extract from ```json fenced block
    3. Regex-extract first {...} object

System prompt templates — swap by env var LLM_BACKEND (openai/anthropic/local).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import httpx

from client import Env
from models import Action, ActionCommand, Observation


# ---------------------------------------------------------------------------
# System prompt templates — DOMAIN: extend with domain-specific hints
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """You are an agent interacting with an OpenEnv environment.

Each turn you receive an Observation and must respond with a single Action as JSON.

Action schema:
    {
        "command": "<one of: inspect, submit, noop>",
        "args": {<command-specific arguments>},
        "justification": "<brief reason>",
        "confidence": <0.0 to 1.0>
    }

Respond ONLY with the JSON object. No explanation, no prose.
"""


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------


def call_openai_compatible(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT_BASE,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint.

    Works with OpenAI, OpenRouter (free models), any compatible gateway.
    """
    api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    model = model or os.getenv("OPENAI_MODEL", "meta-llama/llama-3.2-3b-instruct:free")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Copy .env.example → .env and fill it in."
        )

    resp = httpx.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Triple-fallback JSON parser
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_action_json(text: str) -> dict[str, Any]:
    """Robust JSON extraction from LLM output.

    Triple fallback:
        1. json.loads on full text
        2. Extract from ```json fenced block
        3. First balanced {...} object via regex
    """
    text = text.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fenced code block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Attempt 3: first {...}
    m = _JSON_OBJECT_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM output:\n{text[:500]}")


def parse_action(text: str) -> Action:
    """Parse LLM output into a validated Action. Falls back to NOOP on failure."""
    try:
        raw = parse_action_json(text)
        return Action.model_validate(raw)
    except (ValueError, Exception):
        # Graceful fallback so episodes don't abort on one bad completion
        return Action(
            command=ActionCommand.NOOP,
            args={},
            justification="Fallback: LLM output could not be parsed as Action.",
            confidence=0.0,
        )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    env_url: str = "http://localhost:8000",
    scenario_name: Optional[str] = None,
    system_prompt: str = SYSTEM_PROMPT_BASE,
    model: Optional[str] = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run one episode to completion. Returns trajectory as list of steps."""
    trajectory: list[dict[str, Any]] = []
    with Env(env_url) as env:
        obs = env.reset(scenario_name=scenario_name)
        if verbose:
            print(f"[reset] task={obs.task_id} step={obs.step}/{obs.max_steps}")

        while not (obs.done or obs.truncated):
            prompt = _format_observation(obs)
            try:
                completion = call_openai_compatible(prompt, system_prompt, model=model)
                action = parse_action(completion)
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"[error] {e}; falling back to NOOP")
                action = Action(command=ActionCommand.NOOP)

            obs = env.step(action)
            trajectory.append({
                "step": obs.step,
                "action": action.model_dump(),
                "observation": obs.model_dump(),
            })

            if verbose:
                print(
                    f"[step {obs.step}/{obs.max_steps}] "
                    f"action={action.command.value} reward={obs.reward:.3f} "
                    f"violations={len(obs.rule_violations)}"
                )

        if verbose:
            print(f"[end] total_steps={obs.step} done={obs.done}")
    return trajectory


def _format_observation(obs: Observation) -> str:
    """Render observation for LLM prompt."""
    return json.dumps(obs.model_dump(exclude_none=True), indent=2)


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    run_episode(env_url=url)
