"""OpenEnv HTTP/WebSocket client wrapper.

Thin wrapper around requests that knows the OpenEnv endpoint contract.
Used by inference.py, notebooks, tests, and baseline_rollout.sh.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Optional

import httpx

from models import Action, Observation


class Env(AbstractContextManager["Env"]):
    """Client for a live OpenEnv server.

    Usage:
        with Env("http://localhost:8000") as env:
            obs = env.reset()
            obs = env.step(Action(command=ActionCommand.NOOP))
            print(env.state())
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------ Lifecycle

    def reset(self, scenario_name: Optional[str] = None) -> Observation:
        """Start a new episode. Optionally pin to a named scenario."""
        body: dict[str, Any] = {}
        if scenario_name:
            body["scenario_name"] = scenario_name
        resp = self._client.post("/reset", json=body)
        resp.raise_for_status()
        return _parse_observation(resp.json())

    def step(self, action: Action) -> Observation:
        """Take one action. Returns the next observation."""
        # OpenEnv /step expects body: {"action": {...}}
        resp = self._client.post("/step", json={"action": action.model_dump()})
        resp.raise_for_status()
        return _parse_observation(resp.json())

    def state(self) -> dict[str, Any]:
        """Full state (debug/eval only — NOT for the agent)."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        data = resp.json()
        return data.get("state", data)

    def schema(self) -> dict[str, Any]:
        resp = self._client.get("/schema")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict[str, Any]:
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ Cleanup

    def close(self) -> None:
        self._client.close()

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, D401
        self.close()


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _parse_observation(payload: dict[str, Any]) -> Observation:
    """Extract Observation from OpenEnv response envelope."""
    obs_data = payload.get("observation", payload)
    return Observation.model_validate(obs_data)
