"""Reward backend abstraction — local (in-process) vs remote (HTTP server).

During GRPO training you generate many completions per prompt. For each,
you need the env's reward given that completion as an action sequence.

Two modes:
    local  — instantiate Environment in-process, call reset/step directly.
             Fastest — no network overhead. Use during training.
    remote — hit the deployed OpenEnv server via HTTP.
             Ground truth. Use for final eval against live deploy.

Training script takes `--reward-backend {local,remote}`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from models import Action, Observation
from server.environment import Environment

try:
    from client import Env as HTTPClient
except ImportError:
    HTTPClient = None  # type: ignore[assignment,misc]


class RewardBackend(ABC):
    """Computes rewards for an action sequence, one trajectory at a time."""

    @abstractmethod
    def score(
        self,
        actions: list[Action],
        scenario_name: Optional[str] = None,
    ) -> tuple[float, list[Observation]]:
        """Run actions through a fresh episode; return (total_reward, observations)."""


class LocalBackend(RewardBackend):
    """Instantiates Environment in-process — no HTTP overhead.

    Use during training. Matches the deployed env exactly because it IS
    the same code.
    """

    def __init__(self, env_factory=None) -> None:  # type: ignore[no-untyped-def]
        self._env_factory = env_factory or Environment

    def score(
        self,
        actions: list[Action],
        scenario_name: Optional[str] = None,
    ) -> tuple[float, list[Observation]]:
        env = self._env_factory()
        obs = env.reset(scenario_name=scenario_name)
        observations = [obs]
        total = 0.0
        for action in actions:
            if obs.done or obs.truncated:
                break
            obs = env.step(action)
            observations.append(obs)
            total += obs.reward
        return total, observations


class RemoteBackend(RewardBackend):
    """Hits a deployed OpenEnv server over HTTP.

    Use for final eval against the live deploy — proves the env in the
    HF Space gives the same rewards as training-time env.

    Note: HTTP session is per-request (stateless). A full episode requires
    WebSocket — this remote path is for single-step-evaluation use cases
    (e.g., "reward for this completion given the task context").
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        if HTTPClient is None:
            raise RuntimeError(
                "client.Env is not available — install httpx via `uv sync`"
            )
        self.base_url = base_url

    def score(
        self,
        actions: list[Action],
        scenario_name: Optional[str] = None,
    ) -> tuple[float, list[Observation]]:
        with HTTPClient(self.base_url) as env:
            obs = env.reset(scenario_name=scenario_name)
            observations = [obs]
            total = 0.0
            for action in actions:
                if obs.done or obs.truncated:
                    break
                obs = env.step(action)
                observations.append(obs)
                total += obs.reward
            return total, observations


def make_backend(mode: str, base_url: Optional[str] = None) -> RewardBackend:
    """Factory for training scripts: `--reward-backend {local,remote}`."""
    if mode == "local":
        return LocalBackend()
    if mode == "remote":
        if not base_url:
            raise ValueError("remote backend requires base_url")
        return RemoteBackend(base_url)
    raise ValueError(f"Unknown backend: {mode}. Use 'local' or 'remote'.")
