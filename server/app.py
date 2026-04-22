"""FastAPI app — mounts Environment onto OpenEnv server.

Gotcha #1 (from TEMPLATE_PLAN.md): ``max_concurrent_envs`` MUST be >=4 for
parallel GRPO. OpenEnv's default is 1, which crashes during TRL training
when the trainer spins up multiple rollouts. Override via env var.

Endpoints (exposed by OpenEnv's create_app):
    POST /reset   — start new episode
    POST /step    — apply action
    GET  /state   — full state (debug/eval)
    GET  /schema  — Action/Observation JSON schemas
    WS   /ws      — persistent session
Plus:
    GET  /health  — liveness check
    GET  /web     — static dashboard (if static/index.html exists)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from models import Action, Observation
from server.environment import Environment

# ---------------------------------------------------------------------------
# Try to use the official OpenEnv create_app helper. Fall back to a minimal
# FastAPI wrapper if openenv-core isn't available (e.g. during local dev).
# ---------------------------------------------------------------------------

MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "4"))

try:
    from openenv.core import create_app  # type: ignore[import-not-found]

    # create_app expects: (env_factory, action_cls, observation_cls, ...)
    # Environment's __init__ takes only optional args, so the class itself
    # works as a zero-arg factory.
    app = create_app(
        Environment,
        Action,
        Observation,
        env_name="openenv-r2-kit",
        max_concurrent_envs=MAX_CONCURRENT_ENVS,
    )
    _USING_OPENENV = True
except ImportError:
    # --- Fallback: minimal FastAPI wrapper ---------------------------------
    # Lets uvicorn start the server even without openenv-core installed.
    # On finale day, rely on the real create_app path.
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="openenv-r2-kit (fallback mode)")
    _env = Environment()
    _USING_OPENENV = False

    @app.post("/reset")
    async def _reset(body: dict[str, Any] | None = None) -> dict[str, Any]:
        scenario = (body or {}).get("scenario_name")
        obs = _env.reset(scenario_name=scenario)
        return {"observation": obs.model_dump()}

    @app.post("/step")
    async def _step(body: dict[str, Any]) -> dict[str, Any]:
        # Match OpenEnv wire format: {"action": {...}}
        action_data = body.get("action", body)
        try:
            parsed = Action(**action_data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        obs = _env.step(parsed)
        return {"observation": obs.model_dump()}

    @app.get("/state")
    async def _state() -> dict[str, Any]:
        # state is a @property on Environment (per openenv.core.Environment ABC)
        return {"state": _env.state.model_dump()}

    @app.get("/schema")
    async def _schema() -> dict[str, Any]:
        return {
            "action": Action.model_json_schema(),
            "observation": Observation.model_json_schema(),
        }


# ---------------------------------------------------------------------------
# Static dashboard at /web
# ---------------------------------------------------------------------------

_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.exists() and any(_static_dir.iterdir()):
    try:
        from fastapi.staticfiles import StaticFiles
        app.mount(
            "/web",
            StaticFiles(directory=str(_static_dir), html=True),
            name="web",
        )
    except Exception:
        pass  # static mount is nice-to-have, not mandatory


# ---------------------------------------------------------------------------
# Health + metadata endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "using_openenv": _USING_OPENENV,
        "max_concurrent_envs": MAX_CONCURRENT_ENVS,
    }


# ---------------------------------------------------------------------------
# CLI entry point — required by `openenv validate` for multi-mode deployment
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the server via `python -m server.app` or the `openenv-r2-kit` console script."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
