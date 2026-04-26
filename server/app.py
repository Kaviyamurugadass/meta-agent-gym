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
    GET  /health    — liveness check (status: "healthy")
    GET  /metadata  — environment metadata (name, description)
    POST /mcp       — JSON-RPC 2.0 MCP compliance endpoint
    GET  /web       — static dashboard (if static/index.html exists)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastapi.responses import RedirectResponse
from models import Action, Observation
from server.environment import Environment

# ---------------------------------------------------------------------------
# Try to use the official OpenEnv create_app helper. Fall back to a minimal
# FastAPI wrapper if openenv-core isn't available (e.g. during local dev).
# ---------------------------------------------------------------------------

MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "4"))
# Set to "0" to disable background warm-up (e.g. CPU-constrained dev machines)
WARMUP_MODEL = os.getenv("WARMUP_MODEL", "1") != "0"


def _background_warmup() -> None:
    """Kick off model loading in a thread so the event loop stays free."""
    try:
        from server.inference_service import get_service
        svc = get_service()
        if svc.adapter_available and svc.deps_available:
            import logging
            logging.getLogger(__name__).info("Starting background model warm-up…")
            svc._ensure_loaded()
            logging.getLogger(__name__).info("Model warm-up complete.")
    except Exception:
        pass  # warm-up is best-effort; real errors surface on /generate


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


# Attach background warm-up after app is created (works with both code paths)
@app.on_event("startup")
async def _startup_warmup() -> None:
    if WARMUP_MODEL:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _background_warmup)
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
        from models import State
        return {
            "action": Action.model_json_schema(),
            "observation": Observation.model_json_schema(),
            "state": State.model_json_schema(),
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


@app.get("/")
async def root():
    return RedirectResponse(url="/web/")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "using_openenv": _USING_OPENENV,
        "max_concurrent_envs": MAX_CONCURRENT_ENVS,
    }


@app.get("/metadata")
async def metadata() -> dict[str, Any]:
    return {
        "name": "meta-agent-gym",
        "description": (
            "RL environment that trains a policy to generate AGENT.md files "
            "from task descriptions using GRPO + RLVR."
        ),
        "version": "1.0.0",
        "type": "space",
        "runtime": "fastapi",
    }


@app.post("/mcp")
async def mcp_endpoint(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Minimal JSON-RPC 2.0 endpoint for OpenEnv MCP compliance."""
    req = body or {}
    method = req.get("method", "")
    req_id = req.get("id", 1)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "meta-agent-gym", "version": "1.0.0"},
            },
        }

    # Default: acknowledge with empty result (satisfies the jsonrpc==2.0 check)
    return {"jsonrpc": "2.0", "id": req_id, "result": {}}


# ---------------------------------------------------------------------------
# Trained-model inference endpoint — powers the dashboard's "Generate with
# Trained Model" button. Works once a LoRA adapter is pushed to
# training/grpo-unsloth-output/ (done onsite with HF credits).
# ---------------------------------------------------------------------------


@app.get("/generate/status")
async def generate_status() -> dict[str, Any]:
    """Report whether the trained model is deployed and ready to serve."""
    try:
        from server.inference_service import get_service
        return {"available": True, **get_service().status}
    except Exception as e:
        return {"available": False, "error": f"{type(e).__name__}: {e}"}


@app.post("/generate")
async def generate_with_trained_model(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate an agent spec from a task description using the trained LoRA.

    Request body:
        {"task_description": "Build an agent that scrapes product prices"}

    Response shapes:
        {"status": "ok", "spec": {...}, "actions": [{"command": "...", "args": {...}}, ...]}
        {"status": "no_adapter", "message": "...", "adapter_path": "..."}
        {"status": "deps_missing", "message": "..."}
        {"status": "error", "message": "..."}

    The `actions` list is designed for replay via the existing /step pipeline
    so the UI can animate a trained-model-driven episode using the same reward
    surface as human-driven play.
    """
    task = (body or {}).get("task_description", "").strip()
    if not task:
        return {"status": "error", "message": "task_description is required"}

    try:
        from server.inference_service import get_service, spec_to_actions
    except ImportError as e:
        return {"status": "deps_missing", "message": f"Failed to import inference module: {e}"}

    svc = get_service()
    if not svc.deps_available:
        return {
            "status": "deps_missing",
            "message": (
                "Inference deps not installed on this Space. "
                "Need: transformers, peft, torch. Onsite, add these to the runtime."
            ),
        }
    if not svc.adapter_available:
        return {
            "status": "no_adapter",
            "message": (
                "No trained LoRA adapter deployed yet. Available once we train "
                "onsite with HF credits (2026-04-25/26) and push the adapter to "
                "training/grpo-unsloth-output/ or set META_ADAPTER_PATH."
            ),
            "adapter_path": svc.status["adapter_path"],
        }

    try:
        # Run the blocking model inference in a thread pool so the event loop
        # (and WebSocket heartbeats) stay alive during the 30-60s load time.
        loop = asyncio.get_event_loop()
        spec = await loop.run_in_executor(None, svc.generate_spec, task)
        actions = spec_to_actions(spec)

        # LLM judge — runs async-safely in thread pool, falls back to
        # heuristics if GROQ_API_KEY is not set.
        try:
            from server.judge import judge_spec
            judge_result = await loop.run_in_executor(None, judge_spec, task, spec)
            judge_payload = judge_result.to_dict()
        except Exception as je:
            judge_payload = {"scores": {}, "reasoning": "", "provider": "unavailable",
                             "error": str(je)}

        return {"status": "ok", "spec": spec, "actions": actions, "judge": judge_payload}
    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}


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
