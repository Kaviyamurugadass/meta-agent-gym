"""Inference-service tests that avoid loading model weights."""

from __future__ import annotations

import json

from server.inference_service import (
    InferenceService,
    _extract_spec,
    _strip_thinking_blocks,
    fallback_spec,
    is_memory_load_error,
)


def test_inference_service_uses_training_summary_base_model(tmp_path, monkeypatch):
    monkeypatch.delenv("META_BASE_MODEL", raising=False)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "training_summary.json").write_text(
        json.dumps({"model": "Qwen/Qwen3-1.7B"}),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/qwen3-1.7b-unsloth-bnb-4bit"}),
        encoding="utf-8",
    )

    svc = InferenceService(adapter_path=adapter_dir)

    assert svc.base_model == "Qwen/Qwen3-1.7B"
    assert svc.base_model_source == "training_summary.json"


def test_inference_service_normalises_unsloth_adapter_base(tmp_path, monkeypatch):
    monkeypatch.delenv("META_BASE_MODEL", raising=False)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/qwen3-1.7b-unsloth-bnb-4bit"}),
        encoding="utf-8",
    )

    svc = InferenceService(adapter_path=adapter_dir)

    assert svc.base_model == "Qwen/Qwen3-1.7B"
    assert svc.base_model_source == "adapter_config.json"
    assert svc.inference_dtype == "auto"


def test_memory_load_error_detection():
    exc = OSError("The paging file is too small for this operation to complete. (os error 1455)")

    assert is_memory_load_error(exc)


def test_strip_thinking_blocks_complete():
    raw = "<think>\nsome reasoning here\n</think>\n{\"name\": \"foo\"}"
    assert _strip_thinking_blocks(raw) == '{"name": "foo"}'


def test_strip_thinking_blocks_dangling():
    raw = "<think>\ncut off by max_new_tokens"
    assert _strip_thinking_blocks(raw) == ""


def test_extract_spec_ignores_think_preamble():
    raw = (
        "<think>\nLet me design an agent.\n</think>\n"
        '{"name":"pr-reviewer","description":"Reviews PRs",'
        '"skills":["code-reviewer"],"model":"sonnet","system_prompt":"You review PRs."}'
    )
    spec = _extract_spec(raw)
    assert spec["name"] == "pr-reviewer"
    assert spec["skills"] == ["code-reviewer"]


def test_fallback_spec_produces_replayable_fields():
    spec = fallback_spec(
        "Review a pull request for security issues like SQL injection, XSS, and secrets."
    )

    assert spec["name"]
    assert "code-reviewer" in spec["skills"]
    assert "pattern-matcher" in spec["skills"]
    assert spec["model"] == "sonnet"
    assert len(spec["system_prompt"]) >= 50
