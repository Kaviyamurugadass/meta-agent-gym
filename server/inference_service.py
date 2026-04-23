"""Trained-model inference service — powers the /generate endpoint.

Loads a LoRA adapter produced by `training/grpo_unsloth.py` and generates
agent specs (AGENT.md JSON) from natural-language task descriptions.

Design goals:
  1. Lazy-load on first call (avoid cold-start cost at server boot)
  2. Thread-safe (FastAPI may call concurrently)
  3. Clear structured errors when the adapter or deps are missing — the
     server keeps working without the trained model, the /generate endpoint
     just returns a helpful status payload explaining what's needed.

Onsite workflow (2026-04-25/26):
  1. Train/push adapter to `training/grpo-unsloth-output/` (or set
     META_ADAPTER_PATH env var)
  2. Ensure `transformers`, `peft`, `torch` are installed
  3. Restart Space → /generate works automatically
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional


DEFAULT_ADAPTER_PATH = Path(os.getenv("META_ADAPTER_PATH", "training/grpo-unsloth-output"))
DEFAULT_BASE_MODEL = os.getenv("META_BASE_MODEL", "Qwen/Qwen2.5-0.5B")

# Qwen2.5 uses ChatML; unsloth 4-bit variants sometimes ship without the template.
_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

_VALID_SKILLS = [
    "web-scraping", "http-client", "html-parser", "json-parser",
    "csv-handler", "data-transformer", "data-validator", "data-aggregator",
    "code-reviewer", "code-fixer", "test-generator", "file-reader",
    "file-writer", "log-analyzer", "pattern-matcher", "report-generator",
    "notifier",
]

_SYSTEM_PROMPT = f"""You are an agent designer. Given a task description, output a JSON object describing a complete agent.

Fields (all required):
- name: lowercase-hyphenated identifier (e.g. "price-scraper")
- description: one-line "when to use" description
- skills: list of 1-3 skills, each from this set: {", ".join(_VALID_SKILLS)}
- model: one of "haiku", "sonnet", "opus"
- system_prompt: 2-4 sentence instructions with clear workflow and safety rules

Respond with ONLY the JSON object. No prose, no markdown fences."""


class InferenceService:
    """Singleton wrapper around a LoRA-fine-tuned causal LM."""

    def __init__(
        self,
        adapter_path: Path = DEFAULT_ADAPTER_PATH,
        base_model: str = DEFAULT_BASE_MODEL,
    ) -> None:
        self.adapter_path = Path(adapter_path)
        self.base_model = base_model
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------ status
    @property
    def adapter_available(self) -> bool:
        return (self.adapter_path / "adapter_config.json").exists()

    @property
    def deps_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import peft  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def status(self) -> dict[str, Any]:
        return {
            "adapter_path": str(self.adapter_path),
            "adapter_available": self.adapter_available,
            "deps_available": self.deps_available,
            "loaded": self._model is not None,
            "load_error": self._load_error,
            "base_model": self.base_model,
        }

    # ---------------------------------------------------------------- loading
    def _load(self) -> None:
        """Load base model + apply LoRA adapter. Raises on failure."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if getattr(tokenizer, "chat_template", None) is None:
            tokenizer.chat_template = _CHATML_TEMPLATE
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float32,  # cpu-basic → fp32 is fine, bf16 not supported on some CPUs
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, str(self.adapter_path))
        model.eval()

        self._tokenizer = tokenizer
        self._model = model

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.deps_available:
                raise RuntimeError(
                    "Inference deps not installed. Run "
                    "`pip install transformers peft torch` in the Space environment."
                )
            if not self.adapter_available:
                raise RuntimeError(
                    f"No trained LoRA adapter at {self.adapter_path}. "
                    "Train onsite with HF credits and push the adapter to that path."
                )
            try:
                self._load()
                self._load_error = None
            except Exception as e:
                self._load_error = f"{type(e).__name__}: {e}"
                raise

    # --------------------------------------------------------------- inference
    def generate_spec(self, task_description: str) -> dict[str, Any]:
        """Run the LoRA-tuned model on a task; return a parsed spec dict."""
        self._ensure_loaded()
        import torch

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {task_description}"},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        response = self._tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        return _extract_spec(response)


# Module-level singleton (lazy)
_service: Optional[InferenceService] = None


def get_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
    return _service


# --------------------------------------------------------------------- helpers


def spec_to_actions(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn a generated spec dict into a replayable action sequence.

    The UI replays these against the running env via the existing WS /step
    pipeline — same reward surface as human-driven play.
    """
    actions: list[dict[str, Any]] = []

    if name := spec.get("name"):
        actions.append({"command": "set_name", "args": {"name": str(name)}})

    if desc := spec.get("description"):
        actions.append({"command": "set_description", "args": {"description": str(desc)}})

    for skill in (spec.get("skills") or []):
        if skill in _VALID_SKILLS:
            actions.append({"command": "add_skill", "args": {"skill": str(skill)}})

    if model := spec.get("model"):
        if str(model).lower() in {"haiku", "sonnet", "opus", "inherit"}:
            actions.append({"command": "set_model", "args": {"model": str(model).lower()}})

    if prompt := spec.get("system_prompt"):
        actions.append({"command": "write_prompt", "args": {"prompt": str(prompt)}})

    actions.append({"command": "submit", "args": {}})
    return actions


def _extract_spec(raw: str) -> dict[str, Any]:
    """Parse JSON from a model response, tolerant of surrounding prose/fences."""
    s = raw.strip()
    # Strip common markdown fences
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s[3:] else s[3:]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip().rstrip("`").strip()

    # Direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Extract first balanced {...} block
    start = s.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not parse JSON from model output. Raw response: {raw[:500]!r}")
