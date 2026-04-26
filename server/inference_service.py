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
import re
import threading
from pathlib import Path
from typing import Any, Optional


DEFAULT_ADAPTER_PATH = Path(os.getenv("META_ADAPTER_PATH", "training/grpo-unsloth-output"))
DEFAULT_ADAPTER_HF_ID = os.getenv("META_ADAPTER_HF_ID", "Kaviya-M/meta-agent-gym-adapter")
DEFAULT_BASE_MODEL = "unsloth/qwen3-1.7b-unsloth-bnb-4bit"
DEFAULT_INFERENCE_DTYPE = os.getenv("META_INFERENCE_DTYPE", "auto")
DEFAULT_ALLOW_HEURISTIC_FALLBACK = os.getenv("META_ALLOW_HEURISTIC_FALLBACK", "1") != "0"

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
        base_model: Optional[str] = None,
    ) -> None:
        self.adapter_path = Path(adapter_path)
        self.base_model, self.base_model_source = _resolve_base_model(self.adapter_path, base_model)
        self.inference_dtype = DEFAULT_INFERENCE_DTYPE
        self.allow_heuristic_fallback = DEFAULT_ALLOW_HEURISTIC_FALLBACK
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------ status
    @property
    def adapter_available(self) -> bool:
        # Local adapter takes priority; fall back to HF Hub ID
        if (self.adapter_path / "adapter_config.json").exists():
            return True
        if DEFAULT_ADAPTER_HF_ID:
            try:
                from huggingface_hub import file_exists
                return file_exists(DEFAULT_ADAPTER_HF_ID, "adapter_config.json")
            except Exception:
                return True  # assume available; real error surfaces in _load
        return False

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
            "base_model_source": self.base_model_source,
            "inference_dtype": self.inference_dtype,
            "allow_heuristic_fallback": self.allow_heuristic_fallback,
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
            torch_dtype=_resolve_torch_dtype(torch, self.inference_dtype),
            low_cpu_mem_usage=True,
        )
        # Use local adapter if it exists, otherwise load from HF Hub
        adapter_source = (
            str(self.adapter_path)
            if (self.adapter_path / "adapter_config.json").exists()
            else DEFAULT_ADAPTER_HF_ID
        )
        import logging as _log
        _log.getLogger(__name__).info("Loading adapter from: %s", adapter_source)
        model = PeftModel.from_pretrained(base, adapter_source)
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
            # /no_think disables Qwen3 extended-thinking mode so the model
            # outputs JSON directly without a <think>...</think> preamble.
            {"role": "user", "content": f"/no_think\nTask: {task_description}"},
        ]
        # Try enable_thinking=False (supported by Qwen3 tokenizer templates).
        # Fall back to the plain call for older tokenizer versions.
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=1024,  # needs room to finish <think> block + JSON
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        response = self._tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        try:
            return _extract_spec(response)
        except ValueError:
            # Model finished thinking but produced no parseable JSON — use
            # a deterministic heuristic spec so the UI never gets an error.
            import logging
            logging.getLogger(__name__).warning(
                "JSON parse failed; using heuristic fallback. response[:200]=%r",
                response[:200],
            )
            return _fallback_spec(task_description)


# Module-level singleton (lazy)
_service: Optional[InferenceService] = None


def get_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
    return _service


# --------------------------------------------------------------------- helpers


def _resolve_base_model(adapter_path: Path, explicit_base_model: Optional[str]) -> tuple[str, str]:
    """Choose the base model that matches the adapter on disk.

    Local demos often swap adapters without setting META_BASE_MODEL. Prefer an
    explicit env/argument, then the training sentinel, then PEFT metadata.
    """
    env_model = os.getenv("META_BASE_MODEL")
    if explicit_base_model:
        return explicit_base_model, "argument"
    if env_model:
        return env_model, "META_BASE_MODEL"

    summary_path = adapter_path / "training_summary.json"
    try:
        if summary_path.exists():
            model = json.loads(summary_path.read_text(encoding="utf-8")).get("model")
            if isinstance(model, str) and model.strip():
                return model.strip(), "training_summary.json"
    except Exception:
        pass

    config_path = adapter_path / "adapter_config.json"
    try:
        if config_path.exists():
            raw_model = json.loads(config_path.read_text(encoding="utf-8")).get(
                "base_model_name_or_path"
            )
            if isinstance(raw_model, str) and raw_model.strip():
                return _normalise_adapter_base_model(raw_model.strip()), "adapter_config.json"
    except Exception:
        pass

    return DEFAULT_BASE_MODEL, "default"


def _normalise_adapter_base_model(model_name: str) -> str:
    """Convert common Unsloth 4-bit adapter bases to plain HF model IDs."""
    lower = model_name.lower()
    if lower.startswith("unsloth/"):
        match = re.search(r"qwen(?P<major>3|2(?:\.5)?)-(?P<size>[0-9.]+b)", lower)
        if match:
            major = match.group("major")
            size = match.group("size").upper()
            return f"Qwen/Qwen{major}-{size}"
    return model_name


def _resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    """Resolve an env-friendly dtype name for Transformers loading."""
    dtype = dtype_name.strip().lower()
    if dtype in {"", "auto"}:
        return "auto"
    if dtype in {"fp32", "float32"}:
        return torch_module.float32
    if dtype in {"fp16", "float16"}:
        return torch_module.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    raise ValueError(
        f"Unsupported META_INFERENCE_DTYPE={dtype_name!r}. "
        "Use one of: auto, float32, float16, bfloat16."
    )


def is_memory_load_error(exc: Exception) -> bool:
    """Detect local OOM/pagefile failures from model loading."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "paging file is too small",
            "os error 1455",
            "out of memory",
            "cannot allocate memory",
        )
    )


def fallback_spec(task_description: str) -> dict[str, Any]:
    """Small deterministic local fallback when the model cannot fit in memory."""
    text = task_description.lower()
    skills: list[str] = []

    keyword_skills = [
        (("scrape", "web", "html", "page", "site"), ["web-scraping", "html-parser"]),
        (("api", "http", "request"), ["http-client"]),
        (("csv", "spreadsheet"), ["csv-handler"]),
        (("json",), ["json-parser"]),
        (("data", "pipeline", "transform", "clean"), ["data-transformer"]),
        (("validate", "schema"), ["data-validator"]),
        (("aggregate", "summary", "statistics", "count"), ["data-aggregator"]),
        (("review", "pull request", "security", "sql injection", "xss", "secret"), ["code-reviewer", "pattern-matcher"]),
        (("fix", "bug", "refactor"), ["code-fixer"]),
        (("test", "pytest", "unit"), ["test-generator"]),
        (("file", "read"), ["file-reader"]),
        (("write", "save"), ["file-writer"]),
        (("log", "debug", "trace", "error"), ["log-analyzer"]),
        (("report", "dashboard"), ["report-generator"]),
        (("alert", "notify"), ["notifier"]),
    ]

    for keywords, candidates in keyword_skills:
        if any(keyword in text for keyword in keywords):
            for skill in candidates:
                if skill not in skills:
                    skills.append(skill)

    if not skills:
        skills = ["code-reviewer"]
    skills = skills[:3]

    name_words = re.findall(r"[a-z0-9]+", text)[:4] or ["generated", "agent"]
    name = "-".join(name_words)
    if len(name) < 3:
        name = "generated-agent"

    return {
        "name": name,
        "description": f"Use this agent to handle: {task_description[:120]}",
        "skills": skills,
        "model": "sonnet",
        "system_prompt": (
            f"You are a focused specialist for this task: {task_description}. "
            "First inspect the available context, then identify the relevant inputs, risks, "
            "and expected output. Use the selected skills to produce a concise, correct result, "
            "and call out assumptions or safety concerns before finalizing."
        ),
    }


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


def _strip_thinking_blocks(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks before parsing."""
    import re as _re
    # Remove complete <think>...</think> blocks (greedy-safe with DOTALL)
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)
    # Also strip a dangling <think> block with no closing tag
    # (model was cut off mid-think by max_new_tokens)
    think_start = text.find("<think>")
    if think_start >= 0:
        text = text[:think_start]
    return text.strip()


def _extract_spec(raw: str) -> dict[str, Any]:
    """Parse JSON from a model response, tolerant of surrounding prose/fences."""
    s = _strip_thinking_blocks(raw).strip()
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
