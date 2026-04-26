"""LLM judge for evaluating generated agent specs.

Called only from the /generate endpoint (NOT during GRPO training).
Uses Groq's free API (Llama-3.3-70B) by default — set GROQ_API_KEY in
HF Space secrets.  Falls back to fast heuristics if the key is absent
or the API call fails, so the endpoint always returns a score.

Environment variables:
    GROQ_API_KEY      — Groq API key (get free at console.groq.com)
    JUDGE_MODEL       — override model, default "llama-3.3-70b-versatile"
    JUDGE_PROVIDER    — "groq" (default) or "disabled" to force heuristics
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("server.judge")

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_DEFAULT_MODEL = "llama-3.3-70b-versatile"

_JUDGE_SYSTEM = """You are an expert AI agent designer reviewing agent specifications.
Score the following agent spec on each dimension from 0.0 to 1.0.
Respond ONLY with a JSON object — no prose, no markdown fences."""

_JUDGE_USER_TEMPLATE = """Task description: {task}

Agent spec:
{spec_yaml}

Score each dimension 0.0–1.0 and give a one-line reason per dimension.

Respond with exactly this JSON shape:
{{
  "skill_selection": <0.0-1.0>,
  "description_quality": <0.0-1.0>,
  "workflow_clarity": <0.0-1.0>,
  "model_appropriateness": <0.0-1.0>,
  "best_practices": <0.0-1.0>,
  "overall": <0.0-1.0>,
  "reasoning": "<one sentence summary>"
}}

Scoring guide:
- skill_selection: Are the chosen skills appropriate and sufficient for this task?
- description_quality: Is the description clear, specific, and useful as a delegation hint?
- workflow_clarity: Does the system_prompt give a clear, step-by-step workflow?
- model_appropriateness: Is the model tier (haiku/sonnet/opus) right for the task complexity?
- best_practices: Does the spec mention error handling, validation, or safety rules?
- overall: Holistic quality — would a real agent using this spec succeed at the task?"""


@dataclass
class JudgeResult:
    scores: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    provider: str = "heuristic"
    error: str | None = None

    @property
    def overall(self) -> float:
        return self.scores.get("overall", 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "reasoning": self.reasoning,
            "provider": self.provider,
            **({"error": self.error} if self.error else {}),
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def judge_spec(task_description: str, spec: dict[str, Any]) -> JudgeResult:
    """Score *spec* against *task_description*.

    Tries Groq first; silently falls back to heuristics on any failure.
    """
    provider = os.getenv("JUDGE_PROVIDER", "groq").lower()
    api_key = os.getenv("GROQ_API_KEY", "")

    if provider != "disabled" and api_key:
        try:
            return _groq_judge(task_description, spec, api_key)
        except Exception as exc:
            logger.warning("LLM judge failed, falling back to heuristics: %s", exc)
            result = _heuristic_judge(task_description, spec)
            result.error = f"LLM judge failed ({type(exc).__name__}); heuristics used"
            return result

    return _heuristic_judge(task_description, spec)


# ---------------------------------------------------------------------------
# Groq backend
# ---------------------------------------------------------------------------

def _spec_to_yaml_str(spec: dict[str, Any]) -> str:
    lines = ["---"]
    for k, v in spec.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        elif k == "system_prompt":
            lines.append(f"{k}: |")
            for line in str(v).splitlines():
                lines.append(f"  {line}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def _groq_judge(task: str, spec: dict[str, Any], api_key: str) -> JudgeResult:
    import httpx

    model = os.getenv("JUDGE_MODEL", _DEFAULT_MODEL)
    user_msg = _JUDGE_USER_TEMPLATE.format(
        task=task,
        spec_yaml=_spec_to_yaml_str(spec),
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(_GROQ_URL, json=payload, headers=headers)
        resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"].strip()
    return _parse_judge_response(content, provider=f"groq/{model}")


def _parse_judge_response(content: str, provider: str) -> JudgeResult:
    # Strip markdown fences if present
    s = re.sub(r"^```[a-z]*\n?", "", content.strip())
    s = re.sub(r"\n?```$", "", s).strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        # Try extracting first {...} block
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON found in judge response: {content[:200]!r}")
        data = json.loads(m.group())

    score_keys = ["skill_selection", "description_quality", "workflow_clarity",
                  "model_appropriateness", "best_practices", "overall"]
    scores = {k: float(data.get(k, 0.5)) for k in score_keys}
    # Clamp to [0, 1]
    scores = {k: max(0.0, min(1.0, v)) for k, v in scores.items()}
    reasoning = str(data.get("reasoning", ""))
    return JudgeResult(scores=scores, reasoning=reasoning, provider=provider)


# ---------------------------------------------------------------------------
# Heuristic fallback (same logic as reward.py — no API needed)
# ---------------------------------------------------------------------------

def _heuristic_judge(task: str, spec: dict[str, Any]) -> JudgeResult:
    desc = spec.get("description", "")
    prompt = spec.get("system_prompt", "")
    skills = spec.get("skills", [])
    model = spec.get("model", "sonnet")

    # skill_selection — at least one skill, penalise empty or >5
    n_skills = len(skills)
    skill_score = 1.0 if 1 <= n_skills <= 3 else (0.7 if n_skills <= 5 else 0.3)

    # description_quality — delegation words + length
    deleg_words = ["use", "when", "handles", "specialist", "expert", "proactively"]
    has_deleg = any(w in desc.lower() for w in deleg_words)
    good_len = 20 <= len(desc.split()) <= 100
    desc_score = min(1.0, 0.4 * has_deleg + 0.4 * good_len + 0.2 * (len(desc) > 0))

    # workflow_clarity — step indicators in system_prompt
    step_pats = ["1.", "2.", "step", "first", "then", "finally", "workflow"]
    has_steps = sum(1 for p in step_pats if p in prompt.lower())
    workflow_score = min(1.0, has_steps / 3.0)

    # model_appropriateness — easy→haiku, rest→sonnet, expert→opus
    task_lower = task.lower()
    if any(w in task_lower for w in ["complex", "expert", "advanced", "enterprise"]):
        ideal = "opus"
    elif any(w in task_lower for w in ["simple", "easy", "basic", "quick"]):
        ideal = "haiku"
    else:
        ideal = "sonnet"
    model_score = 1.0 if model == ideal else (0.8 if model == "sonnet" else 0.5)

    # best_practices — safety / error handling keywords
    practice_kws = ["error", "validate", "check", "ensure", "safely", "gracefully", "safe"]
    matches = sum(1 for kw in practice_kws if kw in prompt.lower())
    practices_score = min(1.0, matches / 3.0)

    overall = (skill_score * 0.25 + desc_score * 0.20 + workflow_score * 0.20
               + model_score * 0.15 + practices_score * 0.20)

    return JudgeResult(
        scores={
            "skill_selection": round(skill_score, 2),
            "description_quality": round(desc_score, 2),
            "workflow_clarity": round(workflow_score, 2),
            "model_appropriateness": round(model_score, 2),
            "best_practices": round(practices_score, 2),
            "overall": round(overall, 2),
        },
        reasoning="Heuristic scoring (set GROQ_API_KEY for LLM judge)",
        provider="heuristic",
    )
