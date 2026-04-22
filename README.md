---
title: OpenEnv R2 Kit
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - grpo
  - agentic-ai
  - hackathon
---

# openenv-r2-kit

> **TEMPLATE** — domain-agnostic OpenEnv starter. Fill the slots marked `# DOMAIN:` to specialize for a theme in hours instead of days.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/02_train_grpo.ipynb)

<!-- HEADLINE METRIC SLOT — fill with e.g. "Trained Qwen3-0.6B from 56% → 97.1% on <task>" -->
**Trained**: `<base-model>` → `<trained-model>` &nbsp; · &nbsp; **Metric**: `<X%>` → `<Y%>` over `<N>` episodes

---

## Hackathon Track Alignment

<!-- Direct pitch to judges. Map submission to Round 2 tracks. -->

| Track | How this env fits |
|---|---|
| *(fill on finale day based on Round 2 themes)* | — |

---

## Before / After Training Metrics

<!-- Fill after training runs complete. -->

| Metric | Baseline | Trained | Change |
|---|---:|---:|---:|
| Mean reward | — | — | — |
| Success rate | — | — | — |
| Avg steps to completion | — | — | — |
| Invalid-action rate | — | — | — |
| Rule violations / episode | — | — | — |

---

## Training Results

<!-- Fill each training run with its own image + commentary. Tell the
     iteration story (noisy → plateau → good variance), not just a final
     snapshot. Generate plots with `make plots`. -->

### Run 1 — *\<base-model\>*

![Run 1 reward curve](training/plots/run_1.png)

**Setup:** *(e.g. dataset_episodes=8, num_generations=4, LR=5e-6, DAPO loss)*
**Result:** mean reward `X.X` · best episode `Y.Y` · trend `+0.XX/ep`
**What we learned:** *(1-2 sentences on what the run taught us)*

### Run 2 — *\<base-model\>*

![Run 2 reward curve](training/plots/run_2.png)

**Setup:** *(what changed from Run 1)*
**Result:** …
**What we learned:** …

### Per-episode detail (latest run)

| Episode | Reward | Steps | Notes |
|--------:|-------:|------:|---|
| 1 | — | — | |
| 2 | — | — | |
| *(fill from `data/trained/index.json`)* | | | |

---

## Reward Formula

<!-- Judges weight this heavily. -->

**Mode:** multiplicative (default) — prevents fake wins. All dimensions must succeed.

```
R_total = 10 × correctness × efficiency × quality × safety
        + novelty_bonus
        + communication_bonus
        - regression_penalty × num_regressions
        - violation_penalty × num_soft_violations
```

Per-step decomposition exposed in `Observation.reward_breakdown`:

| Component | Weight | Rationale |
|---|---|---|
| correctness | 0.40 | Did the agent solve the task? |
| efficiency | 0.20 | Budget/step penalty |
| quality | 0.20 | Domain-specific quality (readability, robustness, coverage) |
| safety | 0.20 | Didn't break working state |
| regression_penalty | −0.15/regression | Explicit "don't break it" signal |
| soft_violation_penalty | −0.15/violation | Rule-engine enforcement |
| novelty_bonus | +0.10 | When zero soft violations |

Switch to additive mode via `RewardConfig(mode=RewardMode.ADDITIVE)`.

---

## Reward Justification

> **For judges:** every reward component has a defensible formula, rationale, and example. Nothing is vibe-based.

### Which reward mode and why?

Three modes are available. Pick the one that matches your domain:

| Mode | Formula | Strengths | Weaknesses | When to use |
|---|---|---|---|---|
| **ADDITIVE** | `Σ w_i × c_i` | Smooth gradient, partial credit | Gameable (high in one dim compensates for zero in another) | Bio-like domains where rule engine already blocks bad states |
| **MULTIPLICATIVE** | `10 × ∏ c_i^w_i` | All dims must succeed, un-gameable | Near-zero collapses gradient → weak GRPO signal | Binary-success / security-critical domains |
| **HYBRID** (recommended) | If any gate < threshold → `0`. Else → `10 × Σ w_i × c_i` | Strict on hard constraints, smooth everywhere else | Slightly more config (pick gate components) | Most domains |

**This template uses HYBRID as the recommended pattern.** Gate with multiplicative logic, score with additive — as one RL practitioner put it.

```python
# DOMAIN: pick mode in RewardConfig
RewardConfig(
    mode=RewardMode.HYBRID,
    gate_components=["safety"],   # any of these = 0 → reward = 0
    gate_threshold=0.01,
    component_weights={"correctness": 0.4, "efficiency": 0.2, ...},
)
```

### Component formulas

<!-- DOMAIN: fill this table on finale day. Each row = answer to "how did you measure X?" -->

| Component | Weight | Formula | Why this shape | Source |
|---|---|---|---|---|
| Correctness | 0.40 | *DOMAIN: e.g. `passed_checks / total_checks`* | Linear — each check is equally important | *e.g. PEP 8 §E501, RFC 1035, paper DOI* |
| Efficiency  | 0.20 | *DOMAIN: e.g. `max(0, 1 - step/max_steps)`* | Linear decay — domain has no $ budget | Step count is the scarce resource |
| Quality     | 0.20 | *DOMAIN: e.g. `(docstrings + type_hints) / 2`* | Composite of independent quality dims | Domain-specific sub-rubric |
| Safety      | 0.20 | *DOMAIN: e.g. `1 - regressions / previously_passing`* | Anti-regression | Explicit "don't break it" signal |

### Penalties

| Penalty | Weight | Formula | When it fires |
|---|---|---|---|
| Soft violation | −0.15 / violation | Count of soft rule violations in `rule_violations` | Redundant actions, low confidence, causal errors |
| Regression | −0.15 / regressed check | Count of checks passing before action but failing after | Any step that breaks a previously-working state |

### Bonuses

| Bonus | Condition | Value |
|---|---|---|
| Novelty | Zero soft violations this step | +0.10 |

### Example walkthrough

> *(Fill in after running `make bench` on finale day.)*
>
> On scenario `my_medium_1`, the expert trajectory completes in 6 steps with:
> - `correctness = 28/32 = 0.875`
> - `efficiency = max(0, 1 - 6/10) = 0.4`
> - `quality = 0.9`
> - `safety = 0.97`
> - **Core reward:** `10 × 0.875 × 0.4 × 0.9 × 0.97 ≈ 3.06`
> - **Novelty bonuses (6 clean steps × +0.1):** `+0.60`
> - **Total expert reward:** `≈ 3.66`
>
> Random policy on the same scenario averages `~0.35` → expert is **10.5× better**, which our reward-quality test enforces (`expert >= 3x random`).

### Reward-quality guardrails (automated)

- `tests/test_reward_quality.py` enforces `expert_reward >= 3 × random_reward` for every non-placeholder scenario
- `tests/test_observation_quality.py` enforces observations carry decision-relevant signal
- Both auto-skip while scenarios are `placeholder_*`, auto-activate when renamed on finale day

If a component's formula is broken, the reward-quality test catches it before it reaches training.

---

## Architecture

```
openenv-r2-kit/
├── models.py                # Pydantic Action/Observation/State schemas
├── client.py                # OpenEnv HTTP/WS client wrapper
├── inference.py             # LLM agent with triple-fallback JSON parser
├── server/
│   ├── app.py               # FastAPI — /reset /step /state /schema /ws
│   ├── environment.py       # reset/step/state lifecycle (OpenEnv spec)
│   ├── rules/engine.py      # Hard/soft violation framework
│   ├── rewards/reward.py    # Decomposed + multiplicative modes
│   └── tasks/
│       ├── generator.py     # Domain randomization
│       └── scenarios.py     # Scenario library (DOMAIN slot)
├── training/
│   ├── grpo_trl.py          # TRL GRPO (H100)
│   ├── grpo_unsloth.py      # Unsloth 4-bit LoRA (T4 Colab)
│   ├── trajectory.py        # Episode serialization
│   ├── rollout_collection.py
│   ├── evaluation.py        # Online + benchmark metrics
│   ├── benchmark.py         # Literature/expert benchmark (DOMAIN slot)
│   └── reward_backend.py    # Local vs remote abstraction
├── tests/                   # 100+ target
├── static/index.html        # Interactive dashboard
├── notebooks/               # 01 demo | 02 train | 03 evaluate
└── data/sample/             # Fixtures (DOMAIN slot)
```

---

## Quick Start

```bash
# Install
uv sync --extra dev

# Run env locally
uv run uvicorn server.app:app --reload --port 8000

# Interact via client
uv run python -c "from client import Env; e = Env('http://localhost:8000'); print(e.reset())"

# Validate OpenEnv spec
uv run python -m openenv.cli validate

# Run tests
uv run pytest tests/ -q
```

## Training

**T4 (free Colab) — Unsloth 4-bit LoRA:**

```bash
uv sync --extra train --extra unsloth
uv run python training/grpo_unsloth.py \
    --model-id Qwen/Qwen3-0.6B \
    --per-device-train-batch-size 1 \
    --num-generations 2 \
    --gradient-accumulation-steps 4 \
    --max-seq-length 1024 \
    --output-dir training/grpo-unsloth-qwen3-0.6b
```

**H100 / A100 — TRL full GRPO:**

```bash
uv sync --extra train
uv run python training/grpo_trl.py --model-id Qwen/Qwen3.5-4B
```

### ⚠️ Gotchas (Day-1 killers)

1. Set `max_concurrent_envs=4+` in `server/app.py` or parallel GRPO crashes
2. T4 doesn't support FP8 — use BF16 LoRA only
3. Disable vLLM fast-inference when using Unsloth GRPO path
4. Pin `openenv-core==0.2.1` — still experimental, API may shift
5. Check TRL issue #4543 before relying on multi-step GRPO

See `TEMPLATE_PLAN.md` (private repo) for full gotcha table with source links.

---

## Deployment

```bash
# HF Spaces push
openenv push --repo-id <user>/<env-name>

# Docker build + run
docker build -t openenv-r2-kit .
docker run -p 8000:8000 openenv-r2-kit
```

---

## Filling the Template

**See [`FINALE_GUIDE.md`](FINALE_GUIDE.md) for the full time-estimated checklist.** That file is the one to reference DURING the finale — step-by-step fill sequence, what to KEEP (🔵) vs REMOVE (🔴), emergency pivots, time budget.

Quick overview of fill-in points:

1. **`models.py`** — add domain-specific `ActionCommand` enum values, fill `TaskSpec.expected_findings` schema
2. **`server/tasks/scenarios.py`** — write 3–4 scenarios (easy/medium/hard) grounded in real citations
3. **`server/rules/engine.py`** — add domain rules (hard + soft violations)
4. **`server/rewards/reward.py`** — tune component weights, document rationale
5. **`server/environment.py`** — domain-specific state transitions
6. **`training/benchmark.py`** — expert/literature benchmark trajectory
7. **`README.md`** — fill Headline, Track Alignment, Before/After tables
8. **`static/index.html`** — swap domain widgets (optional)

Everything else (OpenEnv scaffold, GRPO training, trajectory utils, reward framework, rule engine, client, inference, tests) is pre-built — **don't touch**.
