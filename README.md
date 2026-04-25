---
title: meta-agent-gym
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - grpo
  - agent-design
  - rlvr
---

# meta-agent-gym

> Can a 0.5B language model learn to design AI agents — by getting graded on the ones it builds?

An OpenEnv environment where a small language model issues structured commands (`set_name`, `add_skill`, `write_prompt`, `submit`...) to assemble an `AGENT.md` spec, and gets scored on it across multiple independent reward dimensions. GRPO closes the loop.

I built this for the OpenEnv Hackathon 2026 to explore the meta-skill that developers using Cursor and Claude rely on every day: deciding *what an agent should be* before you train it.

**Live demo:** [huggingface.co/spaces/Kaviya-M/meta-agent-gym](https://huggingface.co/spaces/Kaviya-M/meta-agent-gym)

---

## The story

### Act 1 — Building the gym

The hardest design call was the **action space**. Free-form text generation was tempting (it's how humans write `AGENT.md` by hand), but credit assignment for GRPO is brutal on long token sequences.

I went with **12 discrete commands** instead — `set_name`, `set_description`, `add_skill`, `remove_skill`, `set_model`, `write_prompt`, `add_tools`, `set_memory`, `set_max_turns`, `check_score`, `inspect_example`, `submit`. Each command has clean semantics, the reward responds per-step, and the agent can investigate (`check_score`) before committing (`submit`).

The reward function is multi-component on purpose. **Six judge dimensions** (skill_selection, description_quality, workflow_clarity, model_appropriateness, best_practices, efficiency) plus **five hard verifiers** (yaml_valid, has_required_fields, prompt_length_ok, model_valid, skills_format_ok). Three of those hard verifiers act as **gates** — fail any one and the entire step reward zeroes. Defense in depth, in case the judge components get gamed.

### Act 2 — The first training run

Trained Qwen2.5-0.5B + 4-bit LoRA on Colab T4. One epoch, 8 episodes, 2 generations per prompt. Sentinel-verified — `training/grpo-unsloth-output/training_summary.json` contains `"real_training": true` (only written after `trainer.train()` returns, so it can't be faked).

The numbers came back gorgeous:

```
success_rate = 10/10 = 100%
mean_reward = +51.80 per episode
```

I almost called it a day.

### Act 3 — Plugging in Goose

Then I wired up the third verification tier — Goose, the open-source agent runtime — to actually *execute* the AGENT.md files the trained model was producing. I'd been planning this all along; I just hadn't pointed it at the trained adapter yet.

Every single trajectory was empty. Pure `noop → noop → ... → submit`. No `set_name`. No `add_skill`. No `write_prompt`. The "100% successful" model had learned to do nothing.

I spent that evening reading `server/rewards/reward.py` line by line. The bug was on line 158:

```python
total = core + bonus + progress - penalty - regression - sum(anti_hack_penalties.values())
```

The `anti_hack_penalties` values are stored as **negative** numbers in config (`anti_hack_empty_spec = -5.0`, deliberately). Subtracting a negative flips the sign — a -5 *penalty* became a +5 *bonus*. Empty specs scored +7.4 per step. GRPO obediently exploited it within four gradient steps.

One operator. One line.

```diff
- total = ... - sum(anti_hack_penalties.values())
+ total = ... + sum(anti_hack_penalties.values())
```

Plus a parameterised regression test in `tests/test_meta_agent_reward.py` that fails immediately if empty-spec ever receives positive reward, under either HYBRID or ADDITIVE mode. So this can't resurface silently.

**The point isn't the bug. The point is that Goose caught what the in-environment metrics couldn't.** That's what "real-execution tier" means in the three-tier RLVR design — and it justified the whole architecture in one shot.

---

## How it works

```
Task Bank (24 scenarios)  ─►  Environment (OpenEnv)  ─►  Agent (Qwen + LoRA)
                                       │                          │
                                       │     issues commands       │
                                       │                           ▼
                                       │              ┌── set_name ────────┐
                                       │              │   set_description   │
                                       │              │   add_skill         │
                                       │              │   write_prompt      │
                                       │              │   submit            │
                                       │              └─────────┬───────────┘
                                       ▼                        │
                            Three-Tier Verification  ◄──────────┘
                                       │
                            ┌──────────┼──────────┐
                            ▼          ▼          ▼
                       Hard verifiers  Judge   Goose runtime
                       (5; 3 are gates)  (6 dim)  (offline harness)
                            │          │          │
                            └──────────┼──────────┘
                                       ▼
                       Multi-component reward → GRPO advantage
```

### Three-tier verification

| Layer | Cost | What it catches |
|---|---|---|
| Hard verifiers (5) | $0 | YAML structure, required fields, prompt length, model name, skills format |
| Judge (6 dimensions) | ~$0.01 / step (Claude Sonnet in prod, heuristic in dev) | Skill selection, description quality, workflow clarity, model fit, best practices, efficiency |
| Goose (offline) | ~$0 (Claude Code CLI provider) | Whether the AGENT.md actually executes |

The first three hard verifiers (`yaml_valid`, `has_required_fields`, `prompt_length_ok`) act as **gates** — failure on any one zeroes the entire step reward. The other two hard verifiers + six judge dimensions feed the gradient signal.

### Anti-hack penalties

| Penalty | Value | When |
|---|---:|---|
| empty_spec | −5.0 | Prompt < 50 chars or required fields missing |
| over_engineered | −0.5 | More than 10 skills, or `opus` when `sonnet` suffices |
| repetitive | −0.3 | Same action twice consecutively |
| regression | −0.15 | Breaking a previously-passing check |

---

## What's shipped

### Environment
- 24 scenarios across 4 difficulty phases (10 easy, 5 medium, 5 hard, 4 expert)
- Phase 1 has **5 developer-focused tasks** (PR security review, pytest generation, debugging, refactoring, code review for error handling) plus 5 web/data/files tasks
- Live HF Space (Docker, OpenEnv v0.2.1)
- Two-tab dashboard UI: **Build Step-by-Step** (manual play, sees reward signal live) and **Generate from Description** (textarea → trained model)

### Training pipeline
- TRL 0.29 + Unsloth 4-bit LoRA, GRPO with DAPO loss
- Colab notebook: [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb)
- CLI: `python training/grpo_unsloth.py --model-id Qwen/Qwen2.5-0.5B`
- Sentinel-verified Colab T4 run on Qwen2.5-0.5B (8.8M of 502M params trainable, 4 gradient steps, `real_training: true`)

### Verification tier
- Hard verifiers in [`server/verifiers.py`](server/verifiers.py)
- Judge component scoring in [`server/rewards/reward.py`](server/rewards/reward.py) — uses Claude Sonnet in production, heuristic approximations in dev
- Goose harness in [`evaluation/goose_execution.py`](evaluation/goose_execution.py) — adapter from `AgentSpec` to Goose recipe, subprocess runner via Claude Code CLI, grader against expected substrings
- 3 deterministic test fixtures + a captured passing run at [`evaluation/fixtures/smoke_output.txt`](evaluation/fixtures/smoke_output.txt) (3/3 PASS, ~52 s wall time)

### The bug fix
- One-operator fix on `server/rewards/reward.py:158`
- 5 regression tests in [`tests/test_meta_agent_reward.py`](tests/test_meta_agent_reward.py) — `pytest -k empty_spec` runs them in ~1.5 seconds
- Before / after rollouts in [`data/post_fix/`](data/post_fix/) — 5 episodes × 4 policy/mode combinations
- Live demo: `python scripts/demo_reward_fix.py` prints the +7.4 → -4.58 → 0.00 reward swing for an empty spec

---

## Results

### Baselines (20 easy-tier episodes each)

| Policy | Success | Mean reward | Max reward |
|---|---:|---:|---:|
| Random | 0% | 0.00 | 0.00 |
| Competent heuristic | 100% | 21.33 | 30.33 |
| Expert benchmark (mixed difficulty) | 20/21 | 16.79 | 19.57 |

Random gets 0% because the hard-verifier gate blocks any submit without `name`, `description`, and a ≥50-char prompt. The competent heuristic proves the environment is *reachable* — the RL pre-condition for learning. Expert is the mixed-difficulty ceiling.

### Per-component reward signal (50 evaluation episodes)

![Component Curves](monitoring/colab_results/component_curves.png)

| Component | Overall mean | Last-10 mean | Δ |
|---|---:|---:|---:|
| Per-step reward `total` | 1.83 | 3.05 | +67% |
| `description_quality` | 0.31 | 0.51 | +65% |
| `workflow_clarity` | 0.23 | 0.38 | +67% |
| `has_required_fields` | 0.34 | 0.57 | +67% |
| `prompt_length_ok` | 0.34 | 0.57 | +67% |

Episode-level aggregate trend: **+0.62 reward per episode**.

### The bug, in numbers

Same empty-spec-submitting policy, before vs after the sign-flip fix:

| | Per step | Per 7-step episode |
|---|---:|---:|
| Before fix (ADDITIVE mode) | +7.40 | +51.80 |
| After fix (ADDITIVE mode) | −4.58 | −32.20 |
| **Swing** | **11.98** | **84.00** |

84 points of correction. On a reward surface where competent heuristic scores 21.33 per episode, the bug had been paying empty-spec play more than honest play.

---

## Honest scope

There are two distinct rollout sets in this repo and they describe different things:

- **[`data/colab_trained/`](data/colab_trained/) — 10 trajectories.** The actual trained LoRA's output during the Colab run. These showed the empty-spec collapse and triggered the Goose investigation.
- **[`monitoring/colab_results/report.json`](monitoring/colab_results/report.json) — 50 evaluation episodes.** Uses the heuristic policy as a placeholder for the trained adapter at inference time. The rows describe the environment's reward structure under heuristic play, not the adapter's inference behaviour. Wiring adapter inference into evaluation rollout collection is the planned first onsite task (2026-04-25/26).

Other things to be straight about:

- **The Goose harness covers 3 Phase 1 (single-skill) tasks.** Expanding to Phase 2-4 multi-skill tasks is future work; the harness API in `evaluation/goose_execution.py` accepts new tasks without changes.
- **The trained adapter on disk is the pre-fix one** — it was trained when the sign-flip bug was still present, so it learned to emit empty specs. The "Generate" tab in the dashboard reproduces this if you click it; I left it enabled deliberately so you can see the bug live.
- **The "self-improvement" track** (adversarial task generation, curriculum auto-escalation) exists in code (`server/adversarial.py`, `training/curriculum.py`) but hasn't been demonstrated end-to-end. The Colab run was 4 gradient steps — far below what's needed to see curriculum escalate. Specific future-work paths I'd attempt next: VCRL ([arxiv 2509.19803](https://arxiv.org/html/2509.19803v1)) and Self-Evolving Curriculum ([arxiv 2505.14970](https://arxiv.org/pdf/2505.14970)).

---

## Quick start

### Try the live demo

[**huggingface.co/spaces/Kaviya-M/meta-agent-gym**](https://huggingface.co/spaces/Kaviya-M/meta-agent-gym)

The dashboard has two tabs:
- **🛠 Build Step-by-Step** — pick a scenario, issue commands manually, watch the multi-component reward update live
- **✨ Generate from Description** — type a task, the trained model emits commands (currently the pre-fix adapter, so output is empty by design — see the bug live)

### Run locally

```bash
# Install
uv sync

# Start the env
uvicorn server.app:app --reload --port 8000

# Open http://127.0.0.1:8000/web/

# Run the test suite
pytest tests/ -q

# Reproduce the reward bug fix demo (instant, no GPU)
python scripts/demo_reward_fix.py

# Run the Goose harness (needs goose CLI + Claude Code installed)
python -m evaluation.goose_execution --smoke
```

### Train

```bash
# T4 / Colab free tier — the path that produced the shipped run
python training/grpo_unsloth.py --model-id Qwen/Qwen2.5-0.5B

# L4 / A100 — onsite scale-up target with HF credits, Apr 25-26
python training/grpo_unsloth.py \
    --model-id Qwen/Qwen3-1.7B \
    --per-device-train-batch-size 2 \
    --num-generations 4 \
    --gradient-accumulation-steps 4 \
    --max-seq-length 2048 \
    --num-epochs 3
```

Or use the Colab notebook end-to-end: [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb).

---

## Project layout

```
meta-agent-gym/
├── models.py                     # AgentSpec, Action, Observation, RewardConfig
├── client.py                     # OpenEnv HTTP/WebSocket client
├── inference.py                  # LLM episode runner
├── server/
│   ├── app.py                    # FastAPI + WebSocket (OpenEnv compatible)
│   ├── environment.py            # reset → step → verify → reward
│   ├── verifiers.py              # 5 hard verifiers
│   ├── rewards/
│   │   ├── reward.py             # multi-component reward (sign-flip fix on line 158)
│   │   └── rubric_reward.py      # OpenEnv Rubric implementations
│   ├── rules/engine.py           # rule validation engine
│   ├── runtime/goose.py          # Goose runtime stub (training-loop integration is future work)
│   ├── tasks/scenarios.py        # 24 curriculum scenarios
│   └── inference_service.py      # /generate endpoint backend
├── evaluation/
│   ├── goose_execution.py        # working Goose harness (offline real-execution tier)
│   └── fixtures/                 # 3 deterministic tasks + reference AGENT.md + smoke_output.txt
├── training/
│   ├── grpo_unsloth.py           # 4-bit LoRA GRPO (T4 + L4/A100)
│   ├── grpo_trl.py               # full GRPO (H100/A100)
│   ├── rollout_collection.py     # rollout capture (with --reward-mode CLI flag)
│   └── grpo-unsloth-output/      # trained adapter + training_summary.json sentinel
├── tests/                        # 35 tests, all passing
├── data/
│   ├── baseline/                 # random + heuristic baseline trajectories
│   ├── colab_trained/            # 10 trajectories from the Colab run (the empty-spec collapse)
│   └── post_fix/                 # before/after rollouts under fixed reward
├── monitoring/colab_results/     # plots + report.json
├── notebooks/train_colab.ipynb   # end-to-end Colab notebook
├── scripts/demo_reward_fix.py    # live before/after reward demo
├── static/index.html             # 2-tab dashboard
├── docs/
│   ├── competition/              # PITCH.md, HUGGINGFACE_BLOG.md, TRAINING_EVIDENCE.md
│   └── onsite/                   # ONSITE_TRAINING_PLAN.md
├── openenv.yaml
└── Dockerfile
```

---

## Submission materials

| What | Where |
|---|---|
| Mini-blog | [`docs/competition/HUGGINGFACE_BLOG.md`](docs/competition/HUGGINGFACE_BLOG.md) |
| 3-min pitch script | [`docs/competition/PITCH.md`](docs/competition/PITCH.md) |
| Training evidence + honest limitations | [`docs/competition/TRAINING_EVIDENCE.md`](docs/competition/TRAINING_EVIDENCE.md) |
| Onsite training plan | [`docs/onsite/ONSITE_TRAINING_PLAN.md`](docs/onsite/ONSITE_TRAINING_PLAN.md) |
| Sign-flip fix evidence | [`data/post_fix/REWARD_FIX_COMPARISON.md`](data/post_fix/REWARD_FIX_COMPARISON.md) |
| Real plots + report.json | [`monitoring/colab_results/`](monitoring/colab_results/) |

---

## Stack

| Component | Tech |
|---|---|
| Environment | OpenEnv v0.2.1 (gymnasium-compatible) |
| Training | TRL 0.29 + Unsloth 4-bit LoRA, GRPO with DAPO loss |
| Judge (production) | Claude Sonnet, 6-dimension scoring |
| Real execution | Goose 1.27 + Claude Code CLI provider |
| Deployment | Docker on HF Spaces (cpu-basic) |
| Currently shipped model | Qwen2.5-0.5B (Apache 2.0) |
| Onsite scale-up target | Qwen3-1.7B (Apache 2.0, dual-mode reasoning, agent-ready) |

---

Built for the OpenEnv Hackathon 2026 by [@Kaviyamurugadass](https://github.com/Kaviyamurugadass).
