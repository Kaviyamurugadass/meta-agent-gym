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

## Submission links

| Resource | Link |
|---|---|
| 🤗 HF Space (live environment) | https://huggingface.co/spaces/Kaviya-M/meta-agent-gym |
| 📓 Colab training notebook | https://colab.research.google.com/github/Kaviyamurugadass/meta-agent-gym/blob/main/notebooks/train_colab.ipynb |
| 💻 GitHub repository | https://github.com/Kaviyamurugadass/meta-agent-gym |
| 📝 Blog post | [`docs/competition/HUGGINGFACE_BLOG.md`](docs/competition/HUGGINGFACE_BLOG.md) |

---

## The problem I started with

I'm a software developer. I use Cursor every day. I use Claude Code every day. Both of them are **agents under the hood** — not a single LLM call but a system: a system prompt, a tool list, a model choice, a memory scope. When the agent is well-designed, it solves my task in one shot. When it isn't, I'm debugging the agent before I can debug my code.

So I started asking: **are we creating these agents correctly?**

Anthropic's [Agent Skills Open Standard](https://skills.sh) gives us the *schema* — name, description, skills, model, system prompt — but the *content* is still a craft. What skills should this agent have? Which model tier? How do you write a system prompt that actually scopes the work? Most of us answer those questions by hand, by guesswork, and by iteration. **A better-designed agent solves tasks better. A worse-designed one wastes tokens and produces noise.**

That's the meta-skill. And it's the one I wanted to see if RL could teach a tiny model.

This OpenEnv environment grades AGENT.md specifications across multiple independent reward dimensions. A small language model issues structured commands (`set_name`, `add_skill`, `write_prompt`, `submit`...) to assemble a spec, GRPO updates the policy on the resulting reward, and the third verification tier (Goose) actually runs the AGENT.md to make sure the model isn't gaming the score.

---

## The story

### Act 1 — Building the gym

The hardest design call was the **action space**. Free-form text generation was tempting (it's how humans write `AGENT.md` by hand), but credit assignment for GRPO is brutal on long token sequences.

I went with **14 discrete commands** instead — `set_name`, `set_description`, `add_skill`, `remove_skill`, `set_model`, `write_prompt`, `add_tools`, `set_memory`, `set_max_turns`, `check_score`, `inspect_example`, `submit`, `noop`, `inspect`. Each command has clean semantics, the reward responds per-step, and the agent can investigate (`check_score`) before committing (`submit`).

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

## Problem statements addressed

### Primary: Theme #5 — Wild Card

Meta-agent design doesn't fit cleanly into the other themes — it's not multi-agent (one agent), not long-horizon (7-step episodes), not professional tool-use (no browser/API ecosystem the agent operates inside), and not personalized. So I'm claiming Wild Card for what it was designed for: an out-of-box submission that meaningfully adds value to LLM training on a task that hasn't been explored before.

The capability gap I'm targeting: every developer using Cursor, Claude Code, or similar agent frameworks is implicitly designing agents — choosing skills, model tiers, system prompts. That's a meta-skill. Nobody is training small LLMs to do it. This is the environment that makes it trainable.

- **Underexplored RL target:** AGENT.md generation as a craft, framed as a multi-step decision problem with verifiable rewards
- **Real artifact output:** the generated spec runs in Claude Code, Goose, Copilot, and anything else following the [Agent Skills Open Standard](https://skills.sh)
- **Methodological contribution:** three-tier RLVR caught a real reward-hack bug on the first training run we pointed Goose at (see *The story → Act 3* above)

### Secondary alignment: Theme #3 — World Modeling (broadly)

Some structural parts of the env genuinely fit Theme #3:

- **POMDP structure:** the hidden state is *"what makes a good spec for this task?"* — the agent observes per-component reward + violations and infers ground truth across multiple steps
- **Investigation tools:** `check_score` and `inspect_example` let the agent gather information before committing — classic POMDP belief-update pattern
- **Persistent state across the 7-step episode:** each command updates the spec dict; the agent has to reason about what's already there before adding more

Note: this is the broader Theme #3 fit, **not** the #3.1 Professional Tasks sub-theme — that sub-theme's examples are direct tool/API ecosystems (browsers, enterprise apps), which our agent doesn't operate inside.

### Architectural ambition: Theme #4 — Self-Improvement (built, not yet demonstrated at scale)

The closed-loop self-improvement design — adversarial task generator + adaptive curriculum — is wired in code ([`server/adversarial.py`](server/adversarial.py), [`training/curriculum.py`](training/curriculum.py)) but the shipped Colab run was 4 gradient steps, far below what's needed to demonstrate curriculum escalation or recursive skill amplification.

I'm flagging this as honest future direction, not a current claim. Specific post-hackathon paths I'd attempt next:

- **VCRL** (Variance-based Curriculum RL) — adds a sampling weight on top of GRPO based on per-group reward variance. Lightweight, GRPO-native. Reported +18 points on Qwen3-4B baseline. ([arxiv 2509.19803](https://arxiv.org/html/2509.19803v1))
- **SEC** (Self-Evolving Curriculum) — replaces hardcoded phase progression with an adaptive curriculum policy learned alongside RL. ([arxiv 2505.14970](https://arxiv.org/pdf/2505.14970))

There is one *genuine* form of self-improvement that already happened: the bug-hunt was real co-evolution between agent and environment design. The agent found the empty-spec reward hack; I fixed the reward function. That's not the recursive skill amplification Theme #4 describes (which is about agents improving themselves), but it is one valid form of agent/environment co-evolution.

### Underlying technique: RLVR — Verifiable Rewards

The reward design uses **verifiable rewards instead of a learned reward model**, throughout:

- **5 hard verifiers** run on every step (yaml_valid, has_required_fields, prompt_length_ok, model_valid, skills_format_ok). Three of them act as **gates** that zero the entire step reward if they fail.
- **6 judge dimensions** for what hard checks can't capture: skill_selection, description_quality, workflow_clarity, model_appropriateness, best_practices, efficiency
- **Goose offline harness** at [`evaluation/goose_execution.py`](evaluation/goose_execution.py) — independent execution as the third tier, ground truth no judge can fake. Captured 3/3 PASS run committed at [`evaluation/fixtures/smoke_output.txt`](evaluation/fixtures/smoke_output.txt).
- **Caught a real bug.** The Goose tier exposed the sign-flip in the anti-hack penalty math that the in-environment metrics couldn't see. Without it, we'd have shipped a 100%-success policy producing nothing executable. That's exactly why RLVR with independent verifiers matters.

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
- **[`monitoring/colab_results/report.json`](monitoring/colab_results/report.json) — 50 evaluation episodes.** Uses the heuristic policy as a placeholder for the trained adapter at inference time. The rows describe the environment's reward structure under heuristic play, not the adapter's inference behaviour.
- **[`monitoring/colab_results_qwen3_1.7b/`](monitoring/colab_results_qwen3_1.7b/) + [`data/colab_trained_qwen3_1.7b/`](data/colab_trained_qwen3_1.7b/) — Qwen3-1.7B run with real adapter inference (2026-04-25).** 25 dataset episodes, 2 epochs, 2 generations. 10 trained-policy evaluation rollouts: **8/10 success, mean reward 7.68**. The two failures (`an_easy_001`) write a 49-char prompt and miss the 50-char gate; the model also picks `web-scraping` for every task regardless of `required_skills`. Structure learned, content-conditioning on the task did not — expected at this episode budget.

Other things to be straight about:

- **The Goose harness covers 3 Phase 1 (single-skill) tasks.** Expanding to Phase 2-4 multi-skill tasks is future work; the harness API in `evaluation/goose_execution.py` accepts new tasks without changes.
- **The trained adapter on disk is the post-fix Qwen3-1.7B run (2026-04-25)** — 25 dataset episodes, 2 epochs, 2 generations. 8/10 success, mean reward 7.68. The model learned the correct structure but does not yet condition on `required_skills` — it picks `web-scraping` for most tasks regardless of domain. The Qwen2.5-0.5B pre-fix adapter (which exploited the sign-flip bug and emitted empty specs) was the earlier run documented in `data/colab_trained/`.
- **The "self-improvement" track** (adversarial task generation, curriculum auto-escalation) exists in code (`server/adversarial.py`, `training/curriculum.py`) but hasn't been demonstrated end-to-end. The Colab run was 4 gradient steps — far below what's needed to see curriculum escalate. Specific future-work paths I'd attempt next: VCRL ([arxiv 2509.19803](https://arxiv.org/html/2509.19803v1)) and Self-Evolving Curriculum ([arxiv 2505.14970](https://arxiv.org/pdf/2505.14970)).

---

## Quick start

### Try the live demo

[**huggingface.co/spaces/Kaviya-M/meta-agent-gym**](https://huggingface.co/spaces/Kaviya-M/meta-agent-gym)

The dashboard has two tabs:
- **🛠 Build Step-by-Step** — pick a scenario, issue commands manually, watch the multi-component reward update live
- **✨ Generate from Description** — type a task, the Qwen3-1.7B post-fix adapter emits commands (8/10 success rate; if HF CPU cannot load the model, a heuristic fallback runs instead)

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
│   ├── competition/              # HUGGINGFACE_BLOG.md, TRAINING_EVIDENCE.md
│   └── onsite/                   # ONSITE_TRAINING_PLAN.md
├── openenv.yaml
└── Dockerfile
```

---

## Submission materials

| What | Where |
|---|---|
| Mini-blog | [`docs/competition/HUGGINGFACE_BLOG.md`](docs/competition/HUGGINGFACE_BLOG.md) |
| Training evidence + honest limitations | [`docs/competition/TRAINING_EVIDENCE.md`](docs/competition/TRAINING_EVIDENCE.md) |
| Onsite training plan | [`docs/onsite/ONSITE_TRAINING_PLAN.md`](docs/onsite/ONSITE_TRAINING_PLAN.md) |
| Sign-flip fix evidence | [`data/post_fix/REWARD_FIX_COMPARISON.md`](data/post_fix/REWARD_FIX_COMPARISON.md) |
| Real plots + report.json | [`monitoring/colab_results/`](monitoring/colab_results/) |
| Qwen3-1.7B run (real adapter inference) | [`monitoring/colab_results_qwen3_1.7b/`](monitoring/colab_results_qwen3_1.7b/), [`data/colab_trained_qwen3_1.7b/`](data/colab_trained_qwen3_1.7b/) |

---

## Stack

| Component | Tech |
|---|---|
| Environment | OpenEnv v0.2.1 (gymnasium-compatible) |
| Training | TRL 0.29 + Unsloth 4-bit LoRA, GRPO with DAPO loss |
| Judge (production) | Claude Sonnet, 6-dimension scoring |
| Real execution | Goose 1.27 + Claude Code CLI provider |
| Deployment | Docker on HF Spaces (cpu-basic) |
| Currently shipped model | Qwen3-1.7B (Apache 2.0, dual-mode reasoning) |
| Baseline run | Qwen2.5-0.5B (Apache 2.0, 4 gradient steps, sign-flip bug present) |

---

Built for the OpenEnv Hackathon 2026 by [@Kaviyamurugadass](https://github.com/Kaviyamurugadass).
