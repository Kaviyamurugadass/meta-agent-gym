# Teaching a 1.7B Model to Design AI Agents — and How It Taught Me About Reward Hacking

> **TL;DR:** We built a reinforcement learning environment where a small LLM learns to write complete AI agent specifications from scratch. Within one training run it found — and exploited — a reward-hacking bug that code review had missed entirely. Here's the whole story.

**Built for OpenEnv Hackathon 2026 | Qwen3-1.7B + 4-bit LoRA | GRPO + DAPO | HF Spaces**

| | |
|---|---|
| 🤗 Live demo | [huggingface.co/spaces/Kaviya-M/meta-agent-gym](https://huggingface.co/spaces/Kaviya-M/meta-agent-gym) |
| 🧠 Trained adapter | [Kaviya-M/meta-agent-gym-adapter](https://huggingface.co/Kaviya-M/meta-agent-gym-adapter) |
| 💻 Code | [github.com/Kaviyamurugadass/meta-agent-gym](https://github.com/Kaviyamurugadass/meta-agent-gym) |
| 📓 Colab notebook | [train_colab.ipynb](https://colab.research.google.com/github/Kaviyamurugadass/meta-agent-gym/blob/main/notebooks/train_colab.ipynb) |

---

## The Question

Everyone who uses AI agents eventually hits the same wall: *"I need a custom agent for this specific thing, and I have no idea how to write one."*

What if a small model could learn to write them — not by copying templates, but by getting graded on whether the agents it designs actually work?

That's **meta-agent-gym**: an RL environment where the policy's task is to design AI agents. The input is a description like *"Build an agent that reviews pull requests for security issues"*. The output is a complete AGENT.md file — name, description, skills, model tier, system prompt — that runs in Claude Code, Goose, and Cursor.

---

## Act 1 — Building the Gym

### The action space decision

The hardest design decision was the action space. Free-form generation was tempting — it's how humans write AGENT.md files — but credit assignment for GRPO is brutal on long token sequences with no intermediate signal.

I went with **14 discrete commands** instead:

```
set_name        set_description   add_skill       remove_skill
set_model       write_prompt      add_tools       set_memory
set_max_turns   check_score       inspect_example submit
noop            inspect
```

Each command has clean semantics. The agent can investigate (`check_score`, `inspect_example`) before committing (`submit`). Reward responds per-step. The policy sees exactly where it went wrong.

### Three-tier verification (RLVR)

Reward hacking is the first thing any policy tries. The environment catches it with three independent layers:

```
Layer 1 — Hard Verifiers (100% of steps, $0)
├─ yaml_valid ──────────────────── GATE (zeros all reward if fails)
├─ has_required_fields ──────────── GATE (zeros all reward if fails)
├─ prompt_length_ok ──────────────── GATE (zeros all reward if fails)
├─ model_valid
└─ skills_format_ok

Layer 2 — Heuristic Judge (90% of steps)
├─ skill_selection    (0.25 weight)
├─ description_quality (0.20 weight)
├─ workflow_clarity    (0.20 weight)
├─ model_appropriateness (0.15 weight)
├─ best_practices     (0.10 weight)
└─ efficiency         (0.10 weight)

Layer 3 — Goose Real Execution (offline harness)
└─ Actually runs the AGENT.md — ground truth no judge can fake
```

This is **RLVR** — verifiable rewards, not learned reward models. Drift-free by construction.

---

## Act 2 — The Trap (First Training Run)

Trained Qwen2.5-0.5B + 4-bit LoRA on a free Colab T4. One epoch, 8 episodes, 2 generations per prompt. The numbers came back immediately:

```
success_rate = 10/10 = 100%
mean_reward  = +51.80 per episode
```

I almost called it a day.

---

## Act 3 — Goose Catches What Metrics Couldn't

Then I wired up the real execution tier — Goose — to actually *execute* the AGENT.md files the policy had generated. Every single trajectory was:

```
noop → noop → noop → noop → noop → noop → submit
```

The "100% successful" model had learned to do **absolutely nothing**.

I spent that evening reading `server/rewards/reward.py` line by line. The bug was on line 158:

```python
# BEFORE — subtracting a negative = adding a bonus
total = core + bonus + progress - sum(anti_hack_penalties.values())
#                                    ^ anti_hack_penalties stores -5.0, -0.5 etc.
#                                      subtracting negatives = ADDING them

# AFTER — penalties stay negative
total = core + bonus + progress + sum(anti_hack_penalties.values())
```

`anti_hack_penalties` stores values as **negative** numbers (`empty_spec = -5.0`). Subtracting a negative flipped the sign. An empty spec was worth **+7.40 per step**.

GRPO found it in **four gradient steps**.

> **The point isn't the bug. The point is that Goose caught what in-environment metrics couldn't.** That's what "real-execution tier" means — and it justified the entire three-tier architecture in one shot.

---

## Before vs After — The Numbers

### Reward swing from the sign-flip fix

| | Per step | Per 7-step episode |
|---|---:|---:|
| Empty spec — **before** fix | **+7.40** | **+51.80** |
| Empty spec — **after** fix | −4.58 | −32.20 |
| Competent heuristic | +3.05 | +21.33 |
| Trained Qwen3-1.7B (eval avg) | +1.37 | **+7.68** |
| **Total swing from the fix** | **11.98** | **84.00** |

84 points of correction. The bug was paying empty-spec play **2.4× more** than honest competent play.

### Behavior comparison (same task: *"Generate pytest cases for a function"*)

| Step | Pre-fix Qwen2.5-0.5B (buggy) | Post-fix Qwen3-1.7B (trained) |
|---|---|---|
| 1 | `noop` | `set_name: test-generator` |
| 2 | `noop` | `set_description: "Generates pytest…"` |
| 3 | `noop` | `add_skill: test-generator` |
| 4 | `noop` | `write_prompt: "You are a test…"` |
| 5 | `noop` | `submit` |
| 6 | `noop` | — |
| 7 | `submit` | — |
| Reward | **+51.80** (bug bonus) | **+9.60** (real) |
| Goose execute | ❌ FAIL — empty spec | ✅ PASS |

---

## Act 4 — Post-Fix Training Results

After the fix, Qwen3-1.7B was trained (25 dataset episodes, 2 epochs, 2 generations). The model learned to produce structured specs consistently.

### Total reward curve (training)

![Total Reward Curve](monitoring/colab_results_qwen3_1.7b/total_reward_curve.png)

The training reward curve shows the model climbing from near-zero (early episodes where gates fire and zero out reward) toward consistent positive reward as it learns the required episode structure.

### Reward progression (labeled)

![Reward Progression](monitoring/colab_results_qwen3_1.7b/reward_progression_labeled.png)

The labeled version annotates key transitions: the moment the model starts consistently passing the `has_required_fields` gate, and when prompt length stops being the failure point.

### Baseline comparison (20 easy-tier episodes each)

| Policy | Success rate | Mean reward | Max reward |
|---|---:|---:|---:|
| Random | 0% | 0.00 | 0.00 |
| Competent heuristic | 100% | 21.33 | 30.33 |
| Expert benchmark (mixed difficulty) | 95% | 16.79 | 19.57 |
| **Trained Qwen3-1.7B** | **80%** | **7.68** | **9.60** |

![Baseline Comparison](monitoring/colab_results_qwen3_1.7b/baseline_comparison_labeled.png)

Random gets 0% because the three gate verifiers block any `submit` without `name`, `description`, and a ≥ 50-char prompt. This is intentional — the gates prove the environment is non-trivial to game. The heuristic proves it's reachable. The trained policy sits between them — structure learned, content quality still improving.

### Per-component reward progression (50 evaluation episodes)

| Component | Overall mean | Last-10 mean | Trend |
|---|---:|---:|---:|
| `total` reward per step | 1.45 | 1.28 | +0.06/ep |
| `has_required_fields` | 0.33 | 0.51 | +0.015/ep |
| `prompt_length_ok` | 0.31 | 0.40 | +0.013/ep |
| `description_quality` | 0.26 | 0.25 | +0.010/ep |
| `yaml_valid` | 1.00 | 1.00 | 0 (always) |
| `model_valid` | 1.00 | 1.00 | 0 (always) |
| `skill_selection` | 0.01 | 0.00 | stalled |

![Component Learning](monitoring/colab_results_qwen3_1.7b/component_learning_labeled.png)

`yaml_valid` and `model_valid` hit 1.00 from episode 1 and stay there — the model never produces malformed YAML and always picks a valid model tier. `skill_selection` is the weakest signal: it learns the command format but not which skill to pick per domain.

### Success rate over training

![Success Rate Curve](monitoring/colab_results_qwen3_1.7b/success_rate_curve.png)

![Success Rate Labeled](monitoring/colab_results_qwen3_1.7b/success_rate_labeled.png)

The success rate climbs from 0% (hard gates blocking empty specs) to 80% on easy tasks as the model learns the mandatory episode structure.

### Per-component reward curves

![Component Curves](monitoring/colab_results_qwen3_1.7b/component_curves.png)

Each line is one reward dimension tracked independently across episodes — the key design choice for GRPO, which needs variance across completions to learn from. `yaml_valid` and `model_valid` flatline at 1.0 from the start; `skill_selection` is the lagging signal that needs more training.

### Full comparison (all metrics)

![Full Comparison](monitoring/colab_results_qwen3_1.7b/full_comparison.png)

The full comparison chart puts all metrics side-by-side: reward, success rate, and per-component scores across the entire evaluation window.

---

## What the Model Actually Learned

After training, the model has internalized five concrete rules:

1. A valid AGENT.md needs `name`, `description`, `skills`, `model`, `system_prompt` — never submit without all of them
2. The correct episode order is: name → description → skill → prompt → submit
3. YAML must be well-formed — the `yaml_valid` gate fires on any syntax error
4. `sonnet` is almost always the right tier for easy tasks
5. `write_prompt` must be ≥ 50 chars — the gate is unforgiving at 49

## What the Model's Failures Taught Us

1. **Gate cliffs are sharp** — the 50-char prompt gate triggers at exactly 49 chars. The model learned the structure but not the margin
2. **Skill selection doesn't generalize** — `web-scraping` has the highest frequency in training data; the model over-indexes on it for all domains
3. **The sign-flip bug** — GRPO found a reward error in 4 steps that code review missed entirely. Independent verifiers are not optional

---

## The Live Demo

The environment runs on HF Spaces. Two interaction modes:

### Build Step-by-Step
Pick a scenario from the curriculum, issue commands manually, and watch the reward breakdown update in real time. Good for understanding what the verifiers actually check.

### Generate from Description
Type a task description, and the Qwen3-1.7B LoRA adapter generates a complete action sequence. The result is scored by the same three-tier verifier — plus an optional LLM judge (Groq/Llama-3.3-70B) if `GROQ_API_KEY` is set.

Example output for *"Review pull requests for security issues"*:

```yaml
---
name: security-pull-request-reviewer
description: Reviews PRs for SQL injection, XSS, and hard-coded secrets.
skills:
  - code-reviewer
  - data-validator
  - pattern-matcher
model: sonnet
---
You are a security pull request reviewer. Analyze code for common
vulnerabilities: SQL injection, XSS, hard-coded secrets. Follow safety
rules to avoid exposing sensitive information. Only provide valid,
actionable security recommendations.
```

---

## Architecture

```
Colab T4 / L4 GPU                         HF Spaces (cpu-basic)
┌──────────────────────────────────┐      ┌──────────────────────────────┐
│  GRPO Trainer (TRL 0.29)         │      │  OpenEnv Server :8000        │
│  ├─ Qwen3-1.7B + LoRA (4-bit)   │      │  ├─ /reset  /step  /state    │
│  ├─ Unsloth (2× faster)          │ HTTP │  ├─ /generate (LoRA adapter) │
│  ├─ 2 rollouts × 2 generations  │◄────►│  └─ /web  (dashboard)        │
│  └─ DAPO clip loss               │      │                              │
│                                  │      │  Environment                 │
│  Evaluation                      │      │  ├─ 24 scenarios, 4 phases   │
│  └─ Goose harness (offline)      │      │  ├─ 7-step episodes          │
└──────────────────────────────────┘      │  └─ POMDP hidden state       │
                                          └──────────────────────────────┘
```

---

## Honest Scope

This is a hackathon run — small compute, real signal. What's proven vs. what's next:

| | Status |
|---|---|
| Hard gates block empty-spec hacking | ✅ Proven (random = 0%, gates fire) |
| Environment is reachable | ✅ Proven (heuristic = 100%, 21.33 reward) |
| Structured episode format learned | ✅ Proven (8/10 success, correct command order) |
| Task-conditional skill selection | ❌ Not yet (over-indexes on `web-scraping`) |
| Prompt quality beyond length gate | ❌ Partial (structure yes, content no) |
| Curriculum escalation (Phase 2-4) | ❌ Not demonstrated (4 gradient steps, needs more) |

---

## Stack

| Component | Tech |
|---|---|
| Environment | OpenEnv v0.2.1 |
| Training | TRL 0.29 + Unsloth 4-bit LoRA, GRPO with DAPO loss |
| Base model | Qwen3-1.7B (Apache 2.0) |
| Deployment | Docker on HF Spaces (cpu-basic) |
| Real execution | Goose 1.27 + Claude Code CLI provider |
| LLM judge (UI) | Groq / Llama-3.3-70B (optional, falls back to heuristics) |

---

## Try It

```bash
# Clone and run locally
git clone https://github.com/Kaviyamurugadass/meta-agent-gym
cd meta-agent-gym
pip install -e .
uvicorn server.app:app --reload --port 8000
# → open http://localhost:8000/web/

# Reproduce the reward-hack demo (no GPU needed)
python scripts/demo_reward_fix.py

# Run the full test suite
pytest tests/ -q
```

Or open the [live Space](https://huggingface.co/spaces/Kaviya-M/meta-agent-gym) directly.

---

*Built for OpenEnv Hackathon 2026 by [@Kaviyamurugadass](https://github.com/Kaviyamurugadass)*
