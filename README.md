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

### Can a 1.7B model learn to design AI agents — by getting graded on the ones it builds?

We gave a tiny language model a task description, a 14-command action space, and a three-tier verifier that doesn't lie. No few-shot examples. No chain-of-thought scaffolding. Just `set_name`, `add_skill`, `write_prompt`, `submit` — and a reward signal that zeroes out if the spec can't pass a YAML parser.

Within one training run, the model learned to produce structured AGENT.md files that run in Claude Code, Goose, and Cursor. Along the way, it found a reward-hacking bug we hadn't caught in code review — by exploiting it perfectly within four gradient steps.

**This is meta-agent-gym** — a reinforcement learning environment where a policy learns to design AI agents, graded by a three-tier verifier: hard checks, an LLM judge, and Goose real execution.

> Built for OpenEnv Hackathon 2026 | Trained: Qwen3-1.7B + 4-bit LoRA | GRPO + DAPO loss | Deployed on HF Spaces

| Resource | Link |
|---|---|
| 🤗 HF Space (live environment) | https://huggingface.co/spaces/Kaviya-M/meta-agent-gym |
| 📓 Colab training notebook | https://colab.research.google.com/github/Kaviyamurugadass/meta-agent-gym/blob/main/notebooks/train_colab.ipynb |
| 💻 GitHub repository | https://github.com/Kaviyamurugadass/meta-agent-gym |
| 📝 Blog post | [`docs/competition/HUGGINGFACE_BLOG.md`](docs/competition/HUGGINGFACE_BLOG.md) |

---

## The Story: From Reward-Hacking to Real Execution

### Act 1 — Building the Gym

The hardest design decision was the **action space**. Free-form AGENT.md generation was tempting — it's how humans write them by hand — but credit assignment for GRPO is brutal on long token sequences with no intermediate signal.

I went with **14 discrete commands** instead:

```
set_name        set_description   add_skill      remove_skill
set_model       write_prompt      add_tools      set_memory
set_max_turns   check_score       inspect_example  submit
noop            inspect
```

Each command has clean semantics. The agent investigates (`check_score`, `inspect_example`) before committing (`submit`). Reward responds per-step. The policy sees exactly where it went wrong.

The reward function uses **six judge dimensions + five hard verifiers**. Three of those verifiers act as gates — fail any one and the entire step reward zeroes. Defense in depth, in case the judge gets gamed.

### Act 2 — The First Training Run (The Trap)

Trained Qwen2.5-0.5B + 4-bit LoRA on Colab T4. One epoch, 8 episodes, 2 generations per prompt. The numbers came back:

```
success_rate = 10/10 = 100%
mean_reward  = +51.80 per episode
```

I almost called it a day.

### Act 3 — Goose Catches What Metrics Couldn't

Then I wired up the third verification tier — Goose — to actually *execute* the AGENT.md files. Every single trajectory was:

```
noop → noop → noop → noop → noop → noop → submit
```

The "100% successful" model had learned to do absolutely nothing.

I spent that evening reading `server/rewards/reward.py` line by line. The bug was on line 158:

```python
# Before — subtracting a negative = adding a bonus
total = core + bonus + progress - penalty - regression - sum(anti_hack_penalties.values())

# After — penalties stay negative
total = core + bonus + progress - penalty - regression + sum(anti_hack_penalties.values())
```

`anti_hack_penalties` stores values as **negative** numbers (e.g., `empty_spec = -5.0`). Subtracting a negative flipped the sign. An empty spec was worth **+7.4 per step**. GRPO found it in four gradient steps.

**The point isn't the bug. The point is that Goose caught what the in-environment metrics couldn't.** That's what "real-execution tier" means — and it justified the entire three-tier architecture in one shot.

### Act 4 — Post-Fix: Structure Learned, Conditioning Partial

After the fix, Qwen3-1.7B was trained (25 dataset episodes, 2 epochs, 2 generations). The model learned to produce structured specs:

```
set_name → set_description → add_skill → write_prompt → submit
```

**8/10 evaluation episodes succeeded (mean reward 7.68).** The two failures wrote 49-char prompts — one char below the 50-char gate — and the model still defaults to `web-scraping` regardless of domain. Structure learned; content-conditioning on the task is the next training target.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        META-AGENT TRAINING LOOP                         │
│                                                                         │
│  ┌──────────────┐   task desc   ┌───────────────┐   commands  ┌──────┐ │
│  │  Task Bank   │──────────────►│  Qwen3-1.7B   │────────────►│ Env  │ │
│  │ (24 scenarios│               │  + LoRA       │             │ Step │ │
│  │  4 phases)   │               └───────────────┘             └──┬───┘ │
│  └──────────────┘                                                │      │
│                                                                  │      │
│  ┌───────────────────────────────────────────────────────────────┼────┐ │
│  │                   THREE-TIER VERIFICATION                     ▼    │ │
│  │                                                                    │ │
│  │  Layer 1 — Hard Verifiers (100% of steps, $0)                      │ │
│  │  ├─ yaml_valid ────────────────────── GATE (zeros all reward)      │ │
│  │  ├─ has_required_fields ──────────── GATE (zeros all reward)       │ │
│  │  ├─ prompt_length_ok ─────────────── GATE (zeros all reward)       │ │
│  │  ├─ model_valid                                                    │ │
│  │  └─ skills_format_ok                                               │ │
│  │                                                                    │ │
│  │  Layer 2 — LLM Judge (90% of steps, ~$0.01)                        │ │
│  │  ├─ skill_selection (0.25)   ├─ description_quality (0.20)         │ │
│  │  ├─ workflow_clarity (0.20)  ├─ model_appropriateness (0.15)       │ │
│  │  └─ best_practices (0.10)   └─ efficiency (0.10)                  │ │
│  │                                                                    │ │
│  │  Layer 3 — Goose Real Execution (offline harness)                  │ │
│  │  └─ Runs the AGENT.md — ground truth no judge can fake             │ │
│  └─────────────────────────────────────────────────────────────────┬──┘ │
│                                                                     │    │
│                          reward signal                              │    │
│                               ▼                                     │    │
│                    ┌────────────────────┐                           │    │
│                    │  GRPO Advantage     │◄──────────────────────────┘    │
│                    │  (DAPO loss, 4-bit  │                                │
│                    │   LoRA, Unsloth)    │                                │
│                    └────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Architecture

```
Colab T4 / L4 GPU                         HF Spaces (cpu-basic)
┌──────────────────────────────────┐      ┌──────────────────────────────┐
│                                  │      │                              │
│  GRPO Trainer (TRL 0.29)         │      │  OpenEnv Server :8000        │
│  ├─ Qwen3-1.7B + LoRA (4-bit)   │      │  ├─ /reset   /step   /state  │
│  ├─ Unsloth (2× faster)          │ HTTP │  ├─ /schema  /health         │
│  ├─ 2 rollouts × 2 generations  │◄────►│  ├─ /generate (LoRA adapter) │
│  └─ DAPO clip loss               │      │  └─ /web  (dashboard UI)     │
│                                  │      │                              │
│  Evaluation                      │      │  Environment                 │
│  ├─ 10 rollouts post-training    │      │  ├─ 24 scenarios, 4 phases   │
│  └─ Goose harness (offline)      │      │  ├─ 7-step episodes          │
│                                  │      │  └─ POMDP: hidden state =    │
└──────────────────────────────────┘      │    "good spec for this task" │
                                          │                              │
                                          │  Verification               │
                                          │  ├─ 5 hard verifiers         │
                                          │  ├─ 6-dim LLM judge          │
                                          │  └─ Goose CLI executor       │
                                          └──────────────────────────────┘
```

---

## Before vs After Training

### Behavior comparison (same task: "Generate pytest cases for a function")

| | Pre-fix Qwen2.5-0.5B (buggy) | Post-fix Qwen3-1.7B (trained) |
|---|---|---|
| Step 1 | `noop` | `set_name: test-generator` |
| Step 2 | `noop` | `set_description: "Generates pytest..."` |
| Step 3 | `noop` | `add_skill: test-generator` |
| Step 4 | `noop` | `write_prompt: "You are a..."` |
| Step 5 | `noop` | `submit` |
| Step 6 | `noop` | — |
| Step 7 | `submit` | — |
| Reward | **+51.80** (bug) | **+9.60** (real) |
| Goose execute | FAIL — empty spec | PASS |

### Reward before vs after the sign-flip fix

| | Per step | Per 7-step episode |
|---|---:|---:|
| Empty spec — **before** fix | **+7.40** | **+51.80** |
| Empty spec — **after** fix | −4.58 | −32.20 |
| Competent heuristic | +3.05 | +21.33 |
| Trained Qwen3-1.7B (eval avg) | +1.37 | **+7.68** |
| **Swing from the fix** | **11.98** | **84.00** |

84 points of correction. The bug was paying empty-spec play **2.4× more** than honest competent play.

![Reward Progression](monitoring/colab_results_qwen3_1.7b/reward_progression_labeled.png)

---

## Training Runs

### Run 1: Qwen2.5-0.5B — The Trap (pre-fix)

4 gradient steps, 8 episodes, Colab T4.

| Episode | Actions | Reward | Notes |
|---|---|---:|---|
| 1–8 | noop × 6 → submit | +51.80 | Empty-spec exploit discovered in ep 1 |

**Mean: +51.80 | Success: 10/10 (all empty specs)**

The training sentinel (`training_summary.json`) records `real_training: true` — so the run happened. The model just learned the wrong thing.

### Run 2: Qwen3-1.7B — Post-Fix (shipped)

25 dataset episodes, 2 epochs, 2 generations, Colab T4. Evaluation over 10 rollouts:

| Episode | Task | Actions | Reward | Success |
|---|---|---|---:|---:|
| 0 | `an_easy_001` | set_name → … → submit (49-char prompt) | 0.00 | ✗ |
| 1 | `te_easy_001` | set_name → set_desc → add_skill → write_prompt → submit | 9.60 | ✓ |
| 2 | `db_easy_001` | set_name → set_desc → add_skill → write_prompt → submit | 9.60 | ✓ |
| 3 | `ws_easy_001` | set_name → set_desc → add_skill → write_prompt → submit | 9.60 | ✓ |
| 4 | `fi_easy_001` | set_name → set_desc → add_skill → write_prompt → submit | 9.60 | ✓ |
| 5 | `an_easy_001` | set_name → … → submit (49-char prompt) | 0.00 | ✗ |
| 6 | `fi_easy_002` | set_name → set_desc → add_skill → write_prompt → submit | 9.60 | ✓ |
| 7 | `te_easy_001` | set_name → set_desc → add_skill → write_prompt → submit | 9.60 | ✓ |
| 8 | `fi_easy_001` | set_name → set_desc → add_skill → write_prompt → submit | 9.60 | ✓ |
| 9 | `db_easy_001` | set_name → set_desc → add_skill → write_prompt → submit | 9.60 | ✓ |

**Mean reward: 7.68 | Success: 8/10**

The two failures share one root cause: 49-char prompts, one character below the 50-char gate. The model also picks `web-scraping` for most tasks regardless of domain — structure is learned, task-conditional skill selection is not yet.

![Component Learning](monitoring/colab_results_qwen3_1.7b/component_learning_labeled.png)

---

## Results

### Baseline comparison (20 easy-tier episodes each)

| Policy | Success rate | Mean reward | Max reward |
|---|---:|---:|---:|
| Random | 0% | 0.00 | 0.00 |
| Competent heuristic | 100% | 21.33 | 30.33 |
| Expert benchmark (mixed difficulty) | 95% | 16.79 | 19.57 |
| **Trained Qwen3-1.7B** | **80%** | **7.68** | **9.60** |

Random gets 0% because the gate blocks any `submit` without `name`, `description`, and a ≥ 50-char prompt. Heuristic proves the environment is reachable. The trained policy sits between random and heuristic — structure learned, content quality still improving.

![Baseline Comparison](monitoring/colab_results_qwen3_1.7b/baseline_comparison_labeled.png)

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

`yaml_valid` and `model_valid` are 1.00 from the start — the model never produces malformed YAML and always picks a valid model tier. `skill_selection` is the weakest signal: it learns the format (`add_skill` issues the right command) but not the content (which skill to pick).

![Component Curves](monitoring/colab_results_qwen3_1.7b/component_curves.png)

---

## What the Model Learned

1. A valid AGENT.md always needs `name`, `description`, `skills`, `model`, `system_prompt` — never submit without them
2. The correct episode structure is: name → description → skills → prompt → submit
3. YAML must be well-formed — `yaml_valid` gate fires immediately on any syntax error
4. `sonnet` is almost always the right model tier for easy tasks
5. `write_prompt` must be ≥ 50 chars — the gate is unforgiving

## What We Discovered (from the model's failures)

1. **The sign-flip bug** — storing penalty values as negatives and then subtracting them inverts their effect. GRPO found this in 4 gradient steps; code review missed it entirely
2. **Gate cliffs are sharp** — the 50-char prompt gate triggers at exactly 49 chars. The model learned the structure but not the margin
3. **Skill selection doesn't generalize** — `web-scraping` has the highest frequency in training data; the model over-indexes on it for all domains
4. **Real execution is non-negotiable** — the in-environment metrics showed 100% success; Goose showed 0% usefulness. Both numbers were true

---

## Problem Statements Addressed

### Primary: Theme #5 — Wild Card

Meta-agent design doesn't fit the other themes cleanly. It's not multi-agent (one agent), not long-horizon (7-step episodes), and not direct tool-use. Wild Card for what it is: a new RL target nobody is currently training, with verifiable rewards and real-artifact output.

- **Underexplored RL target:** AGENT.md generation framed as a multi-step decision problem with hard-verifiable rewards
- **Real artifact output:** generated specs run in Claude Code, Goose, Copilot, and anything following the [Agent Skills Open Standard](https://skills.sh)
- **Methodological contribution:** three-tier RLVR caught the reward-hack bug on the first Goose-pointed run

### Secondary: Theme #3 — World Modeling

- **POMDP structure:** hidden state = "what makes a good spec for this task?" — observable only through per-component reward + violations
- **Investigation tools:** `check_score` and `inspect_example` let the agent gather information before committing
- **Persistent state:** each command updates the spec dict; the agent must reason about what is already there

### Architectural: Theme #4 — Self-Improvement (built, partially demonstrated)

The closed-loop design (adversarial task generator + adaptive curriculum) is wired in code — [`server/adversarial.py`](server/adversarial.py), [`training/curriculum.py`](training/curriculum.py) — but the Colab run was 4 gradient steps, far below what is needed to show curriculum escalation. One genuine form of self-improvement did occur: the agent found the reward hack; I fixed the reward function. Agent/environment co-evolution, not the recursive skill amplification Theme #4 describes, but real nonetheless.

---

## Anti-Hack Defense

The policy will try to game the reward. Current defenses:

| Hack | Defense |
|---|---|
| Empty spec (noop→submit) | Gate: `has_required_fields` + `prompt_length_ok` zero all reward; `empty_spec` penalty −5.0 |
| Over-engineering (10+ skills) | Penalty: `over_engineered` −0.5 |
| Repetition (same action twice) | Penalty: `repetitive` −0.3 |
| Regression (break a passing check) | Penalty: `regression` −0.15 |
| Judge exploitation | Real execution at steps 3, 6, 9 anchors the judge score |

---

## Quick Start

### Try the live demo

[**huggingface.co/spaces/Kaviya-M/meta-agent-gym**](https://huggingface.co/spaces/Kaviya-M/meta-agent-gym)

Two tabs:
- **🛠 Build Step-by-Step** — pick a scenario, issue commands manually, watch reward update live
- **✨ Generate from Description** — type a task, the Qwen3-1.7B adapter emits commands; heuristic fallback runs if HF CPU cannot load the model

### Run locally

```bash
# Install
uv sync

# Start the environment server
python -m uvicorn server.app:app --reload --port 8000

# Open http://127.0.0.1:8000/web/

# Run the full test suite (35 tests)
pytest tests/ -q

# Reproduce the reward bug demo (instant, no GPU needed)
python scripts/demo_reward_fix.py

# Run the Goose harness (needs goose CLI + Claude Code installed)
python -m evaluation.goose_execution --smoke
```

### Train

```bash
# Colab T4 (free tier) — the path that produced the shipped run
python training/grpo_unsloth.py --model-id Qwen/Qwen2.5-0.5B

# L4 / A100 — onsite scale-up target
python training/grpo_unsloth.py \
    --model-id Qwen/Qwen3-1.7B \
    --per-device-train-batch-size 2 \
    --num-generations 4 \
    --gradient-accumulation-steps 4 \
    --max-seq-length 2048 \
    --num-epochs 3
```

Or use the end-to-end Colab notebook: [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb)

### Validate (hackathon pre-check)

```bash
# Requires bash + Docker + openenv-core
./validate-submission.sh https://kaviya-m-meta-agent-gym.hf.space

# Or run the three checks individually:
curl -s -X POST https://kaviya-m-meta-agent-gym.hf.space/reset \
     -H "Content-Type: application/json" -d '{}'
docker build .
python -c "from openenv.cli.commands.validate import validate; validate()"
```

---

## Project Layout

```
meta-agent-gym/
├── models.py                        # AgentSpec, Action, Observation, RewardConfig
├── client.py                        # OpenEnv HTTP/WebSocket client
├── inference.py                     # LLM episode runner
│
├── server/
│   ├── app.py                       # FastAPI (OpenEnv compatible, /metadata /mcp /health)
│   ├── environment.py               # reset → step → verify → reward
│   ├── verifiers.py                 # 5 hard verifiers (3 gates)
│   ├── rewards/
│   │   ├── reward.py                # multi-component reward (sign-flip fix line 158)
│   │   └── rubric_reward.py         # OpenEnv Rubric implementations
│   ├── rules/engine.py              # rule validation engine
│   ├── adversarial.py               # adversarial task generator
│   ├── tasks/scenarios.py           # 24 curriculum scenarios (4 phases)
│   └── inference_service.py         # /generate endpoint — Qwen3 LoRA adapter
│
├── evaluation/
│   ├── goose_execution.py           # Goose harness (offline real-execution tier)
│   └── fixtures/                    # 3 deterministic tasks + smoke_output.txt (3/3 PASS)
│
├── training/
│   ├── grpo_unsloth.py              # 4-bit LoRA GRPO (T4 + L4/A100)
│   ├── grpo_trl.py                  # full GRPO (H100)
│   ├── curriculum.py                # phase progression + mastery tracking
│   ├── rollout_collection.py        # rollout capture
│   └── grpo-unsloth-output/         # trained adapter + training_summary.json sentinel
│
├── tests/                           # 35 tests, all passing
│   ├── test_meta_agent_reward.py    # 5 regression tests for empty-spec penalty
│   └── test_inference_service.py    # inference + JSON parsing tests
│
├── data/
│   ├── baseline/                    # random + heuristic baseline trajectories
│   ├── colab_trained/               # 10 pre-fix trajectories (the empty-spec collapse)
│   ├── colab_trained_qwen3_1.7b/    # 10 post-fix eval rollouts (8/10 success)
│   └── post_fix/                    # before/after reward demo rollouts
│
├── monitoring/
│   ├── colab_results/               # heuristic-policy evaluation plots
│   └── colab_results_qwen3_1.7b/    # Qwen3 trained-adapter plots + report.json
│
├── notebooks/train_colab.ipynb      # end-to-end Colab training notebook
├── scripts/demo_reward_fix.py       # live +7.4 → -4.58 reward swing demo
├── static/index.html                # 2-tab dashboard
├── docs/
│   ├── competition/                 # HUGGINGFACE_BLOG.md, TRAINING_EVIDENCE.md
│   └── onsite/                      # ONSITE_TRAINING_PLAN.md
├── openenv.yaml
└── Dockerfile
```

---

## Submission Materials

| What | Where |
|---|---|
| Mini-blog | [`docs/competition/HUGGINGFACE_BLOG.md`](docs/competition/HUGGINGFACE_BLOG.md) |
| Training evidence + honest limitations | [`docs/competition/TRAINING_EVIDENCE.md`](docs/competition/TRAINING_EVIDENCE.md) |
| Onsite training plan | [`docs/onsite/ONSITE_TRAINING_PLAN.md`](docs/onsite/ONSITE_TRAINING_PLAN.md) |
| Sign-flip fix evidence | [`data/post_fix/REWARD_FIX_COMPARISON.md`](data/post_fix/REWARD_FIX_COMPARISON.md) |
| Pre-fix trajectories (the collapse) | [`data/colab_trained/`](data/colab_trained/) |
| Post-fix Qwen3-1.7B trajectories | [`data/colab_trained_qwen3_1.7b/`](data/colab_trained_qwen3_1.7b/) |
| Plots + report.json (50-ep heuristic eval) | [`monitoring/colab_results/`](monitoring/colab_results/) |
| Plots + report.json (Qwen3 adapter eval) | [`monitoring/colab_results_qwen3_1.7b/`](monitoring/colab_results_qwen3_1.7b/) |

---

## Stack

| Component | Tech |
|---|---|
| Environment | OpenEnv v0.2.1 (gymnasium-compatible) |
| Training | TRL 0.29 + Unsloth 4-bit LoRA, GRPO with DAPO loss |
| Judge (production) | Claude Sonnet, 6-dimension scoring |
| Real execution | Goose 1.27 + Claude Code CLI provider |
| Deployment | Docker on HF Spaces (cpu-basic) |
| Shipped model | Qwen3-1.7B (Apache 2.0, dual-mode reasoning + thinking) |
| Baseline run | Qwen2.5-0.5B (Apache 2.0, 4 gradient steps, sign-flip bug present) |

---

## Honest Scope

- **The Goose harness covers 3 Phase 1 (single-skill) tasks.** Expanding to Phase 2-4 multi-skill tasks is future work; the harness API accepts new tasks without interface changes.
- **Skill selection doesn't generalize yet.** The Qwen3 adapter picks `web-scraping` for most tasks regardless of domain. More training episodes with diverse task types are the clear next step.
- **The self-improvement track** (adversarial task generation, curriculum auto-escalation) is wired in code but not demonstrated end-to-end. The Colab run was 4 gradient steps — far below what is needed to see curriculum escalate. Specific future-work paths: VCRL ([arxiv 2509.19803](https://arxiv.org/html/2509.19803v1)) and Self-Evolving Curriculum ([arxiv 2505.14970](https://arxiv.org/pdf/2505.14970)).
- **The `monitoring/colab_results/` 50-episode plots** use the heuristic policy as a training-distribution proxy. The Qwen3 adapter inference plots are in `monitoring/colab_results_qwen3_1.7b/`.

---

Built for the OpenEnv Hackathon 2026 by [@Kaviyamurugadass](https://github.com/Kaviyamurugadass).
