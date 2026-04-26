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

### Can a small model learn to design AI agents by getting feedback on the agents it builds?

This repo is an RL environment for training a model to write `AGENT.md` files. The model gets a task description, acts through a small command set (`set_name`, `add_skill`, `write_prompt`, `submit`, etc.), and receives reward from a verifier stack.

The most interesting part was not the happy path. The first training run looked successful, then Goose execution showed the model had learned to exploit a sign bug in the reward function. After fixing that, the Qwen3-1.7B adapter learned the basic structure of an agent spec, but still struggles with task-specific skill selection.

So this project is both a working OpenEnv gym and a case study in why real execution belongs next to reward metrics.

> Built for OpenEnv Hackathon 2026 | Trained: Qwen3-1.7B + 4-bit LoRA | GRPO + DAPO loss | Deployed on HF Spaces

| Resource | Link |
|---|---|
| 🤗 HF Space (live environment) | https://huggingface.co/spaces/Kaviya-M/meta-agent-gym |
| 🧠 Trained LoRA adapter | https://huggingface.co/Kaviya-M/meta-agent-gym-adapter |
| 📓 Colab training notebook | https://colab.research.google.com/github/Kaviyamurugadass/meta-agent-gym/blob/main/notebooks/train_colab.ipynb |
| 💻 GitHub repository | https://github.com/Kaviyamurugadass/meta-agent-gym |
| 📝 Blog post | [`Blog.md`](Blog.md) |

---

## What Happened

### 1. Building the gym

The first big choice was the action space. I considered letting the model generate free-form markdown, but that makes GRPO credit assignment painful: the model writes a long sequence and only learns at the end whether it worked.

I used 14 discrete commands instead:

```
set_name        set_description   add_skill      remove_skill
set_model       write_prompt      add_tools      set_memory
set_max_turns   check_score       inspect_example  submit
noop            inspect
```

Each command has a clear effect on the current spec. The policy can inspect, check score, add fields, and then submit. That gives the environment a chance to reward progress before the final artifact exists.

The verifier has five hard checks and six judge dimensions. Three hard checks are gates: if YAML is invalid, required fields are missing, or the prompt is too short, the step reward goes to zero.

### 2. The first run looked suspiciously good

I first trained Qwen2.5-0.5B + 4-bit LoRA on a Colab T4. One epoch, 8 episodes, 2 generations per prompt. The numbers came back:

```
success_rate = 10/10 = 100%
mean_reward  = +51.80 per episode
```

That was too clean for such a small run, so I checked the generated artifacts.

### 3. Goose caught the real failure

When I ran the generated `AGENT.md` files through Goose, every trajectory looked like this:

```
noop → noop → noop → noop → noop → noop → submit
```

The model had not learned agent design. It had learned that doing nothing paid well.

The bug was in the reward aggregation:

```python
# Before: subtracting a negative adds a bonus
total = core + bonus + progress - penalty - regression - sum(anti_hack_penalties.values())

# After: penalties stay negative
total = core + bonus + progress - penalty - regression + sum(anti_hack_penalties.values())
```

`anti_hack_penalties` stores values as negative numbers, for example `empty_spec = -5.0`. Subtracting those values turned penalties into bonuses. An empty spec was worth **+7.4 per step**. GRPO found that in four gradient steps.

That failure is why the Goose layer is in the project. The environment metric said success; execution said the artifact was useless.

### 4. After the fix

After the fix, I trained Qwen3-1.7B with 4-bit LoRA. This was still small-scale: 25 dataset episodes, 2 epochs, 2 generations. It learned the basic command pattern:

```
set_name → set_description → add_skill → write_prompt → submit
```

**8/10 evaluation episodes succeeded with mean reward 7.68.** The two failures wrote 49-character prompts, one character below the gate. The model also overuses `web-scraping` across domains. So the current adapter has learned structure, but the next target is task-conditioned skill selection.

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
│  │  └─ Runs the AGENT.md as an execution check                        │ │
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
| Goose execute | FAIL, empty spec | PASS |

### Reward before vs after the sign-flip fix

| | Per step | Per 7-step episode |
|---|---:|---:|
| Empty spec — **before** fix | **+7.40** | **+51.80** |
| Empty spec — **after** fix | −4.58 | −32.20 |
| Competent heuristic | +3.05 | +21.33 |
| Trained Qwen3-1.7B (eval avg) | +1.37 | **+7.68** |
| **Swing from the fix** | **11.98** | **84.00** |

That is an 84-point swing over one 7-step episode. Before the fix, an empty spec was rewarded **2.4x more** than the competent heuristic policy.

![Reward Progression](monitoring/colab_results_qwen3_1.7b/reward_progression_labeled.png)

---

## Training Runs

### Run 1: Qwen2.5-0.5B, pre-fix

4 gradient steps, 8 episodes, Colab T4.

| Episode | Actions | Reward | Notes |
|---|---|---:|---|
| 1–8 | noop × 6 → submit | +51.80 | Empty-spec exploit discovered in ep 1 |

**Mean: +51.80 | Success: 10/10, but all empty specs**

The run was real. The training sentinel (`training_summary.json`) records `real_training: true`. The reward was wrong.

### Run 2: Qwen3-1.7B, post-fix

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

The two failures share one root cause: 49-character prompts, one character below the gate. The model also picks `web-scraping` for too many tasks. Structure is learned; task-conditional skill selection is not.

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

Random gets 0% because the gate blocks any `submit` without `name`, `description`, and a prompt of at least 50 characters. The heuristic proves the environment is reachable. The trained policy sits between them: it learned the workflow, but content quality still needs more training.

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

`yaml_valid` and `model_valid` are easy for the model. `skill_selection` is the weak point: it learns to issue `add_skill`, but not always which skill to add.

![Component Curves](monitoring/colab_results_qwen3_1.7b/component_curves.png)

---

## What the Model Learned

1. A valid AGENT.md always needs `name`, `description`, `skills`, `model`, and `system_prompt`. Never submit without them.
2. The correct episode structure is: name → description → skills → prompt → submit
3. YAML must be well-formed. The `yaml_valid` gate fires immediately on any syntax error.
4. `sonnet` is almost always the right model tier for easy tasks
5. `write_prompt` must be at least 50 characters. The gate is unforgiving.

## What the Failures Exposed

1. **The sign-flip bug.** Storing penalties as negatives and then subtracting them inverts the effect. GRPO found this in 4 gradient steps.
2. **Gate cliffs are sharp.** The 50-character prompt gate fails at 49. The model learned the pattern, but not the margin.
3. **Skill selection needs more coverage.** `web-scraping` appears often enough that the model overuses it.
4. **Real execution matters.** The in-environment metric showed 100% success; Goose showed 0% usefulness. Both were accurate views of different things.

---

## Problem Statements Addressed

### Primary: Theme #5 — Wild Card

Meta-agent design does not fit the other themes cleanly. It is not multi-agent, not long-horizon, and not direct tool-use. I placed it under Wild Card because the RL target is unusual: generate a real agent specification and grade the artifact.

- **Underexplored RL target:** AGENT.md generation framed as a multi-step decision problem with hard-verifiable rewards
- **Real artifact output:** generated specs run in Claude Code, Goose, Copilot, and anything following the [Agent Skills Open Standard](https://skills.sh)
- **Methodological contribution:** three-tier RLVR caught the reward-hack bug on the first Goose-pointed run

### Secondary: Theme #3 — World Modeling

- **POMDP structure:** hidden state = "what makes a good spec for this task?", observable through per-component reward and violations
- **Investigation tools:** `check_score` and `inspect_example` let the agent gather information before committing
- **Persistent state:** each command updates the spec dict; the agent must reason about what is already there

### Architectural: Theme #4 — Self-Improvement (built, partially demonstrated)

The closed-loop pieces are in code: adversarial task generation in [`server/adversarial.py`](server/adversarial.py), curriculum tracking in [`training/curriculum.py`](training/curriculum.py). The Colab run was too small to demonstrate full curriculum escalation. The concrete self-improvement loop here is simpler: the policy found a reward bug, I fixed the environment, and the next run learned the intended structure.

---

## Anti-Hack Defense

The first run made this part less theoretical. Current defenses:

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
- **Build Step-by-Step** - pick a scenario, issue commands manually, and watch reward update live
- **Generate from Description** - type a task and let the Qwen3-1.7B adapter emit commands. If HF CPU cannot load the model, the app falls back to a heuristic path

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
# Colab T4 (free tier), the path that produced the shipped run
python training/grpo_unsloth.py --model-id Qwen/Qwen2.5-0.5B

# L4 / A100, onsite scale-up target
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
│   └── inference_service.py         # /generate endpoint for the Qwen3 LoRA adapter
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
│   ├── competition/                 # TRAINING_EVIDENCE.md, PITCH.md, SLIDES.md
│   └── onsite/                      # ONSITE_TRAINING_PLAN.md
├── openenv.yaml
└── Dockerfile
```

---

## Submission Materials

| What | Where |
|---|---|
| Mini-blog | [`Blog.md`](Blog.md) |
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

- **The Goose harness covers 3 Phase 1 tasks.** Expanding it to Phase 2-4 multi-skill tasks is future work. The harness API already accepts more tasks.
- **Skill selection does not generalize yet.** The Qwen3 adapter often picks `web-scraping` regardless of domain. More varied training data is the obvious next step.
- **The self-improvement pieces are wired, not fully demonstrated.** Adversarial tasks and curriculum escalation exist in code, but the Colab run was too small to show the curriculum moving through phases. Useful next references: VCRL ([arxiv 2509.19803](https://arxiv.org/html/2509.19803v1)) and Self-Evolving Curriculum ([arxiv 2505.14970](https://arxiv.org/pdf/2505.14970)).
- **The `monitoring/colab_results/` plots use the heuristic policy as a proxy.** The Qwen3 adapter plots are in `monitoring/colab_results_qwen3_1.7b/`.

---

Built for the OpenEnv Hackathon 2026 by [@Kaviyamurugadass](https://github.com/Kaviyamurugadass).
