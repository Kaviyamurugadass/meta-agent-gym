# Teaching a 1.7B Model to Design AI Agents

> **Short version:** I built a small RL environment where a model learns to write `AGENT.md` specs from task descriptions. The first training run looked great on paper, then Goose execution showed the model had only learned to exploit a reward bug. That failure ended up being the most useful part of the project.

**Built for OpenEnv Hackathon 2026 | Qwen3-1.7B + 4-bit LoRA | GRPO + DAPO | HF Spaces**

| | |
|---|---|
| 🤗 Live demo | [huggingface.co/spaces/Kaviya-M/meta-agent-gym](https://huggingface.co/spaces/Kaviya-M/meta-agent-gym) |
| 🧠 Trained adapter | [Kaviya-M/meta-agent-gym-adapter](https://huggingface.co/Kaviya-M/meta-agent-gym-adapter) |
| 💻 Code | [github.com/Kaviyamurugadass/meta-agent-gym](https://github.com/Kaviyamurugadass/meta-agent-gym) |
| 📓 Colab notebook | [train_colab.ipynb](https://colab.research.google.com/github/Kaviyamurugadass/meta-agent-gym/blob/main/notebooks/train_colab.ipynb) |

---

## The Question

I started with a practical question:

Can a small model learn to design useful agents if we give it a task, a constrained action space, and feedback on the agent it builds?

That became **meta-agent-gym**. The input is a task such as *"Build an agent that reviews pull requests for security issues"*. The output is a complete `AGENT.md`: name, description, skills, model tier, and system prompt. The same spec can be used in Claude Code, Goose, Cursor, and other tools that understand the Agent Skills format.

---

## Act 1 - Building the Gym

### The action space decision

The first real design choice was whether the model should write free-form markdown or act through commands. Free-form generation is closer to how a human writes an `AGENT.md`, but it gives GRPO very little signal until the end.

So I used a small command set instead:

```
set_name        set_description   add_skill       remove_skill
set_model       write_prompt      add_tools       set_memory
set_max_turns   check_score       inspect_example submit
noop            inspect
```

Each command changes the environment state in a predictable way. The agent can inspect, check score, add fields, and only then submit. That made the reward easier to debug, and it gave the policy intermediate feedback instead of one giant pass/fail at the end.

### Three-tier verification (RLVR)

I did not want the whole project to depend on one judge score, so the verifier has three layers:

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
└─ Actually runs the AGENT.md, which gives a check the judge cannot fake
```

The hard checks are deliberately boring. They catch format errors, missing fields, and too-short prompts before anything expensive or subjective runs.

---

## Act 2 - The First Run Looked Too Good

Trained Qwen2.5-0.5B + 4-bit LoRA on a free Colab T4. One epoch, 8 episodes, 2 generations per prompt. The numbers came back immediately:

```
success_rate = 10/10 = 100%
mean_reward  = +51.80 per episode
```

That was the first red flag. A tiny run should not look that perfect.

---

## Act 3 - Goose Caught the Problem

Then I ran the generated files through Goose instead of just trusting the environment metrics. Every single trajectory was:

```
noop → noop → noop → noop → noop → noop → submit
```

So the model had not learned agent design. It had learned that doing nothing was rewarded.

I went back into `server/rewards/reward.py` and eventually found the issue:

```python
# Before: subtracting a negative adds a bonus
total = core + bonus + progress - sum(anti_hack_penalties.values())
#                                    ^ anti_hack_penalties stores -5.0, -0.5 etc.
#                                      subtracting negatives = ADDING them

# After: penalties stay negative
total = core + bonus + progress + sum(anti_hack_penalties.values())
```

`anti_hack_penalties` stores values as **negative** numbers (`empty_spec = -5.0`). Subtracting a negative flipped the sign. An empty spec was worth **+7.40 per step**.

GRPO found it in **four gradient steps**.

That was embarrassing, but useful. The environment metrics said success. Goose said the artifact was useless. Both observations were true, and that is exactly why the real-execution layer belongs in the loop.

---

## Before and After

### Reward swing from the sign-flip fix

| | Per step | Per 7-step episode |
|---|---:|---:|
| Empty spec — **before** fix | **+7.40** | **+51.80** |
| Empty spec — **after** fix | −4.58 | −32.20 |
| Competent heuristic | +3.05 | +21.33 |
| Trained Qwen3-1.7B (eval avg) | +1.37 | **+7.68** |
| **Total swing from the fix** | **11.98** | **84.00** |

That is an 84-point swing over a 7-step episode. Before the fix, the environment paid more for an empty spec than for a competent hand-written policy.

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
| Goose execute | FAIL, empty spec | PASS |

---

## Act 4 - Training After the Fix

After fixing the sign error, I trained Qwen3-1.7B with 4-bit LoRA. This was still a small hackathon run: 25 dataset episodes, 2 epochs, 2 generations. The model did not become a great agent designer, but it did learn the basic structure.

### Total reward curve (training)

![Total Reward Curve](monitoring/colab_results_qwen3_1.7b/total_reward_curve.png)

The curve starts near zero because the hard gates wipe out bad submissions. As training continues, the model starts putting the required fields in the right order.

### Reward progression (labeled)

![Reward Progression](monitoring/colab_results_qwen3_1.7b/reward_progression_labeled.png)

The labeled plot marks the two most important transitions: required fields become consistent, and prompt length stops being the main failure.

### Baseline comparison (20 easy-tier episodes each)

| Policy | Success rate | Mean reward | Max reward |
|---|---:|---:|---:|
| Random | 0% | 0.00 | 0.00 |
| Competent heuristic | 100% | 21.33 | 30.33 |
| Expert benchmark (mixed difficulty) | 95% | 16.79 | 19.57 |
| **Trained Qwen3-1.7B** | **80%** | **7.68** | **9.60** |

![Baseline Comparison](monitoring/colab_results_qwen3_1.7b/baseline_comparison_labeled.png)

Random gets 0% because the gates block any submission without `name`, `description`, and a prompt of at least 50 characters. The heuristic proves the environment is reachable. The trained model lands in the middle: it learned the shape of a solution, but not yet the domain-specific content.

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

`yaml_valid` and `model_valid` are easy for the policy. `skill_selection` is not. The model learns to call `add_skill`, but it often chooses the wrong skill for the task.

### Success rate over training

![Success Rate Curve](monitoring/colab_results_qwen3_1.7b/success_rate_curve.png)

![Success Rate Labeled](monitoring/colab_results_qwen3_1.7b/success_rate_labeled.png)

Success climbs once the model stops submitting incomplete specs.

### Per-component reward curves

![Component Curves](monitoring/colab_results_qwen3_1.7b/component_curves.png)

Each line is one reward component. Keeping them separate made debugging much easier, especially when one component stalled while others improved.

### Full comparison (all metrics)

![Full Comparison](monitoring/colab_results_qwen3_1.7b/full_comparison.png)

This chart is mostly for sanity checking: total reward, success rate, and component scores in one place.

---

## What the Model Learned

The trained adapter picked up a few concrete rules:

1. A valid AGENT.md needs `name`, `description`, `skills`, `model`, and `system_prompt`. Never submit without all of them.
2. The correct episode order is: name → description → skill → prompt → submit
3. YAML must be well-formed. The `yaml_valid` gate fires on any syntax error.
4. `sonnet` is almost always the right tier for easy tasks
5. `write_prompt` must be at least 50 characters. The gate is unforgiving at 49.

## What the Failures Taught Me

1. **Gate cliffs are sharp.** The 50-character prompt gate fails at 49. The model learned the pattern, but not the safety margin.
2. **Skill selection needs more data.** `web-scraping` appears often enough that the model overuses it outside web tasks.
3. **Independent checks matter.** The sign-flip bug survived code review and showed up immediately once Goose executed the artifacts.

---

## The Live Demo

The Space has two modes:

### Build Step-by-Step
Pick a scenario, issue commands manually, and watch the reward breakdown update after each step. This is the best way to see what the verifiers care about.

### Generate from Description
Type a task description and the Qwen3-1.7B LoRA adapter generates the command sequence. The result goes through the same verifier stack. If `GROQ_API_KEY` is set, the UI can also show an LLM judge score.

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

This is still a hackathon project. The useful part is that the environment and failure modes are real, not that the shipped model is finished.

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

## 🎥 Watch My YouTube Demo Video

See the complete project explanation in under 60 seconds - from empty prompt to expert agent designer:

**[🎥 Watch on YouTube](https://www.youtube.com/shorts/F3YGDmDQKJk?si=gX2UVq2Z4D_QPl1e)**

---

*Built for OpenEnv Hackathon 2026 by [@Kaviyamurugadass](https://github.com/Kaviyamurugadass)*
