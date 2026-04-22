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

> **RL environment for training AI agents to design AI agents.** Takes a task description → outputs complete AGENT.md files.

---

## What It Does

```
Input:  "Build an agent that scrapes product prices from e-commerce sites"
Output: Complete AGENT.md with:
  - name: product-price-scraper
  - description: Extract prices from e-commerce pages
  - skills: [web-scraping, html-parser, http-client]
  - model: sonnet
  - system_prompt: (complete workflow instructions)
```

The output works across Claude Code, Goose, Copilot, and other agent frameworks.

---

## Approach: RLVR (Reinforcement Learning with Verifiable Rewards)

Three-tier verification — no learned reward models:

| Tier | What | Frequency | Cost |
|------|------|-----------|------|
| **Hard Verifiers** | YAML parse, required fields | Every step (100%) | ~$0 |
| **Fast Judge** | Claude Sonnet quality scoring | 90% of steps | ~$0.01 |
| **Real Execution** | Goose runtime test | Steps 3, 6, 9 (10%) | ~$1-10 |

### Multi-Component Rewards

All components tracked separately for GRPO variance:

| Component | Type | Weight |
|-----------|------|--------|
| `yaml_valid` | Hard gate | — |
| `has_required_fields` | Hard gate | — |
| `skill_selection` | Judge | 0.25 |
| `description_quality` | Judge | 0.20 |
| `workflow_clarity` | Judge | 0.20 |
| `model_appropriateness` | Judge | 0.15 |
| `best_practices` | Judge | 0.10 |
| `efficiency` | Judge | 0.10 |

**Anti-hacking penalties:** `empty_spec: -5.0`, `over_engineered: -0.5`, `regression: -0.15`

---

## Curriculum

Start with >0 success probability (critical for RL):

| Phase | Skills | Difficulty | Success Target |
|-------|--------|------------|----------------|
| 1 | 1 | Easy | >50% |
| 2 | 2-3 | Medium | >30% |
| 3 | 3-5 | Hard | >10% |
| 4 | 5+ | Expert | >5% |

---

## Hackathon Theme Alignment

### 🥇 Theme #4 - Self-Improvement (Primary)

| Theme Requirement | Our Implementation |
|-------------------|---------------------|
| Generate new challenges | **Adversarial Designer** creates test cases targeting policy weaknesses |
| Escalate difficulty | **Curriculum**: 1 skill → 2-3 → 3-5 → 5+ skills based on performance |
| Adaptive curricula | **CurriculumController** progresses when policy achieves >70% on current level |
| Recursive skill amplification | Policy learns to design agents; adversarial designer makes it harder |

> *"An environment where agents learn to design other agents, with an adversarial designer that escalates difficulty based on policy weaknesses."*

### 🥈 Theme #3.1 - World Modeling (Secondary)

| Theme Requirement | Our Implementation |
|-------------------|---------------------|
| Maintain consistent internal state | **POMDP structure** — hidden state = "what makes a good agent?" |
| Partial observability | Current spec state is partial; full quality only revealed after submit |
| Update beliefs based on outcomes | Investigation commands (`check_score`, `inspect_example`) + feedback loop |
| Multi-step workflows | Command-based actions: set_name → add_skill → write_prompt → submit |
| Real hard work, not shortcuts | Anti-hacking penalties prevent format-only exploits |

> *"POMDP structure where the 'what makes a good agent' state is hidden, requiring investigation commands and feedback-driven belief updates."*

### 🥉 Theme #2 - Long-Horizon Planning (Tertiary)

| Theme Requirement | Our Implementation |
|-------------------|---------------------|
| Multi-step reasoning | 7-step generation process with interdependent decisions |
| Sparse/delayed rewards | Reward only meaningful after submit; per-step rewards are incremental |
| Decompose goals | Policy must decompose "build an agent" into discrete commands |
| Recover from early mistakes | Investigation tools + regression penalties prevent breaking progress |

---

## Quick Start

```bash
# Install
uv sync

# Start environment
uvicorn server.app:app --reload --port 8000

# Run tests
pytest tests/ -q

# Interact
python -c "from client import MetaAgentClient; c = MetaAgentClient(); print(c.reset('task_001'))"
```

---

## Training

**T4 (Colab) — Unsloth 4-bit LoRA:**

```bash
python training/grpo_unsloth.py --model-id Qwen/Qwen3-0.6B
```

**H100/A100 — Full GRPO:**

```bash
python training/grpo_trl.py --model-id Qwen/Qwen3.5-4B
```

---

## File Structure

```
meta_agent_gym/
├── models.py              # Action, Observation, AgentSpec schemas
├── client.py              # OpenEnv HTTP client
├── server/
│   ├── app.py            # FastAPI endpoint
│   ├── environment.py    # OpenEnv reset/step/state
│   ├── verifiers.py      # Hard YAML/field checks
│   ├── judge.py          # Fast judge + calibration
│   ├── rewards/
│   │   ├── reward.py     # Multi-component rewards
│   │   └── anti_hack.py  # Penalty system
│   └── tasks/
│       └── scenarios.py  # Curriculum test cases
├── training/
│   ├── grpo_trl.py       # TRL GRPO (H100)
│   ├── grpo_unsloth.py   # 4-bit LoRA (T4)
│   └── curriculum.py     # Phase progression
└── tests/
```

---

## Stack

- **Environment:** OpenEnv (gymnasium-compatible)
- **Training:** TRL + Unsloth (4-bit LoRA)
- **Fast Judge:** Claude Sonnet
- **Real Execution:** Goose runtime
- **Philosophy:** RLVR — verifiable rewards, not learned models
