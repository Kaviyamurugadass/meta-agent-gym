# meta-agent-gym: RL Environment for Agent Design

> **Goal:** Train a policy that generates complete AGENT.md files from user task descriptions.
>
> **Status:** Hackathon submission — infrastructure complete, 50-episode Qwen2.5-0.5B smoke-validation run (OpenEnv Hackathon 2026)

---

## Project Overview

This is a reinforcement learning environment using **OpenEnv** and **GRPO** to train models to design AI agents. The policy takes a task description and outputs an AGENT.md file (following the Agent Skills Open Standard) that works across Claude Code, Goose, Copilot, and other agent frameworks.

### Core Concept

```
Input: "Build an agent that scrapes product prices from e-commerce sites"
Output: Complete AGENT.md with:
  - name: product-price-scraper
  - description: Extract prices from e-commerce pages
  - skills: [web-scraping, html-parser, http-client]
  - model: sonnet
  - system_prompt: (complete instructions)
```

### Key Technologies

- **Environment:** OpenEnv (gymnasium-compatible RL env)
- **Training:** GRPO via TRL + Unsloth (4-bit LoRA for consumer GPUs)
- **Reward:** RLVR approach - multiple independent verifiers
- **Verification:** Three-tier (hard checks → fast judge → real execution)

---

## Architecture

### Three-Tier Verification (RLVR)

| Layer | What | Frequency | Cost |
|-------|------|-----------|------|
| **Hard Verifiers** | YAML parse, required fields, format | Every step (100%) | ~$0 |
| **Fast Judge** | Claude Sonnet quality scoring | 90% of steps | ~$0.01 |
| **Real Execution** | Goose runtime actual test | Steps 3, 6, 9 (10%) | ~$1-10 |

### Curriculum Strategy

```
Phase 1: 1 skill → Easy tasks → >50% success
Phase 2: 2-3 skills → Medium tasks → >30% success
Phase 3: 3-5 skills → Hard tasks → >10% success
Phase 4: 5+ skills → Expert tasks → >5% success
```

### Reward Components

| Component | Weight | Type |
|-----------|--------|------|
| yaml_valid | gate | Hard verifier |
| has_required_fields | gate | Hard verifier |
| skill_selection | 0.25 | Judge |
| description_quality | 0.20 | Judge |
| workflow_clarity | 0.20 | Judge |
| model_appropriateness | 0.15 | Judge |
| best_practices | 0.10 | Judge |
| efficiency | 0.10 | Judge |

Anti-hacking penalties:
- `empty_spec`: -5.0
- `over_engineered`: -0.5
- `repetitive`: -0.3
- `regression`: -0.15

---

## File Structure

```
openenv-agent-gym/
├── models.py                 # Action, Observation, AgentSpec, State schemas
├── client.py                 # OpenEnv HTTP client (exports class `Env`)
├── inference.py              # LLM-driven episode runner
│
├── server/
│   ├── app.py                # FastAPI + WebSocket endpoints (OpenEnv)
│   ├── environment.py        # reset → step → verify → reward
│   ├── robust_environment.py # defensive wrapper for onsite demos
│   ├── verifiers.py          # hard verifiers (RLVR layer 1)
│   ├── skills.py             # skill registry + domain templates
│   ├── adversarial.py        # targeted task generation
│   ├── rewards/reward.py     # multi-component reward + penalties
│   ├── rules/engine.py       # rule validation engine
│   ├── runtime/goose.py      # real-execution tier (RLVR layer 3)
│   └── tasks/
│       ├── scenarios.py      # 21 curriculum scenarios (phase 1–4)
│       └── generator.py      # adversarial generation
│
├── training/
│   ├── grpo_trl.py           # Full GRPO (H100) — Qwen3-1.7B target
│   ├── grpo_unsloth.py       # 4-bit LoRA (T4/Colab) — Qwen2.5-0.5B
│   ├── curriculum.py         # phase progression
│   ├── evaluation.py         # trajectory metrics
│   ├── benchmark.py          # expert-trajectory runner
│   ├── rollout_collection.py # rollout capture
│   ├── reward_backend.py     # reward RPC
│   └── monitoring.py         # training telemetry
│
├── evaluation/
│   ├── onsite_evaluation.py  # full multi-dimensional evaluator
│   └── simple_evaluation.py  # lightweight fallback
│
├── tests/                    # pytest suite (conftest lives here)
│   ├── conftest.py
│   └── test_*.py             # 11 files — smoke, reward, observation, verifiers…
│
├── data/baseline/            # random + heuristic baseline trajectories
├── notebooks/                # 01_demo, 02_train_grpo, 03_evaluate, train_colab
├── monitoring/               # plots + colab_results/report.json
├── models/colab_model/       # trained-model metadata (weights gitignored)
├── scripts/                  # deploy.sh, generate_plots.py, interactive_test.py, …
├── static/index.html         # interactive dashboard
│
└── docs/
    ├── competition/          # submission narrative + technical evidence
    ├── onsite/               # testing + results guides
    └── learnings/            # hackathon reference material
```

---

## Development Guidelines

### Running the Environment

```bash
# Start the OpenEnv server
uvicorn server.app:app --reload --port 8000

# Run tests
pytest tests/

# Test environment manually
python -c "
from client import Env
from models import Action, ActionCommand
with Env('http://localhost:8000') as env:
    obs = env.reset(scenario_name='ws_easy_001')
    print(obs.summary, obs.reward_breakdown)
"
```

### Working with Rewards

The reward system uses **multiple independent components** (RLVR approach):

```python
# All reward components are tracked separately
reward_data = {
    "total": 7.2,
    "breakdown": {
        "yaml_valid": 1.0,      # Hard gate
        "has_required_fields": 1.0,
        "skill_selection": 0.8,
        "description_quality": 0.9,
        # ...
    },
    "penalties": {
        "empty_spec": 0.0,
        "over_engineered": -0.5,
    }
}
```

**Important:** When modifying rewards, maintain independent components for GRPO variance.

### Adding Test Cases

Test cases live in `server/tasks/scenarios.py`. Follow the curriculum:

```python
# Start with single-skill tasks for Phase 1
EASY_TASKS = [
    {
        "id": "ws_easy_001",
        "domain": "web",
        "difficulty": "easy",
        "task_description": "Extract product prices from a single page",
        "required_skills": ["web-scraping"],  # Single skill!
        # ...
    }
]
```

### Training

```bash
# Full GRPO (H100)
python training/grpo_trl.py

# Unsloth variant (T4/Colab)
python training/grpo_unsloth.py

# Evaluate
python training/evaluation.py
```

---

## Anti-Hacking Patterns

The policy will try to game the reward system. We prevent:

| Hack | Prevention |
|------|------------|
| Empty specs | Hard gate: prompt must be >50 chars |
| Over-engineering | Penalty: >10 skills or wrong model tier |
| Format only | Judge scoring + real execution calibration |
| Judge exploitation | Real execution at steps 3, 6, 9 validates judge |

**When adding features:** Ask "How could a policy hack this?"

---

## Track Alignment (OpenEnv Hackathon)

| Track | How We Fit |
|-------|-----------|
| Multi-Agent | Sub-agent specialist roles (static analyzer, doc reviewer) |
| World Modeling | POMDP: hidden state = "what makes a good agent" |
| Long-Horizon | Multi-step generation with investigation commands |
| Self-Improvement | Adversarial designer generates harder tests |

---

## Key Commands

```bash
# Development server
uvicorn server.app:app --reload --port 8000

# Run specific test
pytest tests/test_reward_quality.py -v

# Format code
black . && isort .

# Type check
mypy .

# Deploy to HF Space
bash scripts/deploy.sh

# Regenerate training plots from monitoring/colab_results/report.json
python scripts/generate_plots.py
```

---

## Important Notes

1. **Deploy early** - Deploy to HF Space before serious training to catch API/packaging issues
2. **Monitor all reward components** - Don't just track total reward
3. **Calibrate judge** - Track fast judge vs real execution drift
4. **Curriculum is critical** - Start with single-skill tasks for >0 success probability
5. **Anti-hacking is ongoing** - New hacks will emerge during training

---

## Memory Context

- Project is in active hackathon mode (OpenEnv 2025)
- Focus is on core components first, UI later
- Using PyTorch stack: TRL + Unsloth + OpenEnv
- Target: Generate production-ready AGENT.md files
- Verification: Hard checks (100%) + Judge (90%) + Real execution (10% at steps 3,6,9)
