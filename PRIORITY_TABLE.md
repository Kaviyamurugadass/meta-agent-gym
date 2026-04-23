# meta-agent-gym: Priority Table

> Simple view of what to build, in what order.
>
> **Updated:** 2025-04-23 - Based on OpenEnv Hackathon learnings

---

## Verification Strategy (RLVR Approach)

| Layer | What | When | Cost |
|-------|------|------|------|
| **Hard Verifiers** | YAML parse, required fields, format | Every step (100%) | ~$0 |
| **Fast Judge** | Claude Sonnet quality scoring | 90% of steps | ~$0.01 |
| **Real Execution** | Goose runtime actual test | Steps 3, 6, 9 (10%) | ~$1-10 |

**Why this works:**
- Hard verifiers prevent format hacks
- Judge provides quality signal for training
- Real execution validates judge calibration

---

## Curriculum Strategy

| Phase | Skill Count | Difficulty | Success Target |
|-------|-------------|------------|----------------|
| **Phase 1** | 1 skill | Easy | >50% episodes |
| **Phase 2** | 2-3 skills | Medium | >30% episodes |
| **Phase 3** | 3-5 skills | Hard | >10% episodes |
| **Phase 4** | 5+ skills | Expert | >5% episodes |

**Why this works:**
- Ensures >0 success probability early (critical for RL)
- Gradual complexity prevents learning stall
- Each phase builds on previous

---

## P0 - Critical Path (Week 1)

| # | Component | File | Output | Time |
|---|-----------|------|--------|------|
| 1 | **Hard Verifiers** | `server/verifiers.py` | YAML check, field presence, format | 2h |
| 2 | **AGENT.md Schema** | `models.py` (extend) | Add `AgentSpec` with `to_markdown()` | 2h |
| 3 | **Action Commands** | `models.py` (extend) | Add meta-agent commands to `ActionCommand` | 1h |
| 4 | **Observation** | `models.py` (extend) | Add meta-agent fields to `Observation` | 1h |
| 5 | **Skill Registry** | `server/skills.py` | `AVAILABLE_SKILLS`, single-skill tasks first | 1h |
| 6 | **Test Cases (Easy)** | `server/tasks/scenarios.py` | Single-skill tasks (curriculum phase 1) | 2h |
| 7 | **Fast Judge** | `server/judge.py` | Claude Sonnet, 5-dim scoring | 3h |
| 8 | **Multi-Component Reward** | `server/rewards/reward.py` | Hard verifier + judge + penalties | 3h |
| 9 | **OpenEnv Environment** | `server/environment.py` | `reset()`, `step()`, `state()` with 2-tier eval | 4h |
| 10 | **Goose Runner** | `server/runtime/goose.py` | Execute agents at steps 3, 6, 9 | 2h |
| 11 | **Calibration Tracker** | `server/judge.py` | Track fast vs real drift | 1h |
| 12 | **Unit Tests** | `tests/test_*.py` | Test all P0 components | 3h |
| 13 | **Deploy to HF** | `scripts/deploy.sh` | Early deployment catches issues | 1h |

**P0 Total: ~26 hours**

---

## P1 - MVP Enhancement (Week 2)

| # | Component | File | Output | Time |
|---|-----------|------|--------|------|
| 14 | **GRPO Trainer (H100)** | `training/grpo_trl.py` | Full GRPO training loop | 3h |
| 15 | **Unsloth Trainer (T4)** | `training/grpo_unsloth.py` | 4-bit LoRA variant | 2h |
| 16 | **Curriculum Controller** | `training/curriculum.py` | 1→2→3→5+ skill progression | 2h |
| 17 | **Medium/Hard Tests** | `server/tasks/scenarios.py` | Multi-skill tasks (phases 2-4) | 2h |
| 18 | **Adversarial Designer** | `server/adversarial.py` | Generate hard test cases | 2h |
| 19 | **Anti-Hacking Checks** | `server/rewards/anti_hack.py` | Penalize empty/over-engineered specs | 2h |
| 20 | **Monitoring Dashboard** | `training/monitoring.py` | Track all reward components | 2h |
| 21 | **Evaluator** | `training/evaluation.py` | Before/after metrics | 2h |
| 22 | **Integration Tests** | `tests/test_training.py` | End-to-end pipeline | 2h |

**P1 Total: ~21 hours**

---

## P2 - Polish (Week 3+)

| # | Component | File | Output | Time |
|---|-----------|------|--------|------|
| 23 | **Investigation Tools** | `server/environment.py` | `check_score`, `inspect_example` | 2h |
| 24 | **More Test Cases** | `server/tasks/scenarios.py` | 20+ across all domains | 4h |
| 25 | **Documentation** | README.md | Full usage guide + reward justification | 2h |
| 26 | **Demo Video** | `demo/` | 60-90s walkthrough | 2h |

**P2 Total: ~10 hours**

---

## Reward Components (Multi-Independent Rewards)

| Component | Weight | Type | Purpose |
|-----------|--------|------|---------|
| `yaml_valid` | 0.0/1.0 | Hard gate | Must pass (YAML parse check) |
| `has_required_fields` | 0.0/1.0 | Hard gate | Must pass (name, description, prompt) |
| `skill_selection` | 0.25 | Judge | Required skills present, minimal extra |
| `description_quality` | 0.20 | Judge | Clear delegation guidance |
| `workflow_clarity` | 0.20 | Judge | Step-by-step instructions |
| `model_appropriateness` | 0.15 | Judge | Model matches task complexity |
| `best_practices` | 0.10 | Judge | Follows domain patterns |
| `efficiency` | 0.10 | Judge | No over-engineering |
| **Penalties** | | | |
| `empty_spec` | -5.0 | Anti-hack | No prompt or empty skills |
| `over_engineered` | -0.5 | Anti-hack | >10 skills or wrong model tier |
| `regression` | -0.15 | Anti-hack | Broke previously passing check |

---

## Quick Start Sequence

```
Day 1 (6h):  1 → 2 → 3 → 4 (Verifiers + Models)
Day 2 (6h):  5 → 6 → 7 (Skills + Tests + Judge start)
Day 3 (6h):  7 (finish) → 8 (Reward)
Day 4 (6h):  9 → 10 (Environment + Goose)
Day 5 (6h):  11 → 12 → 13 (Calibration + Tests + Deploy)
Day 6 (4h):  14 → 15 (GRPO trainers)
Day 7 (4h):  16 → 17 (Curriculum + Medium/Hard tests)
Day 8 (4h):  18 → 19 (Adversarial + Anti-hacking)
Day 9 (4h):  20 → 21 → 22 (Monitoring + Eval + Integration)
```

---

## What Each Component Does

| Component | One-Line Description |
|-----------|---------------------|
| **Hard Verifiers** | Parse YAML, check required fields (fast, no LLM) |
| **AGENT.md Schema** | Defines agent structure (name, description, skills, model, prompt) |
| **Action Commands** | `set_name`, `add_skill`, `write_prompt`, `check_score`, `submit` |
| **Observation** | Task, current spec state, score breakdown, feedback |
| **Skill Registry** | Available skills + curriculum (1 skill → 2 skills → ...) |
| **Test Cases** | Single-skill first, then multi-skill, then complex |
| **Fast Judge** | Claude Sonnet: 5-dim quality scoring (90% of steps) |
| **Multi-Component Reward** | Hard gates + judge scores + anti-hack penalties |
| **Environment** | OpenEnv reset/step/state with 2-tier evaluation |
| **Goose Runner** | Real execution at steps 3, 6, 9 (10% of steps) |
| **Calibration Tracker** | Detect drift between fast judge and real execution |
| **Curriculum Controller** | Progress: 1 skill → 2-3 skills → 3-5 skills → 5+ skills |
| **GRPO Trainers** | H100 full + T4 Unsloth 4-bit LoRA |
| **Adversarial Designer** | Generate test cases targeting policy weaknesses |
| **Anti-Hacking Checks** | Penalize empty specs, over-engineering, format hacks |

---

## Dependencies

```
1 (Verifiers) ← None
2-4 (Models) ← None
5 (Skills) ← None
6 (Tests) ← 5 (curriculum phase 1)
7 (Judge) ← 2, 6
8 (Reward) ← 1, 7
9 (Environment) ← 1-8
10 (Goose) ← 2
11 (Calibration) ← 7, 10
12 (Tests) ← 1-11
13 (Deploy) ← 1-12
14-22 (Training) ← All P0
```

---

## Anti-Hacking Patterns

| Hack | How We Prevent It |
|------|-------------------|
| **Empty specs** | Hard gate: must have prompt >50 chars |
| **Over-engineering** | Penalty: >10 skills or wrong model tier |
| **Format only** | Judge scoring + real execution validation |
| **Repetitive patterns** | Sample outputs + diversity metric |
| **Judge exploitation** | Real execution calibration (10%) |

---

## Checklist

### P0 - Critical Path (Week 1)
- [x] 1. Hard Verifiers (YAML, fields, format) ✅ `server/verifiers.py`
- [x] 2. AGENT.md Schema ✅ `models.py` with `AgentSpec.to_markdown()`
- [x] 3. Action Commands ✅ `models.py` with `ActionCommand` enum
- [x] 4. Observation extension ✅ `models.py` with POMDP structure
- [x] 5. Skill Registry (single-skill first) ✅ `server/skills.py`
- [x] 6. Test Cases (easy, single-skill) ✅ `server/tasks/scenarios.py` (7 scenarios)
- [x] 7. Fast Judge (Claude Sonnet) ✅ Integrated in reward system
- [x] 8. Multi-Component Reward ✅ `server/rewards/reward.py`
- [x] 9. OpenEnv Environment ✅ `server/environment.py`
- [x] 10. Goose Runner (steps 3, 6, 9) ✅ `server/runtime/goose.py`
- [x] 11. Calibration Tracker ✅ Part of reward system
- [x] 12. Unit Tests ✅ 136 tests pass
- [x] 13. Deploy to HF ✅ Files uploaded to https://huggingface.co/spaces/Kaviya-M/meta-agent-gym (Docker building)

### P1 - MVP (Week 2)
- [x] 14. GRPO Trainer (H100) ✅ `training/grpo_trl.py`
- [x] 15. Unsloth Trainer (T4) ✅ `training/grpo_unsloth.py`
- [x] 16. Curriculum Controller ✅ `training/curriculum.py` with auto-advancement
- [x] 17. Medium/Hard Test Cases ✅ Phases 2-4 in `scenarios.py`
- [x] 18. Adversarial Designer ✅ `server/adversarial.py` with 7 strategies (15 templates)
- [x] 19. Anti-Hacking Checks ✅ Penalties in `reward.py`
- [x] 20. Monitoring Dashboard ✅ `training/monitoring.py` with component plots
- [x] 21. Evaluator (before/after) ✅ `training/evaluation.py`
- [x] 22. Integration Tests ✅ `tests/test_training.py`

### P2 - Polish
- [x] 23. Investigation Tools ✅ `check_score`, `inspect_example` in environment
- [x] 24. More Test Cases ✅ 21 scenarios across all domains
- [x] 25. Documentation ✅ README.md complete with theme alignment
- [ ] 26. Demo Video ❌ Not created
