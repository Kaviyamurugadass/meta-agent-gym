# AGENTSMITH-RL: Meta-Learning for Agent Design

> **Input:** Task Description → **Output:** AGENTS.md (Optimal Agent Specification)

A self-improving RL system that learns to design agents for any task using a two-tier evaluation approach (fast proxy judge + periodic real execution validation).

---

## Overview

**Core Idea:** Instead of learning to perform tasks, the policy learns to *design agents* that perform tasks.

| Traditional Agent Learning | AgentSmith-RL |
|---------------------------|---------------|
| Learn to execute tasks | Learn to design agents |
| Action: tool/function call | Action: generate agent spec |
| Reward: task completion | Reward: agent quality |

**What gets generated:**
```yaml
# AGENTS.md - Agent Specification
name: security-review-agent
description: Analyzes code for security vulnerabilities
tools:
  - code-scanner
  - dependency-checker
  - cve-search
model: claude-sonnet-4-20250514
flow:
  - step: scan_code
    tool: code-scanner
  - step: check_dependencies
    tool: dependency-checker
  - step: lookup_cves
    tool: cve-search
  - step: generate_report
    output: security_report.md
constraints:
  max_tokens: 100000
  timeout: 300
  sandbox: true
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENTSMITH-RL SYSTEM                            │
│                  Input: Task Description → Output: AGENTS.md            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          RL TRAINING LOOP                               │
│                                                                         │
│  ┌─────────────┐     ┌─────────────────────────┐     ┌───────────────┐  │
│  │    STATE     │     │      RL POLICY          │     │   ACTION      │  │
│  │ (Task +      │────▶│   Qwen3-1.7B + LoRA    │────▶│  AGENTS.md    │  │
│  │ constraints) │     │                         │     │ (Agent Spec)  │  │
│  └─────────────┘     └───────────┬─────────────┘     └──────┬────────┘  │
│                                  │                          │           │
│                                  │                          │           │
│                                  ▼                          ▼           │
│                        ┌──────────────────────┐   config file           │
│                        │  MEMORY / BUFFER     │◀────────────────────────│
│                        │                      │                         │
│                        │ - past trajectories  │                         │
│                        │ - rewards            │                         │
│                        │ - policy updates     │                         │
│                        └──────────┬───────────┘                         │
│                                   │                                     │
│                                   │ reward signal                       │
│                                   ▼                                     │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
                                    │
                                    │ EVALUATION LAYER
                                    │
┌───────────────────────────────────▼─────────────────────────────────────┐
│                                                                         │
│   ┌──────────────────────────────┐                                      │
│   │      FAST JUDGE (Default)    │                                      │
│   │   (Claude Sonnet)            │                                      │
│   │                              │                                      │
│   │  - evaluates AGENTS.md       │                                      │
│   │  - checks structure/tools    │                                      │
│   │  - gives quick reward        │                                      │
│   └──────────────┬───────────────┘                                      │
│                  │                                                      │
│                  │ most steps (~90%)                                    │
│                  ▼                                                      │
│           reward (fast proxy)                                           │
│                                                                         │
│   ┌──────────────────────────────┐                                      │
│   │   REAL EXECUTION (Periodic)  │                                      │
│   │      (Every N steps)         │                                      │
│   │                              │                                      │
│   │  - run agent in runtime      │                                      │
│   │  - execute tools + flow      │                                      │
│   │  - get real outcome          │                                      │
│   └──────────────┬───────────────┘                                      │
│                  │                                                      │
│                  ▼                                                      │
│           reward (ground truth)                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Two-Tier Evaluation System

### Fast Judge (90% of training steps)

**Purpose:** Provide quick, cheap feedback on agent spec quality

**What it evaluates:**
| Component | Criteria |
|-----------|----------|
| **Tool Relevance** | Tools match task requirements |
| **Flow Logic** | No circular dependencies, clear sequence |
| **Model Choice** | Model capacity matches task complexity |
| **Efficiency** | Minimal tool count, no redundancy |
| **Safety** | Error handling, timeouts, sandboxing |
| **Structure** | Valid YAML, required fields present |

**Reward Components:**
```python
fast_reward = {
    "tool_relevance": 3.0,      # Are tools appropriate?
    "flow_correctness": 2.0,    # Is the workflow logical?
    "efficiency": 1.0,          # Is it optimized?
    "safety": 2.0,              # Error handling?
    "structure": 1.0,           # Valid format?
}
# Range: -5.0 to +9.0
```

**Performance:**
- Time: ~2-5 seconds per evaluation
- Cost: ~$0.01-0.02 per evaluation

### Real Execution (10% of training steps)

**Purpose:** Validate that fast judge rewards correlate with actual performance

**Process:**
1. Sample K agents from recent batch
2. Run each in sandboxed runtime
3. Collect real metrics:
   - Task success rate
   - Token usage
   - Execution time
   - Error frequency
4. Compare fast judge prediction vs reality
5. Calibrate judge if drift detected

**Calibration Logic:**
```python
if abs(fast_reward - real_reward) > DRIFT_THRESHOLD:
    # Judge is misaligned - recalibrate
    update_judge_prompt(misaligned_examples)
    adjust_reward_weights(correction_factor)
```

---

## Training Algorithm

```python
VALIDATION_INTERVAL = 10  # Real execution every N steps
DRIFT_THRESHOLD = 1.0     # Recalibrate if diff exceeds this

def train_step(state, step_num):
    # Generate agent spec
    agent_spec = policy.generate(state)

    # Fast evaluation (most steps)
    if step_num % VALIDATION_INTERVAL != 0:
        reward = fast_judge.evaluate(state, agent_spec)
        memory.store(state, agent_spec, reward)
        policy.update(memory)
        return reward

    # Real execution (periodic)
    else:
        # Get fast judge prediction
        fast_reward = fast_judge.evaluate(state, agent_spec)

        # Run agent for real
        real_reward = run_agent_real(agent_spec, state.task)

        # Store for calibration
        calibration_data.append({
            "spec": agent_spec,
            "fast": fast_reward,
            "real": real_reward
        })

        # Check for drift
        if detect_drift(calibration_data):
            recalibrate_judge()

        # Use real reward for policy update
        memory.store(state, agent_spec, real_reward)
        policy.update(memory)
        return real_reward
```

---

## Training Economics Comparison

| Metric | Original (Full Execution) | Two-Tier (Fast + 10% Real) |
|--------|---------------------------|----------------------------|
| Cost per step | $1-10 | $0.065 |
| Time per step | 5-30 min | 15 sec |
| 1000 steps | $5,000, 3 weeks | $65, 4 hours |

**10-25x cheaper, 100x faster**

---

## AGENTS.md Schema

```yaml
# Agent Specification Schema
name: string                    # Unique agent identifier
description: string             # What this agent does
version: string                 # Spec version

# Model Configuration
model:
  name: string                  # Model name (claude-sonnet-4, qwen-14b, etc.)
  temperature: float            # Sampling temperature (0-1)
  max_tokens: int              # Token limit

# Tools Configuration
tools:
  - name: string
    config:                     # Tool-specific config
      endpoint?: string
      timeout?: int
      retries?: int

# Agent Flow
flow:
  - step: string               # Step name
    tool: string               # Tool to use
    input:                     # Input mapping
      from: string             # Source (previous step or user)
      transform?: string       # Optional transformation
    output?: string            # Output destination
    on_error?: string          # Error handling strategy

# Constraints
constraints:
  max_steps: int               # Maximum steps per execution
  max_tokens: int              # Total token budget
  timeout: int                 # Execution timeout (seconds)
  sandbox: boolean             # Run in sandboxed environment
  allowed_operations: []       # Whitelist of operations

# Metadata
metadata:
  author: string               # Generated by (policy name)
  created: timestamp           # Creation time
  task_type: string            # Task category this targets
```

---

## Hyperparameters

| Parameter | Range | Default | Impact |
|-----------|-------|---------|--------|
| `VALIDATION_INTERVAL` | 10-50 | 20 | Higher = cheaper, more drift |
| `REAL_EXECUTION_SAMPLES` | 5-20 | 10 | More = better calibration |
| `FAST_JUDGE_MODEL` | haiku/sonnet | sonnet | Speed vs accuracy |
| `DRIFT_THRESHOLD` | 0.5-1.5 | 1.0 | Recalibration sensitivity |
| `GRPO_NUM_GENERATIONS` | 4-16 | 8 | Rollouts per update |
| `LEARNING_RATE` | 1e-6 - 1e-4 | 5e-5 | Training stability |

---

## MVP Roadmap

### Phase 1: Proof of Concept (1 week)
**Goal:** Validate that policy can learn to generate good specs

- [ ] Define narrow task domain (web scraping or data processing)
- [ ] Create fast judge prompt with evaluation rubric
- [ ] Generate synthetic task/spec pairs for pre-training
- [ ] Implement basic GRPO loop with fast judge only
- [ ] Manual validation of generated specs

**Deliverables:**
- Working fast judge
- Synthetic training dataset (100+ tasks)
- Policy that generates valid specs

### Phase 2: Real Execution Integration (1-2 weeks)
**Goal:** Validate fast judge predictions

- [ ] Build sandboxed runtime for agent execution
- [ ] Implement periodic real execution pipeline
- [ ] Collect calibration data (fast vs real rewards)
- [ ] Measure correlation and tune calibration
- [ ] Implement drift detection

**Deliverables:**
- Working runtime executor
- Calibration pipeline
- Correlation metrics (>0.7 target)

### Phase 3: Full Training Loop (2-3 weeks)
**Goal:** End-to-end training with both eval modes

- [ ] Complete GRPO training with two-tier eval
- [ ] Curriculum controller for difficulty progression
- [ ] Judge auto-recalibration
- [ ] Evaluation on held-out test tasks
- [ ] Documentation and demos

**Deliverables:**
- Trained policy checkpoint
- Evaluation benchmarks
- Demo with 5+ task types

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Judge drift** | Track correlation, recalibrate when drops below 0.7 |
| **Spec vs execution gap** | Include runtime constraints in spec evaluation |
| **Task distribution shift** | Curriculum with adversarial tasks |
| **Cost overruns** | Start with 5% real execution, increase gradually |
| **Poor correlation** | Improve judge prompt, add more calibration data |

---

## Related Work

| Paper/Project | Relevance |
|---------------|-----------|
| **RLAIF (Constitutional AI)** | AI feedback for training signal |
| **WebRL** | Fast proxy + real validation pattern |
| **Reflection (Meta)** | Two-stage evaluation approach |
| **Training Compute Optimal** | Optimal synthetic/real data ratio |
| **kube-sre-gym** | GRPO + curriculum + judge pattern |

---

## Technical Stack

### Training
- **Framework:** TRL 0.29.0+ (GRPO implementation)
- **Model:** Qwen3-1.7B + LoRA (BF16)
- **Inference:** vLLM (colocated mode)
- **Hardware:** H100 (80GB) or A100 (40GB)

### Evaluation
- **Fast Judge:** Claude Sonnet 4 (via Anthropic API)
- **Runtime:** Custom sandboxed executor
- **Tools:** Function-calling interface with mock/real backends

### Storage
- **Buffer:** ReplayDB or custom SQLite
- **Checkpoints:** HuggingFace Hub
- **Calibration Data:** Parquet files with metadata

---

## Next Steps

1. **Define task domain** - Pick one narrow domain to start
2. **Design fast judge prompt** - Create evaluation rubric
3. **Generate synthetic data** - Create initial training pairs
4. **Implement GRPO loop** - Basic training with fast judge
5. **Build runtime executor** - Sandboxed execution environment
6. **Integrate calibration** - Two-tier evaluation system

---

## Status

| Component | Status |
|-----------|--------|
| Concept Design | ✅ Complete |
| Fast Judge Design | 🔄 In Progress |
| Runtime Executor | ⏳ Not Started |
| Training Loop | ⏳ Not Started |
| Calibration System | ⏳ Not Started |

---

*Created: 2025-04-22*
*Last Updated: 2025-04-22*
