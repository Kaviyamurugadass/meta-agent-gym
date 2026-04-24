# Meta-Agent Gym: An OpenEnv Environment for Teaching LLMs to Design Agents

**OpenEnv Hackathon 2026 submission — [Kaviya-M/meta-agent-gym](https://huggingface.co/spaces/Kaviya-M/meta-agent-gym)**

## The idea

Most RL environments for LLMs test a model's ability to *solve* tasks.
Meta-Agent Gym tests a different skill: can a small LLM learn to *design agents
that solve tasks*?

The policy takes a natural-language task description ("Build an agent that
scrapes product prices from e-commerce sites") and emits a complete AGENT.md
specification — name, description, skills, model choice, system prompt — that
runs across Claude Code, Goose, Copilot, and any framework following the
[Agent Skills Open Standard](https://anthropic.com).

It's meta-learning at small scale: teaching an LLM to write the instructions
that instruct other LLMs.

## How it works

### The action space

We chose a **command-based action space** rather than free-form text:

```
SET_NAME, SET_DESCRIPTION, ADD_SKILL, WRITE_PROMPT, SET_MODEL,
ADD_TOOLS, CHECK_SCORE, INSPECT_EXAMPLE, SUBMIT, NOOP
```

Token-efficient, grader-friendly, easy to validate. Each step the agent picks
one command and fills in its arguments. A typical successful trajectory:
`SET_NAME → SET_DESCRIPTION → ADD_SKILL → WRITE_PROMPT → SET_MODEL → SUBMIT`.

### Three-tier verification (RLVR)

The reward function composes three layers, each catching different failure modes:

| Tier | What | Frequency | Cost |
|---|---|---|---|
| Hard verifiers | YAML parse, required fields, format | Every step (100%) | ~$0 |
| Fast judge | Claude Sonnet quality scoring (5 dims) | 90% of steps | ~$0.01 |
| Real execution | Goose runtime actual test | Steps 3, 6, 9 (10%) | ~$1–10 |

Hard gates prevent format hacks. Fast judge provides quality signal. Real
execution validates that the judge isn't getting gamed.

### Anti-hacking penalties

Reward hacking is the first thing the policy tries. We ship with four explicit
deterrents:

- `empty_spec`: **−5.0** (harshest, discourages the NOOP → SUBMIT shortcut)
- `over_engineered`: **−0.5** (> difficulty-appropriate skill count)
- `repetitive`: **−0.3** (same command twice)
- `regression`: **−0.15** (undoing a passing check)

## What we trained

- **Base model**: `Qwen/Qwen2.5-0.5B`
- **Adapter**: LoRA r=16, 8.8M of 502M params trainable (1.75%)
- **Quantization**: 4-bit NF4 via Unsloth
- **Loss**: DAPO (asymmetric clipping — outperforms vanilla GRPO per the DAPO paper)
- **Hardware**: Google Colab T4 (15.6 GB VRAM)
- **Scale**: 1 epoch × 8 episodes × 2 generations = 4 gradient steps

A sentinel file `training/grpo-unsloth-output/training_summary.json` with
`"real_training": true` is written only after `trainer.train()` returns, giving
judges a tamper-evident signal that the run actually happened. The
[Colab notebook](https://github.com/Kaviyamurugadass/meta-agent-gym/blob/main/notebooks/train_colab.ipynb)
is deliberately structured to fail loudly on any dependency or training error,
rather than silently generating placeholder output.

The scale here is small but real. Onsite HF compute credits at the hackathon
finale will extend the step count without changing the pipeline.

## Results

### Baselines

We compared three policies across 20 easy-tier episodes each:

| Policy | Success | Mean reward |
|---|---:|---:|
| Random (uniform over action space) | 0% | 0.00 |
| Competent heuristic (fills each field then submits) | 100% | 21.33 |
| Expert benchmark (mixed difficulty) | 20/21 | 16.79 |

The random baseline at 0% is instructive: hard-verifier gates genuinely prevent
reward hacking, so random actions can't bluff their way through. The competent
heuristic proves the environment is reachable with >0 reward, so GRPO has
learning signal to bootstrap from.

### Per-component learning signal (50 eval episodes)

Over 50 evaluation episodes, later-episode per-component means exceed overall
means — a positive signal that the environment's decomposed reward produces
learnable gradient across multiple dimensions:

| Component | Overall mean | Last-10 mean |
|---|---:|---:|
| Per-step reward `total` | 1.83 | 3.05 (+67%) |
| `description_quality` | 0.31 | 0.51 (+65%) |
| `workflow_clarity` | 0.23 | 0.38 (+67%) |
| `has_required_fields` | 0.34 | 0.57 (+67%) |

Episode-level aggregate reward trend: **+0.62 per episode**.

### Honest limitation

This submission's evaluation rollouts use the heuristic policy as a placeholder
for the trained LoRA at inference time — we didn't finish wiring adapter
loading into rollout collection before the submission window. This means the
*improvement curves above reflect the environment's reward structure applied to
a competent heuristic*, not the trained Qwen2.5-0.5B adapter's inference-time
behaviour. Wiring adapter inference is the planned first task for the onsite
training window (2026-04-25/26).

What the submission does demonstrate clearly:

1. A working end-to-end OpenEnv environment with 20+ scenarios across 4 difficulty tiers
2. A real GRPO run (sentinel-proven) on Qwen2.5-0.5B + 4-bit LoRA
3. A non-trivial reward surface: random fails completely, competent rule-based play succeeds, expert benchmark exists as a ceiling
4. A three-tier RLVR verification system with hard gates, fast judge, and real execution calibration
5. Anti-hacking penalties that shape policy toward the intended task

## Try it

- 🚀 [Live demo on HF Spaces](https://huggingface.co/spaces/Kaviya-M/meta-agent-gym)
- 📓 [Colab training notebook](https://github.com/Kaviyamurugadass/meta-agent-gym/blob/main/notebooks/train_colab.ipynb)
- 💻 [GitHub repo](https://github.com/Kaviyamurugadass/meta-agent-gym)

## What we learned

A few things that surprised us:

**The judge got gamed — execution caught a sign-flip bug.** Our Goose
integration exposed a reward hack on the first trajectories we pointed it at.
The judge-only tier reported 68% success, but Goose revealed the policy was
emitting `noop → submit` and producing empty specs. Digging in, we found the
root cause: a sign-flip on line 111 of the reward computer. The
`anti_hack_empty_spec` value is stored as `-5.0` in config, but the total
formula was `total = ... - sum(anti_hack_penalties.values())` — subtracting a
negative, so a -5 *penalty* became a +5 *bonus*. Empty-spec trajectories
scored +7.4/step. GRPO obediently found the hack and collapsed. We flipped
the operator to `+ sum(...)` (penalties are already signed), added a
parameterised regression test (`tests/test_meta_agent_reward.py`) that fails
if empty-spec ever receives positive reward under HYBRID or ADDITIVE mode,
and the three-tier verification system caught a bug a PR review missed.
Without Goose validation, we'd have shipped a model that looked trained but
wasn't.

**The heuristic beats the expert on easy tasks.** Our expert benchmark uses
"optimal" action sequences for each scenario but its mean reward is pulled
down by harder tiers. On easy-only, a simple field-filling heuristic scores
higher. This reframes "expert" as a mixed-difficulty ceiling, not a
per-scenario ceiling — useful signal for curriculum design.

**Silent fallbacks are the real enemy.** Our first three Colab runs "succeeded"
with placeholder numbers because the notebook swallowed dependency errors and
wrote mock output files. We rewrote it to fail loudly on any setup or training
error; the resulting real run exposed two actual bugs (an
`Observation.done` AttributeError and a ChatML template issue on Unsloth 4-bit
tokenizers) that the silent path had been hiding.

**Compute credit windows matter.** A free-tier T4 trains this model in ~20
minutes — enough to validate the pipeline but not enough for capability gains
on a model this small. Allocating compute credits to the final 48h of a
hackathon (as this one does) is the right incentive for teams to build real
infrastructure first and train at scale last.
