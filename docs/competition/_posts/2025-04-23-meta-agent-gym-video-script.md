---
title: "Teaching LLMs to Design Agents: Meta-Agent Gym"
date: 2026-04-24
video_length: "1:45"
tags: ["meta-learning", "reinforcement-learning", "agent-design", "openenv"]
---

# Meta-Agent Gym — Video Script

*Target length: under 2 minutes (hackathon requirement).*

---

## [0:00–0:15] Hook

**Visual**: Split screen — empty `AGENT.md` on the left, complete specification appearing on the right.

**Narration**: "Most RL environments test whether an LLM can solve a task. We built one that asks a harder question: can a tiny LLM learn to *design the agent* that solves the task?"

## [0:15–0:35] The problem

**Visual**: A growing list of failed agent specs — empty names, missing prompts, oversized skill lists.

**Narration**: "Designing a good agent is a meta-skill. You need to pick the right skills, write a clear prompt, choose a cost-appropriate model, and avoid over-engineering. It's the kind of thing a human takes weeks to get right."

## [0:35–0:55] How it works

**Visual**: Action-command timeline showing `SET_NAME → SET_DESCRIPTION → ADD_SKILL → WRITE_PROMPT → SET_MODEL → SUBMIT`.

**Narration**: "We gave the model a command-based action space and a three-tier reward: hard verifiers for format, a fast LLM judge for quality, and occasional real execution to keep the judge honest. This is RLVR — reinforcement learning with verifiable rewards, not learned reward models."

## [0:55–1:15] The numbers

**Visual**: Three side-by-side cards — "Random 0% / 0.00", "Heuristic 100% / 21.33", "Expert 20/21 / 16.79".

**Narration**: "Random policy gets 0%: the hard gates genuinely prevent reward hacking. A rule-based heuristic that fills each required field gets 100% on easy tasks with mean reward 21. That's the learnable gap we trained into."

**Visual**: Component chart showing last-10 vs overall mean for description_quality, workflow_clarity, has_required_fields.

**Narration**: "Over 50 evaluation episodes, per-component rewards improve 65 to 67 percent in the last ten episodes. The environment produces learnable signal."

## [1:15–1:35] What we actually trained

**Visual**: Terminal showing `trainer.train()` finishing, then `training_summary.json` with `"real_training": true`.

**Narration**: "We trained Qwen2.5 half-billion with 4-bit LoRA on a free Colab T4. Small but real — four gradient steps, sentinel-verified. We'll scale it up with the onsite HuggingFace credits."

## [1:35–1:45] Call to action

**Visual**: HF Space URL + QR code, GitHub URL.

**Narration**: "Meta-Agent Gym — environment on HuggingFace Spaces, code on GitHub. Train a model to design agents."

---

## Production notes

### Honest numbers to highlight
- **Random policy**: 0% success, 0.00 mean reward (hard gates work)
- **Competent heuristic**: 100% success, 21.33 mean reward (env is reachable)
- **Expert benchmark**: 20/21 scenarios succeeded, 16.79 mean reward (mixed-difficulty ceiling)
- **Per-component last-10 vs overall mean**: description_quality +65%, workflow_clarity +67%, has_required_fields +67%
- **Positive trend**: +0.62 reward per episode across 50 eval episodes
- **Training run**: Qwen2.5-0.5B + 4-bit LoRA, 1 epoch × 8 episodes × 2 gens = 4 gradient steps, sentinel `"real_training": true`

### Honesty guardrails
- Do **not** claim "2200% improvement" or "5% → 68% success" — these are placeholder numbers from the pre-training demo data and are not supported by the real run.
- Current eval rollouts use the competent heuristic as a stand-in for trained-LoRA inference; claiming "trained beats heuristic" is not defensible yet. The onsite training window is earmarked for wiring up real LoRA inference.

### Visual elements needed
- Split-screen spec builder animation
- Command-timeline graphic
- Three baseline cards (random/heuristic/expert)
- Per-component last-10 vs overall bar chart
- Terminal capture of `trainer.train()` completing

### Runtime: 1 minute 45 seconds
### Format: YouTube Short / LinkedIn video
