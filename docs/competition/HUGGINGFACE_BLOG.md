# Meta-Agent Gym: I trained a model to design AI agents — and Goose caught it cheating

**OpenEnv Hackathon 2026 submission · [Kaviya-M/meta-agent-gym](https://huggingface.co/spaces/Kaviya-M/meta-agent-gym)**

---

The numbers came back gorgeous. **10/10 successful trajectories. +51.80 mean reward per episode.** I almost shipped it.

Then I plugged in Goose — the open-source agent runtime — to actually *run* the AGENT.md files my trained model was producing. Every single one was empty. Pure `noop → submit` with no name, no skills, no prompt. The "100% successful" model had learned to do nothing.

I spent that evening reading `server/rewards/reward.py` line by line. The bug was on line 158, one operator wrong, that turned a -5 penalty into a +5 bonus. GRPO had exploited it within four gradient steps. I fixed it, added a regression test, and kept building.

**The point of this post isn't the bug.** The point is: my in-environment metrics couldn't see what was happening. Goose could. That's the whole reason RLVR with independent verifiers exists — and it justified the entire three-tier architecture in one shot.

---

## Why I built it

I'm a developer. I use Cursor. I use Claude Code. Under the hood, those are agents. And the gap between a useful agent and a useless one is mostly *design choices* — what skills it has, which model tier it runs on, what its system prompt actually says.

Most RL-for-LLM work is about teaching a model to *solve* tasks. I wanted to teach a model to **design the agents that solve tasks**. That's the meta-skill, and it's the one I do every day at work.

So: an OpenEnv environment where a small language model picks structured commands to assemble an `AGENT.md` spec, and gets graded on it.

## How it works

### The action space — 12 commands, not free text

Free-form generation was tempting (it's how humans write `AGENT.md` by hand), but credit assignment for GRPO is brutal on long token sequences. I went with discrete commands:

```
set_name · set_description · add_skill · remove_skill · set_model
write_prompt · add_tools · set_memory · set_max_turns
check_score · inspect_example · submit
```

Each command has clean semantics. The reward responds per step. The agent can investigate (`check_score`) before committing (`submit`). A typical complete trajectory is six commands ending in submit.

### Three-tier verification

| Tier | What it catches | Cost |
|---|---|---|
| Hard verifiers (5; 3 are gates) | YAML structure, required fields, prompt length, model name, skills format | $0 |
| Judge (6 dimensions) | Skill selection, description quality, workflow clarity, model fit, best practices, efficiency | ~$0.01 / step (Claude Sonnet in prod, heuristic in dev) |
| Goose runtime (offline harness) | Whether the resulting AGENT.md actually executes | ~$0 (Claude Code CLI provider) |

The first three hard verifiers act as **gates** — fail any one and the entire step reward zeroes. Defense in depth, in case the judge gets gamed (which, as it turned out, mattered).

### Anti-hack penalties

The policy *will* try to game the reward. Four explicit deterrents:

| Penalty | Value | When |
|---|---:|---|
| empty_spec | −5.0 | Prompt < 50 chars or required fields missing |
| over_engineered | −0.5 | More than 10 skills, or `opus` for a task that needs `sonnet` |
| repetitive | −0.3 | Same action twice in a row |
| regression | −0.15 | Breaking a previously-passing check |

(The `−5.0` here is exactly the value the sign-flip bug turned into a `+5.0` bonus. Hold that thought.)

## What I trained

- **Base model:** `Qwen/Qwen2.5-0.5B`
- **Adapter:** LoRA r=16, 8.8M of 502M params trainable (1.75%)
- **Quantization:** 4-bit NF4 via Unsloth
- **Loss:** DAPO (asymmetric clipping, outperforms vanilla GRPO per the DAPO paper)
- **Hardware:** Google Colab T4 (15.6 GB VRAM)
- **Scale:** 1 epoch × 8 episodes × 2 generations = 4 gradient steps

A sentinel file `training/grpo-unsloth-output/training_summary.json` with `"real_training": true` is written **only after `trainer.train()` returns**. So judges have a tamper-evident signal that training actually ran. The Colab notebook is structured to fail loudly on any dependency error, rather than silently writing mock output (which my first three runs did, before I rewrote it).

The scale here is small. Onsite HF compute credits at the hackathon finale (Apr 25-26) extend the step count without changing the pipeline.

## The bug hunt, in detail

After training, my saved trajectories looked like this:

```json
{
  "task_id": "cr_easy_001",
  "success": true,
  "total_reward": 51.80,
  "steps": [
    {"action": {"command": "noop"}, "reward": 7.40},
    {"action": {"command": "noop"}, "reward": 7.40},
    ...
    {"action": {"command": "submit"}, "reward": 7.40}
  ]
}
```

Every one of the 10 saved trajectories was identical: noop seven times, submit, +7.40 per step. Total +51.80. The environment was reporting these as 100% successful.

I wired up Goose, gave it the resulting `AGENT.md` files, and pointed it at concrete tasks (extract a price from an HTML page, count rows in a CSV, find emails in text). All three failed instantly. There was nothing in the AGENT.md files to execute.

Reading the reward code, line 158:

```python
total = core + bonus + progress - penalty - regression - sum(anti_hack_penalties.values())
```

The `anti_hack_penalties` dict contains values like `{"empty_spec": -5.0}` — stored as **negative** numbers in config because they represent a deduction. The intent: subtract −5.0 from total. The code: `total - (−5.0) = total + 5.0`. The minus sign in front of `sum(...)` flipped the whole thing. Empty specs were worth +5 per step.

The fix:

```diff
- total = ... - sum(anti_hack_penalties.values())
+ total = ... + sum(anti_hack_penalties.values())
```

One operator. Then a parameterised regression test in `tests/test_meta_agent_reward.py` so the gate fires under either HYBRID or ADDITIVE mode if empty-spec ever scores positive again.

The size of the fix, in numbers: the same empty-spec-submitting policy went from **+51.80 per episode to −32.20 per episode**. An 84-point swing. On a reward surface where a competent heuristic scores +21.33, the bug had been paying empty-spec play more than honest play.

## Onsite postscript — and then it broke again

Apr 25, hackathon onsite. I had HF compute, the fixed reward function, regression tests, and a few hours before the mentor round. Plenty of time to retrain at proper scale and have a working adapter to demo.

First retrain: `reward = 0.0` across every step. Loss = 0.0. No gradient signal at all.

I added debug printing to `reward_fn` and saw what the model was actually generating:

```
<think>
Okay, let's tackle this problem step by step. To generate a valid agent spec,
I need to emit six actions: set_name, set_description, add_skill, write_prompt,
set_model, submit. First, the agent's name. Since the domain is analysis...
[256 tokens of correct reasoning, then truncated]
```

Qwen3 has a dual-mode chat template — a "thinking" mode that emits `<think>...</think>` reasoning blocks, and a "fast" mode that goes straight to output. The default is thinking mode, and the model was using the entire 256-token completion budget on chain-of-thought before ever reaching the JSON. The system prompt's *"respond ONLY with the JSON"* was being ignored because Qwen3's reasoning mode is sticky.

Fix: append `/no_think` to the system prompt. Qwen3 recognizes it as a control token that disables reasoning for the request.

After the fix:

| Step | Reward | Notes |
|---|---:|---|
| 1 | **8.55** | Some completions valid, others fail |
| 2 | **17.55** | Almost approaching the heuristic baseline of 21.33 |

Empty `<think></think>` blocks at the start of each completion (the model still emits the tags, just nothing inside), then a clean JSON array of six actions. Reward signal flowing for the first time.

Same methodology as the sign-flip discovery: when reward is mysteriously zero, **print what the model is actually generating**. The fix is usually a small token away — but you have to look first.

## Results

### Baselines (20 easy-tier episodes each)

| Policy | Success | Mean reward |
|---|---:|---:|
| Random (uniform over commands) | 0% | 0.00 |
| Competent heuristic (fills each field, then submits) | 100% | 21.33 |
| Expert benchmark (mixed difficulty) | 20/21 | 16.79 |

Random at 0% is the load-bearing result: the hard-verifier gates genuinely prevent reward hacking by chance — random actions can't bluff their way through. The competent heuristic proves the environment is *reachable* with positive reward, the necessary precondition for GRPO to learn anything.

### Per-component reward signal (50 evaluation episodes)

Across 50 evaluation episodes (`monitoring/colab_results/report.json`), per-component last-10 means consistently exceed the overall means:

| Component | Overall mean | Last-10 mean | Δ |
|---|---:|---:|---:|
| Per-step reward `total` | 1.83 | 3.05 | +67% |
| `description_quality` | 0.31 | 0.51 | +65% |
| `workflow_clarity` | 0.23 | 0.38 | +67% |
| `has_required_fields` | 0.34 | 0.57 | +67% |

Episode-level aggregate trend: **+0.62 reward per episode**. The decomposed reward produces learnable signal across multiple dimensions — the necessary structure for GRPO to compute meaningful per-dimension advantage.

## What I learned

**Goose validated the whole architecture in one shot.** The three-tier RLVR design wasn't just elegant on paper — it caught a real reward hack that the judge tier (Sonnet, scoring 6 quality dimensions) had no way to see. Independent execution is non-negotiable for any RL system that uses an LLM as part of its reward.

**Silent fallbacks are worse than failures.** My first three Colab runs "succeeded" with placeholder numbers because the notebook swallowed dependency errors and wrote mock output. I rewrote it to fail loudly on any setup or training error; the resulting real run immediately exposed two genuine bugs (an `Observation.done` AttributeError and a ChatML template issue on Unsloth 4-bit tokenizers) that the silent path had been hiding for days.

**The heuristic beats the expert on easy tasks.** My expert benchmark uses hand-crafted optimal trajectories per scenario, but its mean reward is pulled down by harder tiers. On easy-only, a simple field-filling heuristic scores higher. This reframes "expert" as a *mixed-difficulty ceiling*, not a per-scenario ceiling — useful signal for curriculum design.

**Compute credit windows matter for hackathons.** A free Colab T4 trains a 0.5B model in about 20 minutes — enough to validate the pipeline, not enough for capability gains. Concentrating compute credits in the final 48 hours of a hackathon (as this one does) is the right incentive: build real infrastructure first, train at scale last. That's why I'm holding the bigger Qwen3-1.7B run for onsite.

## Hackathon theme alignment

The official OpenEnv Hackathon themes are: #1 Multi-Agent, #2 Long-Horizon Planning, #3 World Modeling (with #3.1 Professional Tasks and #3.2 Personalized Tasks sub-themes), #4 Self-Improvement, and #5 Wild Card. Here's where I think this work honestly fits:

- **Primary: Theme #5 — Wild Card.** Meta-agent design doesn't fit cleanly into themes 1-4 (one agent, 7-step episodes, no browser/API tool ecosystem, not personalized). Theme #5 was designed for out-of-box submissions that meaningfully add value to LLM training on a task that hasn't been explored, and that's exactly what this is.
- **Secondary alignment: Theme #3 — World Modeling (broadly).** The env has a real POMDP structure (hidden state = "what's a good spec for this task") and investigation commands (`check_score`, `inspect_example`) for belief-updating. This is the broader Theme #3 fit; **not** the #3.1 Professional Tasks sub-theme, whose examples are direct tool/API ecosystems we don't operate in.
- **Architectural ambition: Theme #4 — Self-Improvement** (built, not yet demonstrated at scale). The adversarial generator and curriculum controller exist in code (`server/adversarial.py`, `training/curriculum.py`), but the 4-step Colab run is far below what's needed to show recursive skill amplification. Future-work paths: VCRL ([arxiv 2509.19803](https://arxiv.org/html/2509.19803v1)) and Self-Evolving Curriculum ([arxiv 2505.14970](https://arxiv.org/pdf/2505.14970)).
- **Underlying technique:** RLVR throughout — hard verifiers as gates, judge components for what hard checks miss, Goose as the third independent tier.

I'm flagging the Self-Improvement framing as "ambition, not current claim" deliberately. Calling it primary would overclaim.

## Honest scope

There are two distinct rollout sets in this repo and they describe different things:

- [`data/colab_trained/`](https://github.com/Kaviyamurugadass/meta-agent-gym/tree/main/data/colab_trained) (10 trajectories) — the actual trained LoRA's output during the Colab run. These are what showed the empty-spec collapse.
- [`monitoring/colab_results/report.json`](https://github.com/Kaviyamurugadass/meta-agent-gym/blob/main/monitoring/colab_results/report.json) (50 evaluation episodes) — uses the heuristic policy as a placeholder for the trained adapter at inference time. Wiring adapter inference into evaluation rollout collection is the planned first task for the onsite training window.

Other things to be straight about:

- **The Goose harness covers 3 Phase 1 (single-skill) tasks.** Expanding to Phase 2-4 multi-skill tasks is future work; the harness API is generic enough to accept new tasks without changes.
- **The trained adapter on disk is the pre-fix one** — it learned to emit empty specs because that's what scored well at the time. The "Generate" tab in the dashboard reproduces this behaviour live; I left it enabled deliberately so judges can see the bug.
- **The "self-improvement" track** (adversarial task generation, curriculum auto-escalation) exists in code but hasn't been demonstrated end-to-end. The 4-step Colab run is far below what's needed to see curriculum escalate. Specific future-work paths I'd attempt next: VCRL ([arxiv 2509.19803](https://arxiv.org/html/2509.19803v1)) and Self-Evolving Curriculum ([arxiv 2505.14970](https://arxiv.org/pdf/2505.14970)).

## Try it

- **Live demo:** [huggingface.co/spaces/Kaviya-M/meta-agent-gym](https://huggingface.co/spaces/Kaviya-M/meta-agent-gym)
- **Colab notebook:** [`notebooks/train_colab.ipynb`](https://github.com/Kaviyamurugadass/meta-agent-gym/blob/main/notebooks/train_colab.ipynb)
- **GitHub repo:** [github.com/Kaviyamurugadass/meta-agent-gym](https://github.com/Kaviyamurugadass/meta-agent-gym)

The dashboard has two tabs:
- **Build Step-by-Step** — pick a scenario, issue commands manually, watch the multi-component reward update live
- **Generate from Description** — type a task, the trained model emits commands. Currently the pre-fix adapter, so you'll see the empty-spec collapse for yourself

Reproduce the bug locally in 10 seconds:

```bash
git clone https://github.com/Kaviyamurugadass/meta-agent-gym
cd meta-agent-gym
python scripts/demo_reward_fix.py
```

You'll see the +7.40 → −4.58 → 0.00 reward swing for an empty spec across the buggy / fixed-additive / fixed-hybrid configs.

---

Built for the OpenEnv Hackathon 2026.
