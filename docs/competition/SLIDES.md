# Slide Deck Outline — Meta-Agent Gym

**For:** OpenEnv Hackathon 2026 · 3-minute pitch
**Pairs with:** [`PITCH.md`](PITCH.md) (spoken script)
**Format:** 9 slides · ~20s per slide average

> Copy these into PowerPoint / Google Slides / Keynote. Each slide has short bullets plus a visual suggestion — keep text minimal, let your voice carry the story.

---

## Slide 1 — Title *(5s)*

**Meta-Agent Gym**
*Teaching small LLMs to design AI agents*

- OpenEnv Hackathon 2026 · Kaviya-M
- *Visual*: Clean title card + Meta PyTorch / OpenEnv logos

---

## Slide 2 — The Question *(20s)*

**Most RL environments ask: can an LLM solve this task?**

**We ask a harder one:**
- Can a small LLM learn to *design the agent* that solves the task?
- Meta-learning at small scale
- Teaching a model to write the instructions that instruct other models

*Visual*: Split diagram — "solve task" (left) vs "design the agent" (right)

---

## Slide 3 — The Environment *(25s)*

**Input:** task description
**Output:** complete AGENT.md (name, description, skills, model, system prompt)

**Action space — command-based, not free text:**
- `SET_NAME`, `SET_DESCRIPTION`, `ADD_SKILL`, `WRITE_PROMPT`, `SET_MODEL`, `SUBMIT`
- Token-efficient · grader-friendly · hard to hack

*Visual*: Animated command timeline building an AGENT.md

---

## Slide 4 — Three-Tier Verification (RLVR) *(30s)*

**Reward hacking is the first thing a policy tries. We catch it in 3 layers:**

| Tier | Coverage | Cost |
|---|---|---|
| Hard verifiers (YAML, fields, format) | 100% | ~$0 |
| LLM judge (5 quality dimensions) | 90% | ~$0.01 |
| Real execution (Goose runtime) | 10% | ~$1–10 |

Plus anti-hacking penalties: empty spec **–5.0**, over-engineered **–0.5**.

*Visual*: 3-layer funnel with coverage % labels

---

## Slide 5 — Baselines Tell the Story *(30s)*

**20 easy-tier episodes each:**

| Policy | Success | Mean reward |
|---|---:|---:|
| Random | **0%** | 0.00 |
| Competent heuristic | 100% | **21.33** |
| Expert (mixed difficulty) | 20/21 | 16.79 |

- Random 0% proves hard gates **prevent reward hacking**
- Heuristic 100% proves the environment is **reachable**

*Visual*: 3 large number cards, side-by-side

---

## Slide 6 — What We Trained *(25s)*

**Qwen2.5-0.5B + 4-bit LoRA · GRPO with DAPO loss · Colab T4**

- 1 epoch × 8 episodes × 2 generations = **4 gradient steps**
- Sentinel-verified: `training_summary.json` with `"real_training": true`
- Written *only* after `trainer.train()` returns

*Visual*: Terminal screenshot showing the sentinel JSON

---

## Slide 7 — Measured Learning Signal *(30s)*

**50 evaluation episodes — last-10 vs overall mean:**

| Component | Overall | Last-10 | Δ |
|---|---:|---:|---:|
| Total reward | 1.83 | 3.05 | +67% |
| description_quality | 0.31 | 0.51 | +65% |
| workflow_clarity | 0.23 | 0.38 | +67% |
| has_required_fields | 0.34 | 0.57 | +67% |

**Positive trend: +0.62 reward per episode.**

*Visual*: `monitoring/colab_results/component_curves.png`

---

## Slide 8 — Honest Scope *(20s)*

**What's real:** pipeline validated, run sentinel-verified, baselines clean.

**What's next (onsite 25–26 with HF credits):**
- Wire trained LoRA into inference (placeholder today)
- Scale the run
- Push adapter to HF Model Hub

**Our notebook fails loudly**, not silently — earlier runs produced placeholder numbers and we removed those paths.

*Visual*: Split card — "Real" ✅ / "Onsite" 🚀

---

## Slide 9 — Close *(15s)*

**Meta-Agent Gym**
*Teaching small LLMs a high-value meta-skill: designing agents.*

- 🚀 Live on HF Spaces: `Kaviya-M/meta-agent-gym`
- 💻 GitHub: `Kaviyamurugadass/meta-agent-gym`
- *QR codes visible*

**Thank you.**

*Visual*: Two big QR codes, dark-mode friendly

---

## Delivery tips

- **Don't read slides.** They're anchors — your voice tells the story.
- **Pause on Slide 5.** Let the 0% / 21.33 / 16.79 sink in before moving on. That's the most credible moment.
- **Slide 7 is your evidence slide.** If a judge is going to ask you one tough question, it'll come from here — know the numbers cold.
- **Slide 8 is a weapon.** Volunteering limitations before they're asked is a storytelling move, not an apology.

---

## Assets you'll need

- [ ] Screenshot of `training/grpo-unsloth-output/training_summary.json` (sentinel) → Slide 6
- [ ] Export of `monitoring/colab_results/component_curves.png` → Slide 7
- [ ] QR codes for HF Space URL + GitHub URL → Slide 9
- [ ] Meta PyTorch / OpenEnv logos → Slide 1
- [ ] **Theme**: dark background recommended for projector visibility; sans-serif body font (Inter / IBM Plex Sans / Roboto)
