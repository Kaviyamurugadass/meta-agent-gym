# Presentation Checklist — OpenEnv Hackathon 2026

**Event:** Meta PyTorch / OpenEnv Hackathon 2026
**Format:** 3-min pitch + HF Space submission (onsite)
**Training window:** 2026-04-25 / 2026-04-26 (HF compute credits provided onsite)
**HF Space:** https://huggingface.co/spaces/Kaviya-M/meta-agent-gym

---

## Official Minimum Requirements

| # | Requirement | Status | Notes |
|---|---|---|---|
| 1 | OpenEnv (latest release) | ✅ | `pyproject.toml` pins `openenv-core==0.2.1` |
| 2 | TRL/Unsloth training script, Colab-runnable | ✅ | `notebooks/train_colab.ipynb` ran end-to-end on T4. Silent-failure fallbacks removed. |
| 3 | Real loss + reward plots | ✅ | Real plots in `monitoring/colab_results/` (sentinel-verified: `training_summary.json` has `"real_training": true`) |
| 4 | Mini-blog / <2min video / slide deck | ⚠️ | `docs/competition/HUGGINGFACE_BLOG.md` rewritten with real numbers — still needs to be published to HF |
| 5 | HF Space deployment | ✅ | `Kaviya-M/meta-agent-gym` live at https://kaviya-m-meta-agent-gym.hf.space |
| 6 | README with links to all materials | ✅ | `README.md` links HF Space + competition docs |
| 7 | `openenv.yaml` manifest | ✅ | Exists at repo root |
| 8 | Client/server separation | ✅ | `client.py` doesn't import from `server/` |

**Legend:** ✅ done · ⚠️ partial · ❌ blocking

---

## Judging Criteria (weights & current readiness)

| Criterion | Weight | Current state |
|---|---|---|
| Environment Innovation | 40% | Strong — meta-learning angle, command-based action space, three-tier RLVR |
| Storytelling | 30% | Docs rewritten with honest numbers; **pitch script not drafted** |
| Showing Improvement in Rewards | 20% | Real plots + report.json in repo; baseline separation clear; per-component learning signal documented |
| Reward & Training Pipeline | 10% | 5-component RLVR + anti-hacking + real GRPO run verified |

---

## What's been done (commits on `main`)

- `4695acd` — fix `Observation.done` + Colab dependency pins aligned to Unsloth 2026.4.8
- `7f9c324` — fix ChatML template fallback for Unsloth 4-bit tokenizers
- `b65a0ac` — fix heuristic baseline (was NOOP→SUBMIT, now builds valid agent spec)
- `aa6abee` — replace placeholder metrics with real Colab training run; 6 real plots committed

---

## Pre-onsite work (today + tomorrow)

- [x] Fix Colab silent-failure — TRL/Unsloth/xformers install + fail-loud error paths
- [x] Verify training loop connects to `MetaAgentEnv` (dry-run + real run passed)
- [x] Run real GRPO training on Colab T4 (sentinel `"real_training": true`)
- [x] Reconcile claimed numbers across README, TRAINING_EVIDENCE, HUGGINGFACE_BLOG, _posts, REWARD_PIPELINE_EXCELLENCE, competition README
- [x] Draft 3-min pitch script → `docs/competition/PITCH.md`
- [x] Slide deck outline + generated .pptx → `docs/competition/SLIDES.md` + `slides.pptx`
- [~] ~~Publish `HUGGINGFACE_BLOG.md` to HF~~ (skipped — needs HF Pro. Content is ready for HF Space README or slide deck handout)
- [x] **Sanity-run HF Space end-to-end** — dashboard runs full 7-step episode, ends with "Agent Accepted (5.83)" banner, reward breakdown fully populated. Demo-ready.
- [~] ~~Verify `ANTHROPIC_API_KEY` handling~~ (skipped — dashboard test showed hard verifiers + local rubric working without Claude API; judge tier's Claude scoring isn't invoked at runtime)
- [ ] Confirm reward system uses OpenEnv's composable Rubric (vs. custom dict)
- [ ] Rehearse pitch (at least twice, with stopwatch)

## Onsite (25–26 with HF credits)

**Concrete step-by-step plan**: [`docs/competition/ONSITE_PLAN.md`](docs/competition/ONSITE_PLAN.md)

- [ ] **Scale the GRPO run** (3 epochs × 32 episodes × 4 generations = ~96 steps, up from 4)
- [ ] Push trained adapter to HF Model Hub as `Kaviya-M/meta-agent-gym-grpo-qwen2.5-0.5b`
- [ ] Add `transformers`, `peft`, `torch` to `pyproject.toml` core deps (or Dockerfile)
- [ ] Wire adapter loading in `server/inference_service.py` (HF Hub `snapshot_download` at boot)
- [ ] Verify `/generate` endpoint returns real spec + actions (curl test)
- [ ] Demo the "Generate with Trained Model" button end-to-end on the live Space
- [ ] Regenerate plots from scaled run; update docs (README, TRAINING_EVIDENCE, HUGGINGFACE_BLOG)
- [ ] Update pitch to show live trained generation instead of honest-limitation note
- [ ] Final push to HF Space

### Infrastructure already shipped (ready for onsite adapter)

- ✅ `server/inference_service.py` — lazy-loaded LoRA inference with structured error reporting
- ✅ `POST /generate` + `GET /generate/status` endpoints (returns `ok` / `no_adapter` / `deps_missing` / `error`)
- ✅ Dashboard card "Generate with Trained Model" with task-description input + status badge
- ✅ UI auto-replays generated actions through the existing reward pipeline (same scoring as human-driven play)

---

## Non-blocking polish (if time allows)

- [ ] Embed plots in README with one-line captions (partially done)
- [ ] Add reward curve caption explaining what each curve represents
- [ ] Confirm no reserved MCP tool names (`reset`, `step`, `state`, `close`)
- [ ] Replace the `monitoring/*_labeled.png` placeholder plots at repo root with the real ones from `monitoring/colab_results/`

---

## Defensible numbers (for pitch)

These are measured, not fabricated — safe to quote:

- **Baselines (20 episodes, easy tier):** Random 0% / 0.00 · Heuristic 100% / 21.33 · Expert ceiling 16.79
- **Training setup:** Qwen2.5-0.5B + 4-bit LoRA, 8.8M of 502M params trainable, GRPO w/ DAPO loss
- **Run scale (Colab T4):** 1 epoch × 8 episodes × 2 generations = 4 gradient steps
- **Per-component learning (50 eval episodes):** description_quality +65%, workflow_clarity +67%, has_required_fields +67% (last-10 vs overall mean)
- **Positive trend:** +0.62 reward per episode

## Do NOT claim (unsupported by real run)

- "2200% reward improvement" — placeholder from `demo_data` era
- "5% → 68% success rate" — placeholder
- "680% improvement in skill selection" — placeholder
- "R² = 0.89" — fabricated

---

## Live Demo Commands (pitch Q&A, PowerShell-safe)

Keep this terminal open during the pitch. Copy-paste commands are Windows
PowerShell 5.1 compatible (no `&&` -- use `;` or `if ($?)` instead).

### Tier 1 -- "Show me the bug is fixed" (instant)

```powershell
python scripts/demo_reward_fix.py
```

Prints the `+7.40 -> -4.58` reward swing for an empty spec. One screen,
no internet needed. **Primary Q&A demo.**

### Tier 2 -- "Show me the regression tests" (~2 sec)

```powershell
python -m pytest tests/test_meta_agent_reward.py -q -k "empty_spec or sign"
```

5 tests pass in ~1.5 sec. Proves the fix is enforced.

### Tier 3 -- "Run the Goose harness live" (~55 sec)

```powershell
python -m evaluation.goose_execution --smoke
```

Runs 3 hand-written AGENT.md files through the real Goose runtime (via
Claude Code CLI -- no API key needed). Expected: `3/3 tasks passed`.
Requires internet + the `GOOSE_PROVIDER=claude-code` env already
persisted on this machine.

### Tier 4 -- "Show me the policy-level evidence" (~30 sec)

```powershell
python -m training.rollout_collection --policy random --episodes 5 --reward-mode additive --seed 42 --output-dir /tmp/test_run
```

Reproduces the Option B result: `mean_reward=-32.196` for 5 random-policy
episodes under ADDITIVE mode. Historical buggy value: +51.80.

### Tier 5 -- "Show me the writeup"

```powershell
cat data/post_fix/REWARD_FIX_COMPARISON.md
```

Pre-rendered ASCII so raw `cat` renders cleanly in any terminal.

### One-liner combo (PS 5.1 safe)

```powershell
python -m pytest tests/test_meta_agent_reward.py -q; if ($?) { python scripts/demo_reward_fix.py }
```

Runs tests; only runs the demo if tests pass.

---

*Judges' TL;DR:* "A messy but ambitious environment with real training evidence beats a polished but boring one."
