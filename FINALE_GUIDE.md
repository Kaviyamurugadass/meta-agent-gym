# Using `openenv-r2-kit` — Template Guide

> A reference for anyone using this template to build a domain-specific OpenEnv environment with GRPO training. Works for the April 25 hackathon finale, for ongoing research, or for any OpenEnv + RL project.

---

## What this template gives you

**Pre-built (don't touch — works end-to-end):**

| Layer | Files | What it does |
|---|---|---|
| OpenEnv scaffold | `server/app.py`, `openenv.yaml`, `Dockerfile`, `server/Dockerfile` | FastAPI via OpenEnv `create_app`; HF Spaces deploy-ready |
| Environment lifecycle | `server/environment.py` | `reset/step/state` on `openenv.core.Environment` ABC, structured `[START]/[STEP]/[END]` logging |
| Rule engine framework | `server/rules/engine.py` | Hard + soft violations, pluggable rules, 3 built-in guards |
| Reward framework | `server/rewards/reward.py` | Additive / multiplicative / hybrid modes, decomposed breakdown |
| Task generator | `server/tasks/generator.py` | Scenario selection + domain randomization |
| Client + agent | `client.py`, `inference.py` | HTTP/WS client, LLM inference with triple-fallback JSON parser |
| Training pipeline | `training/grpo_trl.py`, `training/grpo_unsloth.py` | GRPO via TRL (H100) or Unsloth 4-bit LoRA (T4 Colab) |
| Trajectory + eval | `training/trajectory.py`, `training/evaluation.py`, `training/rollout_collection.py` | Save/load episodes, compute metrics, fill before/after table |
| Benchmark | `training/benchmark.py` | Expert-trajectory runner — proves env isn't gameable |
| Dashboards | `static/index.html` + OpenEnv's auto-generated UI | Two UIs: custom glassmorphism + form-per-action |
| Tests + fixtures | `tests/`, `conftest.py` | 32 passing tests, shared fixtures for domain code |
| Scripts | `scripts/deploy.sh`, `smoke_test.sh`, `baseline_rollout.sh` | One-command workflows |
| Makefile | 15 targets: install/test/run/train/deploy/smoke | Muscle memory for time pressure |

**Fill-in slots (marked `# DOMAIN:` in the code):**

| What | Where | Time to fill |
|---|---|---|
| Action commands | `models.py` (`ActionCommand` enum) | 10 min |
| Scenarios | `server/tasks/scenarios.py` (`SCENARIOS` list) | 60 min |
| Domain rules | `server/rules/engine.py` (`_domain_rules`) | 45 min |
| Reward components | `server/rewards/reward.py` (`_component_scores`, `_regression_penalty`) | 30 min |
| State transitions | `server/environment.py` (`_execute_action`, `_build_observation`) | 60 min |
| Expert trajectory | `training/benchmark.py` (`EXPERT_TRAJECTORIES`) | 45 min |
| Sample fixtures | `data/sample/` | 15 min |
| README slots | `README.md` (headline, track alignment, before/after) | 30 min |

**Total specialization time: ~5 hours** for a complete fill.

---

## Prerequisites

| Requirement | Why | Install |
|---|---|---|
| Python ≥3.10 | Modern typing features | https://python.org |
| `uv` ≥0.11 | Fast dep manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker Desktop | Local container build | https://docker.com |
| `hf` CLI | HF auth + push | `uv run hf auth login` |
| A GPU (training only) | Colab T4 free or rental | — |
| `PYTHONUTF8=1` on Windows | `openenv push` encoding | `setx PYTHONUTF8 1` |

---

## Installation

```bash
git clone <this-template-repo> my-env
cd my-env
uv sync --extra dev
cp .env.example .env         # fill in HF_TOKEN, OPENAI_API_KEY, etc.
```

---

## Verification (~2 minutes — do this BEFORE editing anything)

Every command below must succeed. Fix failures now, not at 3 AM.

```bash
make test               # 32+ tests pass
make validate           # openenv CLI: multi-mode ready
make train-dry          # GRPO dry-run (no GPU) succeeds
make run &              # serve locally on :8000
sleep 3
URL=http://localhost:8000 make smoke     # 5/5 endpoints healthy
```

Expected output:
```
[1/5] GET /health       OK
[2/5] GET /schema       OK
[3/5] POST /reset       OK
[4/5] POST /step        OK
[5/5] GET /state        OK
==> All 5 smoke tests passed.
```

If all pass, the template is healthy. Now specialize it.

---

## Specialization — the 8-step fill

Do these in order. Each step feeds the next. Run `make test` after every step to catch regressions.

> ### ⚠️ READ FIRST — The One Step That Decides Whether You Win
>
> **Step 5 (`_build_observation`) is where templates become submissions.**
>
> The template ships with a placeholder observation that contains only metadata (`task_id`, `step`, `summary="Step 1/10"`, empty `history`). This is deliberately bare — the domain signal goes in on Step 5.
>
> If your final observation still looks like the placeholder (metadata-only, no decision-relevant signal), your agent will behave randomly regardless of how good your reward or training is. **Judges will see this immediately.**
>
> See Step 5 below. Do not skim it.

### Step 1 — `models.py` (10 min)

Extend the `ActionCommand` enum with domain-specific commands.

```python
class ActionCommand(str, Enum):
    # Keep these from the template
    INSPECT = "inspect"
    SUBMIT = "submit"
    NOOP = "noop"
    # Add domain — e.g. for a code auditor:
    EDIT_FILE = "edit_file"
    ADD_FILE = "add_file"
    CHECK_SCORE = "check_score"
```

**Don't remove** `INSPECT`/`SUBMIT`/`NOOP` — the tests rely on them.

Checkpoint: `make test` still passes.

### Step 2 — `server/tasks/scenarios.py` (60 min)

Replace the `SCENARIOS` list with 3–4 real scenarios. Each:
- Has real citations (paper DOI, RFC number, PEP, benchmark URL)
- `expected_findings` lists the ground-truth answers
- Difficulty scaling is **architectural**, not additive:
  - **easy**: single component, direct signal
  - **medium**: 2–3 components, some ambiguity
  - **hard**: cross-file / cross-agent dependencies + red herrings

```python
SCENARIOS = [
    TaskSpec(
        task_id="my_easy_1",
        difficulty="easy",
        problem_statement="Fix the null-pointer bug in X.",
        max_steps=5,
        citations=["https://cwe.mitre.org/data/definitions/476.html"],
        expected_findings={"file": "main.py", "line": 42, "fix": "null check"},
    ),
    # ... medium, hard
]
```

### Step 3 — `server/rules/engine.py` (45 min)

Override `_domain_rules()` to return a list of `Rule` callables. Target 3 hard + 3 soft violations.

```python
class RuleEngine:
    def _domain_rules(self):
        return [self._check_valid_file, self._check_not_redundant, ...]

    def _check_valid_file(self, action, state, task):
        if action.command == ActionCommand.EDIT_FILE:
            if action.args.get("path") not in state.hidden_truth["allowed_files"]:
                return [RuleViolation(
                    severity="hard",
                    category="prerequisite",
                    message=f"File not in repo: {action.args.get('path')}",
                )]
        return []
```

### Step 4 — `server/rewards/reward.py` (30 min)

Override `_component_scores` and `_regression_penalty`:

```python
class RewardComputer:
    def _component_scores(self, action, state, task):
        # Return each component in [0, 1]
        return {
            "correctness": self._fraction_of_checks_passing(state),
            "efficiency": max(0, 1 - state.step / task.max_steps),
            "quality": self._quality_score(state),
            "safety": 1.0 - self._fraction_broken(state),
        }
```

Tune `component_weights` in `RewardConfig` — document **why** each weight. Judges read this.

#### ⚠️ GRPO-friendly reward shape (don't skip this)

Your reward isn't just a score — it's training signal. GRPO computes advantages as `reward - group_mean` across N completions per prompt. If those N rewards all look the same, there's no gradient and training fails silently (flat reward curve).

Four rules to prevent silent training failure:

1. **Continuous, not binary.** `correctness = passing/total` beats `correctness = 1 if all_pass else 0`. Partial progress creates a gradient.
2. **Variance must exist within a group.** If 4 random completions all score the same, GRPO can't learn. Design so different completions produce different rewards.
3. **No collapse to zero.** Pure multiplicative with any-dim-zero will produce `reward = 0` for most random rollouts. Use HYBRID mode (gate on safety only, smooth elsewhere) if this is a risk.
4. **No collapse to ceiling.** If 90% of random completions hit max reward, the task is too easy — GRPO can't differentiate. Tighten constraints.

Quick sanity check after writing `_component_scores`:

```bash
# Run 20 random rollouts. If mean_reward ≈ max_reward or std is near zero,
# your reward is flat — fix before training.
uv run python training/rollout_collection.py --policy random --episodes 20 --output-dir /tmp/variance_check
uv run python training/evaluation.py --input-dirs /tmp/variance_check
# Look for: std_reward > 0.1 × mean_reward  → healthy variance
```

If `std ≈ 0`, revisit rule 1 (switch to continuous) or rule 2 (diversify scenario outcomes).

The existing `make readiness` test catches the worst case (all rewards identical → expert/random ratio < 3x). This sanity check catches the subtler case (non-zero variance but too-narrow to train on).

### Step 5 — `server/environment.py` (60 min) · ⚠️ MOST CRITICAL STEP

This step decides whether your submission is **winning-grade or template-grade**. Get the observation design wrong and no amount of reward tuning or training will save you.

#### 5a. `_execute_action` — domain state transitions

```python
class Environment(OpenEnvBase):
    def _execute_action(self, action):
        # Mutate self._state.hidden_truth + progress_flags based on action
        if action.command == ActionCommand.EDIT_FILE:
            path = action.args["path"]
            self._state.hidden_truth["files"][path] = action.args["content"]
```

Standard — just describe the state machine.

#### 5b. `_build_observation` — DECISION-RELEVANT SIGNAL (do not half-ass)

**The golden question for every field you expose:**

> "Can the agent make an intelligent next-action decision using this field?"

If yes → put it in Observation. If no → move to `State.hidden_truth` (visible to reward/eval only, not agent).

**⛔ What weak observations look like** (what every losing submission ships):

```python
# Just metadata — agent can't act on this
obs = Observation(
    task_id=...,
    step=1,
    summary="Step 1/10",       # ← useless to agent
    history=[...],              # ← supporting, not signal
    reward_breakdown={},        # ← post-hoc, not signal
)
```

**✅ What winning observations look like** (decision-relevant signal per domain):

| Domain | Put in Observation | Put in State.hidden_truth |
|---|---|---|
| Code auditor | file contents (partial), lint output, test pass/fail counts | correct fix, secret test cases, grading rubric weights |
| SRE / ops | CPU %, latency, error rate, alert text, pod status | root cause, injected fault type, target resolution |
| Customer service | customer message, ticket history, sentiment score | true intent category, escalation threshold |
| Email triage | sender, subject, first paragraph, trust score | true priority, labeled category |
| Clinical EHR | chart highlights, lab summaries (with noise), visit history | true diagnosis, ground-truth treatment |
| Finance / trading | price indicators, volume, position summary | true market regime, future return |
| Multi-agent | other agents' public statements, resources visible | other agents' private info, true payoffs |

#### 5c. POMDP discipline

- **State** = the truth (hidden biology, correct code, true fault)
- **Observation** = a *noisy or partial projection* of state (what the agent can realistically know)
- Never leak ground truth into Observation — makes the env trivial, agent never actually learns the domain

#### 5d. The override template

```python
def _build_observation(self, reward, violations):
    obs = super()._build_observation(reward, violations)
    assert self._state is not None and self._task is not None

    # (1) Rich summary — one line the agent can read at a glance
    obs.summary = (
        f"latency={self._state.hidden_truth['latency']:.0f}ms · "
        f"errors={self._state.hidden_truth['error_rate']:.1%} · "
        f"alerts={len(self._state.hidden_truth.get('alerts', []))}"
    )

    # (2) Structured signal — what the agent reasons over
    obs.latest_output = {
        "metrics": {
            "cpu": self._state.hidden_truth["cpu"],
            "memory": self._state.hidden_truth["memory"],
            "latency_p99": self._state.hidden_truth["latency"],
        },
        "recent_logs": self._tail_logs(n=5),   # sampled, not full logs
        "active_alerts": self._render_alerts(),
    }

    # (3) NEVER include ground-truth answer fields:
    # obs.latest_output["root_cause"] = ...        # ❌ leaks
    # obs.latest_output["correct_fix"] = ...       # ❌ leaks
    return obs
```

#### 5e. Noise + partial observability (advanced)

Judges reward realistic observations. Consider:

- **Dropout**: some metrics missing at random (real monitoring is flaky)
- **Noise**: numeric values perturbed within tolerance (real sensors)
- **Delay**: "latest log" is actually 30s stale
- **Sampling**: show first/last N items instead of everything

The template already has `NoiseModel`-friendly state in `State.hidden_truth` — you just inject noise during observation projection.

#### 5f. Self-check before you move on

**Automated guardrail:** `tests/test_observation_quality.py` enforces this. The 4 checks auto-skip for `placeholder_*` scenarios and auto-activate the moment you rename your scenarios to domain names. Run anytime:

```bash
make readiness      # just the observation-quality checks, verbose output
make test           # full suite (readiness checks included)
```

What the checks verify:

- [ ] `latest_output` is not `None` and not empty
- [ ] `summary` contains domain context beyond `"Step N/M"`
- [ ] Ground-truth values from `expected_findings` do NOT appear verbatim in observations
- [ ] Different actions produce different observations (INSPECT vs NOOP → visible delta)

If any fails, the error message points at the exact fix path (this guide section).

**Manual self-check — ask yourself:**

- [ ] Can the agent identify the *problem* from the Observation alone?
- [ ] Can the agent choose between 2 different actions based on Observation differences?
- [ ] If you replaced `latest_output` with `None`, would the env become un-solvable?

All "yes". If not, redesign before Step 6.

#### 5g. Reference: `examples/number_guess/environment.py`

Minimal but correct — shows the pattern in 20 lines:

```python
obs.summary = f"range=[{truth['low']}, {truth['high']}] · last={self._last_feedback}"
obs.latest_output = {
    "feedback": self._last_feedback,
    "current_range": [truth["low"], truth["high"]],
}
```

Hidden: the actual `target` number. Observable: the comparison result + narrowed range. Agent can do binary search from this; can't just read off the answer.

### Step 6 — `training/benchmark.py` (45 min)

Author one expert trajectory per scenario:

```python
EXPERT_TRAJECTORIES = {
    "my_easy_1": [
        Action(command=ActionCommand.INSPECT, args={"file": "main.py"}),
        Action(command=ActionCommand.EDIT_FILE, args={"path": "main.py", "content": "..."}),
        Action(command=ActionCommand.SUBMIT, confidence=0.9),
    ],
    # ... one per scenario
}
```

### Step 7 — `data/sample/` (15 min)

Generate + commit pre-seeded trajectories:

```bash
make baseline            # runs random + heuristic rollouts
```

Commit `data/baseline/*` to the repo. Makes the env immediately useful for offline-RL research — scores utility points.

### Step 8 — `README.md` (30 min)

Fill the placeholder slots:
- Headline metric (`<base-model>` → `<trained-model>`, `<X%>` → `<Y%>`)
- "Hackathon Track Alignment" table (map to actual judging tracks)
- Reward Formula (list your components + weight rationale)
- Before/After Training Metrics table (leave blank until training completes)
- **Training Results section — reward curve images + per-episode table**

After each training run, generate the curve:
```bash
make plots                                          # default: data/trained/
# or with custom labels
uv run python training/plot_rewards.py \
    --input-dir data/trained \
    --output training/plots/run_1.png \
    --title "Run 1: <base-model> + GRPO"
```

Chain 2-3 runs together to tell an iteration narrative (noisy → plateau → good variance). Judges pattern-match "this team iterated" far more than "one clean run." Even failed runs tell a story if captioned well.

Remove: the `> **TEMPLATE** — domain-agnostic` banner.

---

## Development workflow

| Task | Command |
|---|---|
| Install deps | `make install` / `make install-train` / `make install-unsloth` |
| Run locally | `make run` (port 8000) |
| Run tests | `make test` (full) / `make test-cov` (with coverage) |
| Lint | `make lint` / `make format` |
| Validate OpenEnv spec | `make validate` |
| One-shot smoke test | `URL=http://localhost:8000 make smoke` |
| Collect baseline rollouts | `make baseline` |
| Clean caches + outputs | `make clean` |

---

## Training

### Quick dry-run (no GPU)

```bash
make train-dry
# Verifies: reward backend, rollout pipeline, arg parsing
```

### T4 Colab (Unsloth 4-bit LoRA)

```bash
make install-unsloth
make train-unsloth ARGS='--model-id Qwen/Qwen3-0.6B --max-seq-length 1024'
```

Memory-tuned defaults (T4 16GB):
- `per_device_train_batch_size=1`
- `num_generations=2`
- `gradient_accumulation_steps=4`
- `max_seq_length=1024` (drop to 768 if OOM)
- `use_gradient_checkpointing="unsloth"`
- `use_vllm=False` (GRPO conflicts with vLLM)

### H100 / A100 (full TRL GRPO)

```bash
make train-trl ARGS='--model-id Qwen/Qwen3.5-4B --num-generations 8'
```

### Evaluate + fill before/after table

```bash
# Collect trained-model trajectories
uv run python training/rollout_collection.py --policy heuristic --episodes 20 --output-dir data/trained

# Compare vs baseline
uv run python training/evaluation.py \
    --input-dirs data/baseline/random data/baseline/heuristic data/trained \
    --reference data/baseline/heuristic \
    --output data/eval.json
```

Copy the `mean_reward` / `success_rate` / `mean_length` numbers into the README before/after table.

### Expert benchmark

```bash
uv run python training/benchmark.py
# Runs EXPERT_TRAJECTORIES against the env — reports reward, steps, match_ratio.
# Use as the "upper bound" row in your README.
```

### Publish trained model

```bash
uv run hf upload <user>/<env-name>-grpo-qwen3-0.6b training/grpo-unsloth-output
```

---

## Deploy to HF Spaces

```bash
REPO_ID=<user>/<env-name> bash scripts/deploy.sh
```

This runs `openenv validate` + `pytest` + `openenv push` in sequence. Fails loud on any issue.

Verify the live deploy:

```bash
URL=https://<user>-<env-name>.hf.space bash scripts/smoke_test.sh
```

**Which org to deploy to?**
- During prep or iteration: **your personal namespace** (`Kaviya-M/...`)
- Round 2 finale: check the brief — SF convention was `openenv-community/<name>`

---

## Verified gotchas (fixes already in the template)

These cost us hours during dry-run — now pre-solved. Knowing them lets you override safely.

| # | Issue | Fix encoded in template |
|---|---|---|
| 1 | `openenv-core` Python API: `Action/Observation/State` must subclass OpenEnv base classes | `models.py` imports `_OpenEnvAction / _OpenEnvObservation / _OpenEnvState` defensively |
| 2 | `Environment` must subclass `openenv.core.Environment` ABC with abstract `reset/step/state` | `server/environment.py` inherits + implements |
| 3 | `state` is a `@property`, NOT a method | `Environment.state` is decorated `@property` |
| 4 | `SUPPORTS_CONCURRENT_SESSIONS = True` required to use `max_concurrent_envs > 1` | Set at class level in `server/environment.py` |
| 5 | OpenEnv HTTP is **stateless** per request (WebSocket holds sessions) | `step()` auto-resets if state is `None` |
| 6 | `/step` body is wrapped: `{"action": {...}}`, not the action directly | `client.py` + `scripts/smoke_test.sh` handle the wrap |
| 7 | `openenv validate` needs `[project.scripts]` + `main()` function in app | Pre-added: `server:server.app:main` |
| 8 | `openenv push` requires **`server/Dockerfile`** (not just root `Dockerfile`) | Template ships both |
| 9 | `pip install -e .` fails on openenv-base image | Dockerfile uses `uv sync --frozen --no-editable` multi-stage pattern |
| 10 | T4 Colab doesn't support FP8 | `grpo_unsloth.py` defaults to BF16 LoRA |
| 11 | Unsloth GRPO conflicts with vLLM fast inference | `use_vllm=False` pre-set |
| 12 | Windows: `openenv push` needs `PYTHONUTF8=1` | `scripts/deploy.sh` sets it |
| 13 | `uv.lock` must be committed — fresh resolution on HF build is slow/flaky | `uv.lock` tracked |
| 14 | Logging needs `logs/` dir for file handlers | Dockerfile does `mkdir -p /app/env/logs` |

---

## Command reference

### Makefile targets

```
Setup
  install          Install core deps
  install-train    Install training deps (TRL, torch, transformers)
  install-unsloth  Install Unsloth deps (T4 Colab LoRA)

Quality
  test             Run test suite
  test-cov         Run tests with coverage report
  lint             Ruff check
  format           Ruff format
  validate         openenv CLI validate

Run / Serve
  run              Start local dev server
  docker-build     Build Docker image
  docker-run       Run Docker container on :8000

Training
  train-trl        Full GRPO (TRL, H100/A100)
  train-unsloth    Unsloth 4-bit LoRA (T4 Colab)
  train-dry        Training dry-run (no GPU)
  eval             Evaluate trained model
  baseline         Collect baseline rollouts (random + heuristic)

Deploy
  deploy           Push to HF Spaces (needs REPO_ID=user/name)
  smoke            Post-deploy health check (needs URL=...)

Housekeeping
  clean            Remove caches + training outputs
```

### Scripts

| Script | Purpose | Usage |
|---|---|---|
| `scripts/deploy.sh` | Validate + test + push to HF | `REPO_ID=user/name bash scripts/deploy.sh` |
| `scripts/smoke_test.sh` | 5 HTTP endpoints health check | `URL=https://... bash scripts/smoke_test.sh` |
| `scripts/baseline_rollout.sh` | Random + heuristic rollouts + eval | `bash scripts/baseline_rollout.sh` (or `make baseline`) |

---

## Finale-day time budget (April 25 only)

Reference only — use the specialization sequence above as the detailed guide.

| Hours | Focus | Cumulative |
|---|---|---|
| 0.5 | Read theme + sketch domain on paper | 0.5 |
| 5.0 | Fill 8 template slots | 5.5 |
| 2.0 | Baseline rollouts + start training | 7.5 |
| 8–12 | GRPO training (Colab T4) | 15.5–19.5 |
| 1.0 | Evaluate + fill before/after table | 16.5–20.5 |
| 2.0 | Publish model(s) + deploy Space | 18.5–22.5 |
| 3.0 | Demo video + README polish | 21.5–25.5 |
| 2.0 | Pitch prep | 23.5–27.5 |
| 20–25 | Buffer + sleep + on-site fixes | 48 |

**The one rule:** *If you're at hour 30 and reward still isn't moving, stop training and polish what you have.* A well-pitched working env with mediocre training beats a broken training run with no pitch.

### Emergency pivots

| If... | Then... |
|---|---|
| Colab T4 OOM | `--max-seq-length 768`, `--num-generations 1`, `--gradient-accumulation-steps 8` |
| Training reward flat | Check reward variance; reduce `num_generations`, swap to smaller base model |
| OpenEnv spec fails | `make validate` for exact error; compare against `bio-experiment` on HF |
| Docker build fails | Skip Docker locally; `make deploy` uploads and HF builds |
| HF push fails near deadline | Fall back: GitHub repo + local demo video |
| Pitch partner drops out | Solo pitch is fine — judges understand hackathon chaos |

---

## Appendix — what each file does

```
openenv-r2-kit/
├── models.py                     # Pydantic Action/Observation/State/TaskSpec/RewardConfig
├── client.py                     # OpenEnv HTTP client wrapper
├── inference.py                  # LLM episode runner + triple-fallback JSON parser
├── conftest.py                   # Shared pytest fixtures
├── Makefile                      # 15 one-liner targets
├── Dockerfile                    # Root — local `docker build .` convenience
├── .env.example                  # HF_TOKEN, OPENAI_API_KEY, MAX_CONCURRENT_ENVS, etc.
│
├── server/
│   ├── app.py                    # FastAPI via openenv.core.create_app + fallback
│   ├── environment.py            # reset/step/state lifecycle (subclasses OpenEnv ABC)
│   ├── Dockerfile                # Multi-stage uv-based build (per openenv push convention)
│   ├── rules/engine.py           # Hard/soft violation framework
│   ├── rewards/reward.py         # Decomposed + multiplicative + hybrid reward modes
│   └── tasks/
│       ├── generator.py          # Scenario picker + domain randomization
│       └── scenarios.py          # TEMPLATE: scenario library
│
├── training/
│   ├── trajectory.py             # TrajectoryStep/Trajectory/TrajectoryDataset
│   ├── rollout_collection.py     # Random + heuristic policies; CLI
│   ├── evaluation.py             # Online / behavior / fidelity metrics; CLI
│   ├── reward_backend.py         # Local (in-process) vs Remote (HTTP) abstraction
│   ├── benchmark.py              # Expert-trajectory runner
│   ├── grpo_trl.py               # TRL GRPOTrainer — full fine-tune
│   ├── grpo_unsloth.py           # Unsloth 4-bit LoRA — T4 Colab
│   ├── logging.yaml              # Structured [START]/[STEP]/[END] format
│   └── log_filters.py            # DefaultHackathonTagFilter
│
├── tests/
│   ├── test_smoke.py             # Env lifecycle tests (13)
│   └── test_training.py          # Training pipeline tests (19)
│
├── scripts/
│   ├── deploy.sh                 # Validate + test + openenv push
│   ├── smoke_test.sh             # 5-endpoint health check
│   └── baseline_rollout.sh       # Random + heuristic rollout + eval
│
├── static/index.html             # Custom glassmorphism dashboard at /web
├── data/sample/                  # Pre-seeded trajectories (domain fills here)
├── examples/                     # Optional: reference domain fills
└── notebooks/                    # Optional: Colab-ready demo/train/eval
```

---

## When something breaks

1. **Tests fail** → `make test-cov` to see which lines aren't covered; revert last edit
2. **Server won't start** → check logs at `logs/episode.log`; most often a Pydantic field error
3. **`make validate` fails** → compare `pyproject.toml` against template commit log
4. **Deploy fails** → fetch the Dockerfile from your HF Space (`curl https://huggingface.co/spaces/<user>/<name>/raw/main/Dockerfile`) and diff against template
5. **Training diverges** → check reward variance (flat rewards = GRPO can't learn); print per-step reward breakdown

---

**This template has been end-to-end validated: GitHub → HF Spaces → live HTTP smoke test.** The infrastructure works. Specialization is the only remaining variable. Good luck.
