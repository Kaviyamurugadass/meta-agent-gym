# Onsite Plan — 2026-04-25 / 2026-04-26

**Goal**: Make the "Generate with Trained Model" button on the HF Space dashboard work end-to-end with a real trained model — so judges can type a task description and watch the trained model build an agent live.

**Window**: 48 hours of HF compute credits on 25th + 26th.

**Core principle**: The infrastructure is already in place. All that's missing is the trained adapter + a couple of deployment tweaks. Don't over-engineer — aim to deliver a working demo, not a new feature.

---

## Dependency stack that must be ready

Before anything else on day 1, confirm:

- [ ] You have HF compute credits (or a promo code) applied to your account
- [ ] You can log in to `huggingface_hub` with `hf auth login`
- [ ] The GitHub repo is in sync with `main`
- [ ] Colab has the latest notebook (pull from `main`)

---

## Phase 1 — Scale training (Day 1, ~4 hours of compute)

**Goal**: produce a trained LoRA adapter that's meaningfully better than the 4-step smoke run already on `main`.

### Step 1.1 — Train with more steps (2-3 hours)

Use the existing `notebooks/train_colab.ipynb` — it runs end-to-end. Edit cell 5 params:

```python
NUM_EPOCHS = 3            # was 1
DATASET_EPISODES = 32     # was 8
NUM_GENERATIONS = 4       # was 2 (T4 memory permitting — drop to 2 if OOM)
```

Resulting gradient steps: `3 × 32 × 4 / (1 × 4) = 96`. That's **24× the current run** — enough to produce observable behavioral change in a 0.5B model.

If you have access to an **A10G or L40** (HF Pro credits for Spaces upgrade to GPU runtimes), re-run the same notebook there instead of Colab T4 — faster, larger batches, still the same script.

### Step 1.2 — Verify real training happened

After cell 5 finishes, confirm:

- [ ] No red traceback in any cell
- [ ] `training/grpo-unsloth-output/training_summary.json` has `"real_training": true`
- [ ] `training/grpo-unsloth-output/adapter_model.safetensors` exists (~17 MB)
- [ ] Final reward on the last training step is higher than the first (check cell 5 logs)

### Step 1.3 — Push adapter to HF Model Hub

From the Colab notebook, add a cell at the end:

```python
from huggingface_hub import HfApi, login
login(token="hf_...")  # your token with write permission

api = HfApi()
api.create_repo(repo_id="Kaviya-M/meta-agent-gym-grpo-qwen2.5-0.5b",
                repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="training/grpo-unsloth-output",
    repo_id="Kaviya-M/meta-agent-gym-grpo-qwen2.5-0.5b",
    repo_type="model",
)
```

This keeps the adapter off the main repo (it's in `.gitignore`) but publicly accessible.

---

## Phase 2 — Wire inference into the HF Space (Day 1 evening OR Day 2 morning, ~2 hours)

**Goal**: Make `/generate` actually respond with trained output on the live Space.

### Step 2.1 — Add inference deps to the Space runtime

The Space currently ships without `transformers`, `peft`, `torch` in core deps. Add them. **Two options:**

**Option A (recommended — faster iteration):** Edit `pyproject.toml` and move the `train` extras into core so they install in the Space build:

```toml
dependencies = [
    "openenv-core==0.2.1",
    "pydantic>=2.0",
    "numpy",
    "fastapi",
    "uvicorn[standard]",
    "httpx",
    "websockets",
    # Added for /generate endpoint:
    "transformers>=4.51.3,<5.0.0",
    "peft>=0.10.0",
    "torch>=2.0",
    "sentencepiece",
]
```

**Option B:** Add them in `Dockerfile` as an explicit `pip install` step after the `uv sync`. Only do this if Option A causes lock conflicts.

Commit and push. HF Space will rebuild automatically (~8-12 min).

### Step 2.2 — Upload the trained adapter to the Space

Two methods — pick one:

**Method A — ship adapter inside the repo (simpler, but bloats repo):**

```bash
# Only do this if the adapter is small enough (<50MB is OK)
# Check .gitignore — you need to force-add it
git add -f training/grpo-unsloth-output/adapter_config.json
git add -f training/grpo-unsloth-output/adapter_model.safetensors
git add -f training/grpo-unsloth-output/tokenizer*
git add -f training/grpo-unsloth-output/training_summary.json
git commit -m "deploy: trained LoRA adapter for /generate endpoint"
git push
```

**Method B — download adapter at Space boot (cleaner, recommended):**

Edit `server/inference_service.py` — change the `_load()` method to download from HF Model Hub at first call. Replace the adapter-loading line:

```python
# Before:
model = PeftModel.from_pretrained(base, str(self.adapter_path))

# After:
from huggingface_hub import snapshot_download
adapter_dir = snapshot_download(
    repo_id=os.getenv("META_ADAPTER_REPO", "Kaviya-M/meta-agent-gym-grpo-qwen2.5-0.5b"),
)
model = PeftModel.from_pretrained(base, adapter_dir)
```

Then override the `adapter_available` property to always return True once deps are OK (it'll download lazily).

### Step 2.3 — Verify the endpoint

```bash
# GET /generate/status should say available=true, adapter_available=true, deps_available=true
curl https://kaviya-m-meta-agent-gym.hf.space/generate/status

# POST /generate should return a spec
curl -X POST https://kaviya-m-meta-agent-gym.hf.space/generate \
  -H 'Content-Type: application/json' \
  -d '{"task_description":"Build an agent that scrapes product prices from e-commerce sites"}'
```

If the JSON response has `status: ok` with a populated `spec` and `actions`, you're done.

### Step 2.4 — Test in the dashboard UI

1. Open https://kaviya-m-meta-agent-gym.hf.space/web/
2. "Generate with Trained Model" card should show green **Ready** badge
3. Pick a scenario, type a task description, click **Generate Agent**
4. Watch the UI replay the trained model's actions step-by-step
5. Verify the reward breakdown populates and final score is non-zero

---

## Phase 3 — Update documentation & pitch (Day 2, ~1 hour)

**Goal**: the pitch matches the demoable reality.

- [ ] Update `docs/competition/TRAINING_EVIDENCE.md` — add the scaled run's real numbers (new `report.json`)
- [ ] Update `README.md` — same
- [ ] Update `docs/competition/HUGGINGFACE_BLOG.md` — same
- [ ] Update `docs/competition/PITCH.md` (spoken script) — remove the "honest limitation" paragraph about trained-LoRA not being wired (it now IS wired)
- [ ] Update `docs/competition/SLIDES.md` — slide 8 ("Honest Scope") → change to show the live Generate button instead of the placeholder note

---

## Time budget summary

| Phase | Task | Estimated time | Compute cost |
|---|---|---:|---|
| 1 | Scale training run | 3h | ~$3 HF credits (Colab-T4 equivalent) |
| 1 | Upload adapter to HF Model Hub | 15 min | free |
| 2 | Add deps to pyproject + Dockerfile rebuild | 30 min | Space rebuild (free) |
| 2 | Wire adapter download at boot | 45 min | free |
| 2 | End-to-end test dashboard flow | 15 min | Space runtime (free-tier ok) |
| 3 | Update docs | 1h | free |
| **Total** | | **~6h active work** | **~$3 credits** |

**Leaves ~42 hours of the credit window for contingencies, bigger runs, or pitch rehearsal.**

---

## Fallback plan (if something breaks)

### If Colab training fails onsite
- Use a different Colab runtime (switch to L4 or A100 if available via credits)
- Fall back to the already-committed 4-step adapter — it exists, it's sentinel-verified, and the pitch is already honest about scale limits
- Worst case, present with the current run + honest "small scale" framing (pitch already handles this gracefully)

### If the Space rebuild fails after adding deps
- Revert the `pyproject.toml` change, push again
- Keep the Space running the existing green build
- Record a screencast of /generate working *locally* and link it from the pitch

### If the adapter produces garbage output
- 0.5B models with limited training can produce invalid JSON — the UI handles this gracefully (`alert("Generate failed")`) but it's a bad demo
- Workaround: implement an output-sanity check in `inference_service.py` — if parse fails or spec invalid, fall back to showing the raw output with a "model is still learning" message
- Longer-term fix: upgrade to Qwen2.5-1.5B and retrain

---

## Post-submission commits to avoid

Per hackathon rules (minimum-requirements doc): **changes after the submission deadline will not be considered for judging**. So:

- [ ] Confirm the submission deadline **before** starting Phase 1
- [ ] Treat deadline as a hard stop — even if training finishes late, don't push partial results
- [ ] Lock in a "good-enough" commit before deadline; you can still demo onsite from it

---

## One question to resolve with the hackathon organizers onsite

**Do the HF credits include Space GPU runtime upgrades, or only Jobs/Training compute?**

If Space GPU upgrades are included, we can upgrade `Kaviya-M/meta-agent-gym` to A10G ($1/hr) and serve inference much faster (~2s per generation instead of ~20s). If not, cpu-basic is fine — just slower during demo.
