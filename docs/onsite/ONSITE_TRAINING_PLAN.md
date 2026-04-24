# Onsite Training Plan -- Apr 25-26, 2026

What to do tomorrow when HF compute credits become available.

## TL;DR

- Use **Colab Pro+ on A100** if HF gives generic compute credits
- Train **`Qwen/Qwen3-1.7B`** (already in the notebook dropdown)
- Scale to **3 epochs x 50 episodes x 4 generations** (~600 gradient steps)
- Expected runtime: **1-2 hours on A100**
- Sentinel-verify, then re-run the Goose harness on the new model
- **After training finishes**: follow [`docs/competition/ONSITE_PLAN.md`](../competition/ONSITE_PLAN.md) Phases 2 & 3 — wire the adapter into the HF Space `/generate` endpoint and update the pitch docs. Do NOT skip this; the pitch references the live Generate flow.

## Cost summary (so there are no surprises)

| Component | Cost |
|---|---|
| Qwen3-1.7B model download | $0 (Apache 2.0, no HF token needed) |
| Unsloth + TRL libraries | $0 |
| Colab Pro+ subscription (if used) | ~$50/month — covered by HF credits if granted |
| HF Inference API | Not used during training |
| HF Hub upload (optional, post-train) | Free, but needs an HF write token |

You don't need an HF token to **train**. You only need one if you choose to **push the trained adapter** back to HF Hub afterwards.

## Step-by-step

### Step 0 -- understand what HF gave you (5 min)

Log in to HF, check what's enabled:

| Granted | Where to look | Use it for |
|---|---|---|
| HF Pro / Spaces GPU upgrade | Spaces -> Settings -> Hardware | Spaces inference + small training |
| AutoTrain credits | huggingface.co/autotrain | UI-based fine-tuning (no notebook) |
| Inference Endpoints credits | huggingface.co/inference-endpoints | **Inference only**, NOT training |
| Direct cloud / HF Jobs credits | wherever HF tells you | One-off training runs on managed GPUs |

**If unsure or none of the above:** fall back to Colab Pro+ ($50/month, A100 access). The existing notebook works as-is.

### Step 1 -- pick environment

| Path | When to choose |
|---|---|
| Colab Pro+ + A100 | **Default.** Reuses the notebook that already works. |
| HF Space with GPU upgrade | Only if you want everything on HF (more setup) |
| AutoTrain | Only if HF specifically gave AutoTrain credits |

### Step 2 -- change the model in cell 5

Open `notebooks/train_colab.ipynb`, scroll to cell 5 (the GRPO training cell). The dropdown:

```python
MODEL_ID = 'Qwen/Qwen2.5-0.5B' #@param ['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen3-1.7B', 'Qwen/Qwen2-0.5B']
```

Pick **`Qwen/Qwen3-1.7B`** from the dropdown.

### Step 3 -- scale up training

Same cell 5, change defaults:

```python
NUM_EPOCHS = 3              # was 1
NUM_GENERATIONS = 4         # was 2 (better GRPO variance)
DATASET_EPISODES = 50       # was 8 (more learning signal)
```

Math: 3 x 50 x 4 = ~600 gradient steps.

### Step 4 -- run cells in order

`1 -> 2 -> 5 -> 6 -> 7 -> 8`

Skip cell 3 (baseline collection) and cell 4 (expert benchmark) if those are already in the repo (they are -- `data/baseline/{random,heuristic}` exist).

Expected runtime on A100: **1-2 hours**.

### Step 4-alt -- CLI path (if you don't want the notebook)

If the onsite GPU is a local Linux box (not Colab) and you'd rather run training directly, `training/grpo_unsloth.py` now ships an L4 / A100 scale-up command in its module docstring:

```bash
uv sync --extra train --extra unsloth

uv run python training/grpo_unsloth.py \
    --model-id Qwen/Qwen3-1.7B \
    --per-device-train-batch-size 2 \
    --num-generations 4 \
    --gradient-accumulation-steps 4 \
    --max-seq-length 2048 \
    --num-epochs 3 \
    --dataset-episodes 50
```

Notes:
- `--per-device-train-batch-size 2` and `--max-seq-length 2048` are the A100-friendly bumps over the Colab T4 defaults (batch 1, seq 1024). Drop them back down if you hit OOM on L4.
- `--loss-type dapo` is the default already — don't override unless you know why.
- The script writes the same sentinel (`training/grpo-unsloth-output/training_summary.json` with `real_training: true`) so Step 5 verification works identically.
- For a dry run without a GPU: `--dry-run`.

### Step 5 -- verify the sentinel

After cell 5 finishes, open `training/grpo-unsloth-output/training_summary.json`. It must show:

```json
{
  "real_training": true,
  "model": "Qwen/Qwen3-1.7B",
  "dataset_episodes": 50,
  "num_epochs": 3,
  "num_generations": 4
}
```

If `real_training: false` or the file is missing -> **training did not complete**, check logs.

### Step 6 -- download the adapter, run the Goose harness

Back on your Windows laptop:

```powershell
# After downloading training/grpo-unsloth-output/ from Colab to local
python -m evaluation.goose_execution --smoke
```

If 3/3 still pass with the trained model -> you have a defensible
*"trained model produces working AGENT.md files"* claim. Update the pitch
to reflect this.

If 0-1/3 pass -> the trained adapter is no better than random; the pitch
stays as-is (infrastructure + finding + fix), no false "trained model
works" claim.

## Failure modes + mitigations

| Failure | Fix |
|---|---|
| OOM on A100 with Qwen3-1.7B | `--max-seq-length 1024`, `--num-generations 2` |
| Unsloth complains about Qwen3 support | `pip install -U unsloth` (Apr 2026 supports Qwen3) |
| Training takes >3 hours | Cut `DATASET_EPISODES` to 20, `NUM_EPOCHS` to 2 |
| No HF compute granted at all | Fall back: existing Qwen2.5-0.5B Colab path is the shipped baseline -- pitch survives |

## What to update in the README after onsite

If the trained Qwen3-1.7B run is successful and produces non-empty specs:

1. Update `### What the GRPO run actually produced` (around line 309) with the new numbers
2. Remove the *"current eval rollouts use the competent heuristic as a placeholder for the trained LoRA at inference time"* limitation note (lines 114, 329-333)
3. Add a *"Trained model: N/3 Goose tasks pass"* line under the RLVR case study (around line 198)
4. Update the architecture diagram caption to remove the *"deferred to onsite"* qualifier on the real-execution tier

If the trained run fails or produces weak results: do NOTHING to the README. The current narrative survives ("infrastructure + finding + fix; trained adapter scaled-up onsite, results in submission artifacts").

## Defensive posture

The pitch already works without the onsite training succeeding. Anything from
onsite is upside, not load-bearing. Treat it that way. Do not over-promise
during the pitch on the back of an unfinished run.

---

## After training finishes — don't miss these pieces

The training step is ~30% of the onsite work. The rest is in
[`docs/competition/ONSITE_PLAN.md`](../competition/ONSITE_PLAN.md):

| Phase | What | Why you can't skip it |
|---|---|---|
| **Phase 2.1** — Add deps to Space | Edit `pyproject.toml` to include `transformers`, `peft`, `torch` in core deps | `/generate` endpoint is wired but Space doesn't have these installed yet — endpoint will keep returning `deps_missing` until you fix this |
| **Phase 2.2** — Push adapter | Either commit the adapter into the repo (if <50MB) OR push to HF Model Hub as `Kaviya-M/meta-agent-gym-grpo-qwen3-1.7b` and have the Space download it at boot | Without this, the "Generate with Trained Model" button never turns green |
| **Phase 2.3** — Smoke-test `/generate` | `curl` the endpoint + open the dashboard, click Generate, verify a full replay works | The pitch's slide 6-7 story depends on this flow working during the demo |
| **Phase 3** — Update docs | `README.md`, `TRAINING_EVIDENCE.md`, `HUGGINGFACE_BLOG.md`, `PITCH.md` all need updating with the real trained-model numbers | If numbers in docs contradict what's in the repo, judges mark it as fabricated |

**Total additional time after training: ~2 hours.** Don't let training eat the whole 48h window — budget for deploy + verify + doc update from Day 2 afternoon onwards.

---

## Related onsite docs

- [`COMPLETE_TESTING_GUIDE.md`](COMPLETE_TESTING_GUIDE.md) — pre-onsite validation checklist
- [`RESULTS_GUIDE.md`](RESULTS_GUIDE.md) — how to read + report the trained-model results
- [`../competition/ONSITE_PLAN.md`](../competition/ONSITE_PLAN.md) — full deploy pipeline (training → Space → docs)
- [`../competition/PITCH.md`](../competition/PITCH.md) — spoken script the trained-model output needs to back up
