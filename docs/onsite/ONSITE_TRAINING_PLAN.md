# Onsite Training Plan -- Apr 25-26, 2026

What to do tomorrow when HF compute credits become available.

## TL;DR

- Use **Colab Pro+ on A100** if HF gives generic compute credits
- Train **`Qwen/Qwen3-1.7B`** (already in the notebook dropdown)
- Scale to **3 epochs x 50 episodes x 4 generations** (~600 gradient steps)
- Expected runtime: **1-2 hours on A100**
- Sentinel-verify, then re-run the Goose harness on the new model

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
