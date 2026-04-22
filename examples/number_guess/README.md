# Example — Number Guess

The simplest possible domain fill: agent guesses a hidden integer in `[1, N]`. Each guess returns "higher", "lower", or "correct". Optimal strategy is binary search.

Shows how each of the 8 template slots gets specialized. Read alongside `FINALE_GUIDE.md` → *Specialization — the 8-step fill*.

## Why this domain

- **Tiny** — all logic fits on one screen per file
- **Clear ground truth** — "the answer is 37"
- **Clear optimal strategy** — binary search; expert trajectory is obvious
- **Teaches POMDP** — agent doesn't see the number; only comparison feedback
- **Catches buggy rewards** — easy to verify binary-search agent gets maximum reward

## Files in this fill

| File | What changed from template | Step in FINALE_GUIDE |
|---|---|---|
| `models.py` | Added `GUESS` command; `args={"value": int}` | 1 |
| `scenarios.py` | 3 difficulty levels: `[1,16]`, `[1,256]`, `[1,10000]` | 2 |
| `rules.py` | Hard: value out of range. Soft: repeating a guess. | 3 |
| `rewards.py` | Components: `correctness` (1.0 on win), `efficiency` (fewer steps = higher), `quality` (narrow search = higher) | 4 |
| `environment.py` | `_execute_action` updates `low`/`high`; `_build_observation` surfaces the comparison result | 5 |
| `benchmark.py` | Expert = binary search trajectory | 6 |

## How to "install" this fill

This folder is **reference only** — don't copy it into the main template as-is. Instead:

1. Read each file alongside the template version it overrides
2. Copy the pattern (not the code) into your domain
3. Run `make test` after each change

The template's tests will still pass with the number-guess fill because the scenarios are additive (3 new placeholder scenarios, existing template ones retained).

## What this example is NOT

- **Not production-grade** — no randomization, no cross-file hard task, no literature citations
- **Not an enterprise simulation** — it's a pedagogical toy
- **Not optimized for winning a hackathon** — real domains (auditor, SRE, EHR) have much richer state spaces

For finale submissions, pick a real-world domain. Use this only to learn the pattern.

## Expected binary-search benchmark

On `guess_medium` (`[1, 256]`), expert (binary search) should converge in ≤ 8 steps:

```
step 1: guess 128 → lower   (so: 1..127)
step 2: guess 64  → higher  (so: 65..127)
step 3: guess 96  → lower
...
```

Reward ≈ 10.0 × (1.0 × ~0.9 × ~0.9 × 1.0) = ~8.1 (multiplicative mode, high efficiency, high quality).
