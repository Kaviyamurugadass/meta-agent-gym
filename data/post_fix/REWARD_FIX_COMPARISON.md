# Reward Sign-Flip Fix — Policy-Level Evidence

Before/after comparison of per-step and per-episode reward for simple
policies, demonstrating that the fix on `server/rewards/reward.py:158`
(formerly line 111) changes behaviour *at the policy level*, not only in
unit tests.

## The bug, in one line

```diff
- total = core + bonus + progress - penalty - regression - sum(anti_hack_penalties.values())
+ total = core + bonus + progress - penalty - regression + sum(anti_hack_penalties.values())
```

`anti_hack_penalties` are stored as negative values (e.g.
`anti_hack_empty_spec = -5.0`). Subtracting them flipped the sign, turning
a −5 penalty into a +5 bonus. GRPO obediently exploited it.

## Before fix — historical Colab trajectory

Source: `data/colab_trained/trajectory_0000.json` (preserved; do not regenerate).

- Actions: `noop → noop → noop → noop → noop → noop → submit`
- `current_spec` at end: `{}` (empty)
- **Per-step reward: +7.40** (should have been −5.00 or 0.00)
- **Episode total: +51.80** across 7 steps
- `success: True` — the policy "succeeded" by producing nothing

## After fix — fresh rollouts (5 episodes each, seed=42)

Collected via:

```
python -m training.rollout_collection --policy random    --reward-mode additive --episodes 5 --seed 42 --output-dir data/post_fix/additive_random
python -m training.rollout_collection --policy heuristic --reward-mode additive --episodes 5 --seed 42 --output-dir data/post_fix/additive_heuristic
python -m training.rollout_collection --policy random    --reward-mode hybrid   --episodes 5 --seed 42 --output-dir data/post_fix/hybrid_random
python -m training.rollout_collection --policy heuristic --reward-mode hybrid   --episodes 5 --seed 42 --output-dir data/post_fix/hybrid_heuristic
```

| Mode     | Policy    | Mean reward / episode | Success rate | Per-step reward for empty spec |
|----------|-----------|----------------------:|-------------:|-------------------------------:|
| HYBRID   | random    |                **0.000** |         0% | 0.00 (gate fires — defence in depth) |
| HYBRID   | heuristic |               **+20.333** |       100% | +2.91 avg (valid spec builds cleanly) |
| ADDITIVE | random    |               **−32.196** |         0% | −4.58 (−5 penalty applied correctly) |
| ADDITIVE | heuristic |                **−9.507** |         0% | final-step +0.67 when spec complete |

## The magnitude of the fix

Same empty-spec-submitting policy, before and after:

|                     | Per step | Per 7-step episode |
|---------------------|---------:|-------------------:|
| **Before fix**      |    +7.40 |             +51.80 |
| **After fix (ADD)** |    −4.58 |             −32.20 |
| **Swing**           |  **11.98** |            **84.00** |

That's an **84-point per-episode correction** on a reward surface where the
GRPO advantage for "correct behaviour" is order of 20 points. Before the
fix, the hack rewarded empty specs more than a fully valid heuristic did.

## Why HYBRID looked "fine" in earlier baselines

`data/baseline/random` (pre-fix) showed `total_reward: 0.0` because it ran
under HYBRID mode — the gate fired, short-circuited before the sign-flip
math ran, and zeroed the reward. The bug was latent until a config change
(apparently during the Colab training run) lowered or disabled the gate,
at which point the +5 bonus surfaced and the policy collapsed to
`noop → submit` within a handful of updates.

**Lesson for RLVR:** defence-in-depth matters. A hack masked by one layer
can still poison training if any other layer mis-fires. Hard gates and
penalty signs both need to be correct.

## Regression coverage

```
tests/test_meta_agent_reward.py::test_empty_spec_never_rewarded[hybrid]    PASS
tests/test_meta_agent_reward.py::test_empty_spec_never_rewarded[additive]  PASS
tests/test_meta_agent_reward.py::test_empty_spec_penalty_sign_is_negative  PASS
```

These fail immediately if the operator is reverted.
