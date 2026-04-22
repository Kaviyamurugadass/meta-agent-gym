#!/usr/bin/env bash
# Collect baseline rollout metrics before training.
# Fills the "Baseline" column of the README before/after table.
#
# Usage:
#   bash scripts/baseline_rollout.sh
#   EPISODES=20 OUTPUT_DIR=data/baseline bash scripts/baseline_rollout.sh

set -euo pipefail

cd "$(dirname "$0")/.."

EPISODES="${EPISODES:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-data/baseline}"

mkdir -p "$OUTPUT_DIR/random" "$OUTPUT_DIR/heuristic"

echo "==> Random policy rollout ($EPISODES episodes)"
uv run python training/rollout_collection.py \
    --policy random \
    --episodes "$EPISODES" \
    --output-dir "$OUTPUT_DIR/random"

echo "==> Heuristic policy rollout ($EPISODES episodes)"
uv run python training/rollout_collection.py \
    --policy heuristic \
    --episodes "$EPISODES" \
    --output-dir "$OUTPUT_DIR/heuristic"

echo "==> Evaluating baselines"
uv run python training/evaluation.py \
    --input-dirs "$OUTPUT_DIR/random" "$OUTPUT_DIR/heuristic" \
    --output "$OUTPUT_DIR/baseline_metrics.json"

echo ""
echo "==> Baseline collection complete."
echo "    Trajectories: $OUTPUT_DIR/random, $OUTPUT_DIR/heuristic"
echo "    Metrics:      $OUTPUT_DIR/baseline_metrics.json"
echo ""
echo "    Copy these numbers into the README 'Baseline' column before training."
