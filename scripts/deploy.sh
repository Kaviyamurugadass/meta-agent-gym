#!/usr/bin/env bash
# Deploy OpenEnv environment to Hugging Face Spaces.
#
# Usage:
#   REPO_ID=Kaviya-M/my-env-name bash scripts/deploy.sh
#   REPO_ID=openenv-community/my-env-name PRIVATE=1 bash scripts/deploy.sh

set -euo pipefail

cd "$(dirname "$0")/.."

REPO_ID="${REPO_ID:-}"
PRIVATE_FLAG=""

if [ -z "$REPO_ID" ]; then
    echo "ERROR: REPO_ID env var not set."
    echo "Usage: REPO_ID=user/env-name bash scripts/deploy.sh"
    echo "       REPO_ID=openenv-community/env-name PRIVATE=1 bash scripts/deploy.sh"
    exit 1
fi

if [ "${PRIVATE:-0}" = "1" ]; then
    PRIVATE_FLAG="--private"
fi

echo "==> Pre-flight: validating OpenEnv spec"
uv run python -m openenv.cli validate

echo "==> Pre-flight: running tests"
uv run pytest tests/ -q

echo "==> Pushing to HF Spaces: $REPO_ID"
# PYTHONUTF8=1 required on Windows, no-op on Linux/Mac
PYTHONUTF8=1 uv run python -m openenv.cli push --repo-id "$REPO_ID" $PRIVATE_FLAG

echo ""
echo "==> Deploy complete."
echo "    Space URL: https://huggingface.co/spaces/$REPO_ID"
echo "    Run 'URL=https://${REPO_ID//\//-}.hf.space bash scripts/smoke_test.sh' to verify"
