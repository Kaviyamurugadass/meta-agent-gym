#!/usr/bin/env bash
# Post-deploy health check for a live OpenEnv server.
#
# Usage:
#   URL=http://localhost:8000 bash scripts/smoke_test.sh              # local
#   URL=https://kaviya-m-my-env.hf.space bash scripts/smoke_test.sh   # deployed

set -euo pipefail

URL="${URL:-http://localhost:8000}"

echo "==> Smoke-testing: $URL"
echo ""

echo "[1/5] GET /health"
curl -sf "$URL/health" -o /tmp/smoke_health.json || { echo "FAIL: /health"; exit 1; }
cat /tmp/smoke_health.json
echo ""

echo "[2/5] GET /schema"
curl -sf "$URL/schema" -o /tmp/smoke_schema.json || { echo "FAIL: /schema"; exit 1; }
echo "Schema OK ($(wc -c < /tmp/smoke_schema.json) bytes)"

echo "[3/5] POST /reset"
curl -sf -X POST "$URL/reset" -H "Content-Type: application/json" -d '{}' -o /tmp/smoke_reset.json \
    || { echo "FAIL: /reset"; exit 1; }
echo "Reset OK"

echo "[4/5] POST /step (noop action)"
# OpenEnv wraps step body: {"action": {...}}
curl -sf -X POST "$URL/step" \
    -H "Content-Type: application/json" \
    -d '{"action": {"command": "noop", "args": {}}}' \
    -o /tmp/smoke_step.json \
    || { echo "FAIL: /step"; exit 1; }
echo "Step OK"

echo "[5/5] GET /state"
curl -sf "$URL/state" -o /tmp/smoke_state.json || { echo "FAIL: /state"; exit 1; }
echo "State OK"

echo ""
echo "==> All 5 smoke tests passed."
echo "    Env is healthy and responding at $URL"
