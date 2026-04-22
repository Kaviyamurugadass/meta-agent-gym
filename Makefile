.PHONY: help install install-train install-unsloth test test-cov lint format validate run docker-build docker-run train-trl train-unsloth train-dry eval baseline deploy smoke clean

help:
	@echo "Available targets:"
	@echo ""
	@echo "  Setup"
	@echo "    install          Install core deps"
	@echo "    install-train    Install training deps (TRL, torch, transformers)"
	@echo "    install-unsloth  Install Unsloth deps (for T4 Colab LoRA)"
	@echo ""
	@echo "  Quality"
	@echo "    test             Run test suite"
	@echo "    test-cov         Run tests with coverage report"
	@echo "    readiness        Observation + reward-quality checks (skips placeholder scenarios)"
	@echo "    bench            Run expert benchmark trajectories"
	@echo "    lint             Ruff check"
	@echo "    format           Ruff format"
	@echo "    validate         openenv CLI validate"
	@echo ""
	@echo "  Run / Serve"
	@echo "    run              Start local dev server"
	@echo "    docker-build     Build Docker image"
	@echo "    docker-run       Run Docker container on :8000"
	@echo ""
	@echo "  Training"
	@echo "    train-trl        Full GRPO (TRL, H100/A100)"
	@echo "    train-unsloth    Unsloth 4-bit LoRA (T4 Colab)"
	@echo "    train-dry        Training dry-run (no GPU needed, verifies setup)"
	@echo "    eval             Evaluate trained model"
	@echo "    baseline         Collect baseline rollouts (random + heuristic)"
	@echo "    plots            Render reward curves from data/trained/"
	@echo ""
	@echo "  Deploy"
	@echo "    deploy           Push to HF Spaces (needs REPO_ID=user/name)"
	@echo "    smoke            Post-deploy health check (needs URL=...)"
	@echo ""
	@echo "  Housekeeping"
	@echo "    clean            Remove caches + training outputs"

install:
	uv sync

install-train:
	uv sync --extra train

install-unsloth:
	uv sync --extra train --extra unsloth

test:
	uv run pytest tests/ -q

readiness:
	uv run pytest tests/test_observation_quality.py tests/test_reward_quality.py -v

bench:
	uv run python training/benchmark.py

plots:
	uv run python training/plot_rewards.py \
		--input-dir data/trained \
		--output training/plots/run_latest.png \
		--title "Latest training run"

test-cov:
	uv run pytest tests/ --cov --cov-report=term-missing -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

validate:
	uv run python -m openenv.cli validate

run:
	uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t openenv-r2-kit .

docker-run:
	docker run -p 8000:8000 --env-file .env openenv-r2-kit

train-trl:
	uv run python training/grpo_trl.py

train-unsloth:
	uv run python training/grpo_unsloth.py

train-dry:
	uv run python training/grpo_unsloth.py --dry-run

eval:
	uv run python training/evaluation.py

baseline:
	bash scripts/baseline_rollout.sh

deploy:
	bash scripts/deploy.sh

smoke:
	bash scripts/smoke_test.sh

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf training/grpo-output training/grpo-unsloth-*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
