"""GRPO training via TRL — full fine-tune or LoRA on H100-class hardware.

Pattern: use TRL's `GRPOTrainer` with an OpenEnv-backed reward function.
For each prompt, the trainer samples N completions; we run each completion
as an action sequence in our env and return the total reward.

CLI:
    # Dry-run (no GPU, no training — validates setup)
    uv run python training/grpo_trl.py --dry-run

    # Real training on H100
    uv run python training/grpo_trl.py \\
        --model-id Qwen/Qwen3.5-4B \\
        --output-dir training/grpo-output \\
        --dataset-episodes 8 --rollout-steps 6 \\
        --num-generations 4

⚠️ Gotchas (from TEMPLATE_PLAN.md):
    - Set max_concurrent_envs≥4 in server/app.py (SUPPORTS_CONCURRENT_SESSIONS=True already)
    - Pin TRL to a version that handles multi-step OpenEnv GRPO cleanly (issue #4543)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT = "training/grpo-output"

SYSTEM_PROMPT = """You are an agent interacting with an OpenEnv environment.

Respond with a single JSON object matching the Action schema:
  {"command": "<cmd>", "args": {...}, "justification": "...", "confidence": 0.0-1.0}

Respond ONLY with the JSON object. No explanation, no prose.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training via TRL.")

    # Model + training
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--max-prompt-length", type=int, default=1024)

    # Loss config — DAPO is recommended default
    p.add_argument(
        "--loss-type",
        choices=["grpo", "dapo", "bnpo"],
        default="dapo",
        help="DAPO (asymmetric clipping + dynamic sampling) outperforms vanilla GRPO",
    )
    p.add_argument(
        "--mask-truncated-completions",
        action="store_true",
        default=True,
        help="Exclude token-capped completions from loss (recommended)",
    )

    # Rollout / data
    p.add_argument("--dataset-episodes", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=6)
    p.add_argument("--collection-policy", choices=["random", "heuristic"], default="heuristic")
    p.add_argument("--scenario-name", action="append", default=None,
                   help="Repeatable — restrict training to specific scenarios")
    p.add_argument("--domain-randomise", action="store_true")

    # Reward backend
    p.add_argument("--reward-backend", choices=["local", "remote"], default="local")
    p.add_argument("--base-url", default="http://localhost:8000",
                   help="Used when --reward-backend=remote")

    # Curriculum
    p.add_argument("--curriculum", action="store_true",
                   help="Enable curriculum learning with automatic phase progression")
    p.add_argument("--curriculum-state", type=str, default=None,
                   help="Path to saved curriculum state JSON (resume training)")

    # Dev / debug
    p.add_argument("--dry-run", action="store_true",
                   help="Validate setup without launching training")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def dry_run(args: argparse.Namespace) -> None:
    """Validate setup — import deps, build reward fn, collect one rollout."""
    print("=== GRPO TRL dry-run ===")
    print(f"model: {args.model_id}")
    print(f"output: {args.output_dir}")
    print(f"num_generations: {args.num_generations}")
    print(f"reward_backend: {args.reward_backend}")

    # Verify trajectory + rollout imports work
    from training.rollout_collection import run_episode  # noqa: F401
    from training.reward_backend import make_backend
    from training.trajectory import Trajectory  # noqa: F401
    from server.environment import Environment

    backend = make_backend(
        args.reward_backend,
        args.base_url if args.reward_backend == "remote" else None,
    )
    print(f"reward backend built: {type(backend).__name__}")

    # Run one real rollout through the reward backend
    from models import Action, ActionCommand
    actions = [Action(command=ActionCommand.NOOP) for _ in range(args.rollout_steps)]
    total, obs = backend.score(actions, scenario_name=args.scenario_name[0] if args.scenario_name else None)
    print(f"sample rollout: total_reward={total:.3f} observations={len(obs)}")

    # Check TRL is available but don't import heavy deps unless real run
    try:
        import trl  # noqa: F401
        print(f"trl available: {trl.__version__ if hasattr(trl, '__version__') else 'unknown'}")
    except ImportError:
        print("trl NOT installed — run `uv sync --extra train` before real training")

    print("\n[OK] Dry-run passed. Remove --dry-run to launch real training.")


def _build_prompt_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Collect `dataset-episodes` initial observations as training prompts.

    GRPO needs a dataset of prompts. We use initial observations from reset()
    as prompts — each sampled episode becomes one training example.
    """
    from server.environment import Environment

    prompts: list[dict[str, Any]] = []
    scenarios = args.scenario_name or [None]

    controller = None
    if args.curriculum:
        from training.curriculum import CurriculumController
        controller = (
            CurriculumController.load(args.curriculum_state)
            if args.curriculum_state
            else CurriculumController()
        )

    for i in range(args.dataset_episodes):
        scenario = scenarios[i % len(scenarios)]
        phase = controller.current_phase if controller else 1
        env = Environment(domain_randomise=args.domain_randomise, seed=args.seed + i)
        obs = env.reset(scenario_name=scenario, curriculum_phase=phase)
        prompts.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(obs.model_dump(exclude_none=True), indent=2)},
            ],
            "scenario_name": scenario or obs.task_id,
        })
    return prompts


def _make_reward_fn(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    """Return a TRL-compatible reward function.

    Signature: (completions, **kwargs) -> list[float]
        - completions: list of strings (the N generations per prompt)
        - kwargs: may include `scenario_name` from the dataset row

    Each completion is parsed as an Action, scored via the backend.
    """
    from inference import parse_action
    from training.reward_backend import make_backend

    backend = make_backend(
        args.reward_backend,
        args.base_url if args.reward_backend == "remote" else None,
    )

    def reward_fn(completions: list[str], **kw: Any) -> list[float]:
        scenarios = kw.get("scenario_name") or [None] * len(completions)
        rewards = []
        for completion, scenario in zip(completions, scenarios):
            try:
                action = parse_action(completion)
                r, _ = backend.score([action], scenario_name=scenario)
                rewards.append(float(r))
            except Exception:
                rewards.append(-1.0)  # parse failure → penalty
        return rewards

    return reward_fn


def launch_training(args: argparse.Namespace) -> None:  # pragma: no cover — needs GPU
    """Real training — imports TRL + torch + transformers. Heavy deps."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )

    prompts = _build_prompt_dataset(args)
    dataset = Dataset.from_list(prompts)

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=1,
        save_steps=50,
        bf16=torch.cuda.is_available(),
        seed=args.seed,
        loss_type=args.loss_type,
        mask_truncated_completions=args.mask_truncated_completions,
    )

    reward_fn = _make_reward_fn(args)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\n==> Training complete. Model → {args.output_dir}")


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        dry_run(args)
        return
    launch_training(args)


if __name__ == "__main__":
    main()
