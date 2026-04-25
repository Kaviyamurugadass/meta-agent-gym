"""GRPO training via Unsloth 4-bit LoRA — Colab T4 / mid-range GPUs.

Pattern: same GRPOTrainer flow as grpo_trl.py, but using Unsloth's
`FastLanguageModel` with 4-bit quantization + LoRA adapters for memory
efficiency on T4 (16GB) / RTX 4090 (24GB) / A10 (24GB).

⚠️ Critical gotchas (from TEMPLATE_PLAN.md — verified from Unsloth docs):
    1. Free Colab T4 does NOT support FP8 — must use BF16 LoRA
    2. Disable vLLM fast-inference in GRPO (known conflict)
    3. On T4 (16GB), use: batch=1, num_generations=2, grad_accum=4,
       max_seq_length=1024 (drop to 768 if OOM)

CLI (T4 Colab — shipped path):
    uv sync --extra train --extra unsloth
    uv run python training/grpo_unsloth.py \\
        --model-id Qwen/Qwen2.5-0.5B \\
        --per-device-train-batch-size 1 \\
        --num-generations 2 \\
        --gradient-accumulation-steps 4 \\
        --max-seq-length 1024

CLI (L4 / A100 — onsite scale-up target):
    uv run python training/grpo_unsloth.py \\
        --model-id Qwen/Qwen3-1.7B \\
        --per-device-train-batch-size 2 \\
        --num-generations 4 \\
        --gradient-accumulation-steps 4 \\
        --max-seq-length 2048 \\
        --num-epochs 3

CLI (dry-run, no GPU):
    uv run python training/grpo_unsloth.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"  # shipped Colab T4 path; pass --model-id Qwen/Qwen3-1.7B onsite
DEFAULT_OUTPUT = "training/grpo-unsloth-output"

SYSTEM_PROMPT = """You are an agent interacting with an OpenEnv environment.

The environment requires MULTI-STEP trajectories to build a complete agent spec.
For each turn, you will receive an Observation (JSON). You must respond with a
JSON ARRAY of Actions to execute in order in a fresh episode.

Each Action must match the Action schema:
  {"command": "<cmd>", "args": {...}, "justification": "...", "confidence": 0.0-1.0}

Your action array should generally:
  1) set_name
  2) set_description
  3) add_skill (1-3)
  4) write_prompt (>= 50 chars)
  5) set_model (usually sonnet)
  6) submit

Respond ONLY with the JSON array. No explanation, no prose, no markdown fences.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training via Unsloth 4-bit LoRA.")

    # Model + training
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=2,
                   help="T4 default 2 to fit memory. H100 can go 4-8.")
    p.add_argument("--max-seq-length", type=int, default=1024,
                   help="Drop to 768 if OOM on T4.")
    p.add_argument("--max-completion-length", type=int, default=256)

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

    # Unsloth / LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--disable-4bit", action="store_true",
                   help="Use BF16 LoRA instead of 4-bit (H100+ only)")

    # Rollout / data
    p.add_argument("--dataset-episodes", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=6)
    p.add_argument("--collection-policy", choices=["random", "heuristic"], default="heuristic")
    p.add_argument("--scenario-name", action="append", default=None)
    p.add_argument("--domain-randomise", action="store_true")

    # Reward backend
    p.add_argument("--reward-backend", choices=["local", "remote"], default="local")
    p.add_argument("--base-url", default="http://localhost:8000")

    # Curriculum
    p.add_argument("--curriculum", action="store_true",
                   help="Enable curriculum learning with automatic phase progression")
    p.add_argument("--curriculum-state", type=str, default=None,
                   help="Path to saved curriculum state JSON (resume training)")

    # Dev
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # T4-specific flag (explicit BF16)
    p.add_argument("--trust-remote-code", action="store_true")

    return p.parse_args()


def dry_run(args: argparse.Namespace) -> None:
    """Validate Unsloth setup — minimal imports, no GPU needed."""
    print("=== GRPO Unsloth dry-run ===")
    print(f"model: {args.model_id}")
    print(f"output: {args.output_dir}")
    print(f"num_generations: {args.num_generations}")
    print(f"max_seq_length: {args.max_seq_length}")
    print(f"4bit: {not args.disable_4bit}")
    print(f"lora: r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")

    from training.reward_backend import make_backend
    from training.trajectory import Trajectory  # noqa: F401

    backend = make_backend(
        args.reward_backend,
        args.base_url if args.reward_backend == "remote" else None,
    )
    print(f"reward backend built: {type(backend).__name__}")

    # Check Unsloth import
    try:
        import unsloth  # noqa: F401
        print(f"unsloth available")
    except ImportError:
        print("unsloth NOT installed — run `uv sync --extra train --extra unsloth`")

    print("\n[OK] Dry-run passed. Remove --dry-run to launch real training.")


def _build_prompt_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
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
        # Keep the training prompt small enough to fit T4 max_seq_length.
        # The full Observation contains many fields that aren't needed for choosing
        # an action sequence. Include task context + current spec only.
        task = env._task  # internal but stable in this codebase
        compact_obs = {
            "task_id": obs.task_id,
            "step": obs.step,
            "max_steps": obs.max_steps,
            "domain": getattr(task, "domain", None),
            "difficulty": getattr(task, "difficulty", None),
            "problem_statement": getattr(task, "problem_statement", None),
            "required_skills": getattr(task, "required_skills", None),
            "recommended_skills": getattr(task, "recommended_skills", None),
            "available_skills": list(getattr(env, "_task", None).required_skills or []) + list(getattr(env, "_task", None).recommended_skills or []),
            "current_spec": obs.current_spec,
        }
        prompts.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(compact_obs, indent=2)},
            ],
            "scenario_name": scenario or obs.task_id,
        })
    return prompts


def _make_reward_fn(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from inference import parse_actions
    from training.reward_backend import make_backend

    backend = make_backend(
        args.reward_backend,
        args.base_url if args.reward_backend == "remote" else None,
    )

    def reward_fn(completions: list[str], **kw: Any) -> list[float]:
        scenarios = kw.get("scenario_name")
        if scenarios is None:
            scenarios = [None] * len(completions)
        elif isinstance(scenarios, str):
            # TRL sometimes passes a scalar string even when batch size > 1.
            scenarios = [scenarios] * len(completions)
        elif isinstance(scenarios, list) and len(scenarios) != len(completions):
            # Be defensive: pad/trim to match completions length.
            scenarios = (scenarios + [None] * len(completions))[: len(completions)]

        rewards = []
        printed = 0
        fail_fast = os.getenv("REWARD_FN_FAIL_FAST", "0") == "1"
        for completion, scenario in zip(completions, scenarios):
            try:
                # TRL/Unsloth can pass completions as:
                #   - a string
                #   - a list[str] of token-chunks
                # Normalize to a single string before parsing.
                if isinstance(completion, list):
                    completion_text = "".join(str(x) for x in completion)
                else:
                    completion_text = str(completion)

                actions = parse_actions(completion_text)
                r, _ = backend.score(actions, scenario_name=scenario)
                rewards.append(float(r))
            except Exception as e:
                # Return -1.0 to penalize unparseable/errored completions,
                # but print the first few exceptions so debugging is possible
                # in Colab logs. (stdout is more reliably captured than stderr.)
                if printed < 3:
                    head = (completion_text or "")[:300].replace("\n", "\\n")
                    print(f"[reward_fn] error: {type(e).__name__}: {e}")
                    print(f"[reward_fn] scenario={scenario!r} completion_head={head!r}")
                    printed += 1
                if fail_fast:
                    raise
                rewards.append(-1.0)
        return rewards

    return reward_fn


def launch_training(args: argparse.Namespace) -> None:  # pragma: no cover — needs GPU
    """Real training. Unsloth + TRL GRPO with vLLM fast-inference DISABLED."""
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported

    # Load model with 4-bit quantization (unless --disable-4bit)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.disable_4bit,
        trust_remote_code=args.trust_remote_code,
    )

    # Unsloth's 4-bit variants sometimes ship without a chat_template.
    # TRL's GRPOTrainer calls tokenizer.apply_chat_template on the prompt dataset
    # and crashes if chat_template is unset. Apply Qwen's ChatML template as a
    # safe default (compatible with Qwen2.5 / Qwen3 / any ChatML-trained model).
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
        print("[setup] Applied ChatML chat_template (tokenizer had none)")

    # Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",  # Gotcha: required for T4 memory
        random_state=args.seed,
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
        max_prompt_length=args.max_seq_length - args.max_completion_length,
        logging_steps=1,
        save_steps=50,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        seed=args.seed,
        # Gotcha: vLLM fast inference conflicts with Unsloth GRPO — disable
        use_vllm=False,
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

    # Save LoRA adapter + merged model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n==> Training complete. LoRA adapter → {args.output_dir}")
    print("    To push: `hf upload Kaviya-M/my-env-grpo-qwen3-0.6b " + args.output_dir + "`")


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        dry_run(args)
        return
    launch_training(args)


if __name__ == "__main__":
    main()
