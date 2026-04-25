"""Rollout collection — generate trajectories with random or heuristic policy.

Used by:
    - Pre-training baseline collection (fills README's Baseline column)
    - Data augmentation for imitation learning
    - Reward calibration sanity checks
    - Post-training evaluation using a saved LoRA adapter (policy="adapter")

CLI:
    uv run python training/rollout_collection.py \\
        --policy random --episodes 20 --output-dir data/baseline/random
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Optional

from models import Action, ActionCommand, RewardConfig, RewardMode
from server.environment import Environment
from training.trajectory import Trajectory, TrajectoryDataset, TrajectoryStep


# ---------------------------------------------------------------------------
# Policies — DOMAIN: extend with domain-aware heuristics on finale day
# ---------------------------------------------------------------------------

def _action_system_prompt() -> str:
    """System prompt for adapter eval — MATCHES the training prompt format
    (JSON array of 6 actions). The trained adapter learned this format; if
    we ask it for single per-step actions instead, it defaults to safe noop
    (no learned policy for that schema). So we ask for the full trajectory
    once per episode and the policy caches + replays it step-by-step.
    """
    return (
        "You are an agent interacting with an OpenEnv environment.\n\n"
        "Goal: emit ONE complete trajectory that builds a valid agent spec and submits it.\n\n"
        "Input: a small JSON Observation.\n"
        "Output: a JSON ARRAY of EXACTLY 6 Actions (no extra text):\n"
        "  1) set_name\n"
        "  2) set_description\n"
        "  3) add_skill   (pick ONE from required_skills if present)\n"
        "  4) write_prompt (>= 80 chars, concise workflow)\n"
        '  5) set_model   ("sonnet" unless expert)\n'
        "  6) submit\n\n"
        "Each action must be minimal JSON:\n"
        '  {"command": "<cmd>", "args": {...}}\n\n'
        "Do NOT include justification/confidence. Do NOT wrap in markdown fences.\n"
        "Do NOT think out loud or explain. Output the JSON array directly.\n\n"
        "/no_think"
    )


def make_adapter_policy(
    *,
    adapter_path: str | Path,
    base_model: str,
    device: Optional[str] = None,
    max_new_tokens: int = 768,
) -> Callable[[dict, random.Random], Action]:
    """Create a policy that uses a saved LoRA adapter to emit Actions.

    This is intended for evaluation/rollout collection after training. It loads
    the base model + adapter once (lazy) and then generates one Action per step.
    """

    adapter_path = Path(adapter_path)
    system_prompt = _action_system_prompt()

    model = None
    tokenizer = None

    def _ensure_loaded() -> None:
        nonlocal model, tokenizer
        if model is not None and tokenizer is not None:
            return

        import torch
        from peft import PeftModel

        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No LoRA adapter found at {adapter_path}. Expected adapter_config.json."
            )

        # Prefer GPU if available; fall back to CPU for correctness.
        if device is None:
            device_ = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_ = device

        # Try Unsloth first on CUDA — required because the adapter was trained
        # against Unsloth-patched attention layers (Qwen3Attention.apply_qkv etc.)
        # and won't bind cleanly to vanilla transformers. Fall back to vanilla
        # transformers + bf16/fp32 on CPU or if Unsloth import fails.
        m = None
        tok = None
        used_unsloth = False
        if device_ == "cuda":
            try:
                from unsloth import FastLanguageModel
                m, tok = FastLanguageModel.from_pretrained(
                    model_name=base_model,
                    # Eval prompts (Qwen3 chat template + JSON observation) hit
                    # 800-940 tokens; 768 caused truncation -> garbled actions
                    # -> reward 0. 2048 fits comfortably even on T4 4-bit.
                    max_seq_length=2048,
                    load_in_4bit=True,
                )
                m = PeftModel.from_pretrained(m, str(adapter_path))
                FastLanguageModel.for_inference(m)
                used_unsloth = True
            except Exception:
                m = None  # fall through to vanilla path

        if m is None:
            # Vanilla transformers fallback (CPU path or no Unsloth)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(base_model)
            load_kwargs: dict = {}
            if device_ == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                load_kwargs.update({"torch_dtype": dtype, "device_map": "auto"})
                try:
                    import bitsandbytes  # noqa: F401
                    load_kwargs["load_in_4bit"] = True
                except Exception:
                    pass
            else:
                load_kwargs.update({"torch_dtype": torch.float32, "device_map": None})
            base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
            m = PeftModel.from_pretrained(base, str(adapter_path))

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m.eval()

        model = m
        tokenizer = tok
        print(f"[adapter_policy] Loaded {'Unsloth' if used_unsloth else 'transformers'} model + adapter")

    # Debug-print counter (module-local to this policy instance)
    _debug_state = {"printed": 0}
    import os as _os
    import re as _re
    _DEBUG_LIMIT = int(_os.environ.get("SHOW_ADAPTER_COMPLETIONS_LIMIT", "5"))
    _DEBUG_ON = _os.environ.get("SHOW_ADAPTER_COMPLETIONS", "0") not in ("0", "", "false", "False")

    # Per-episode action cache. Trained model emits the full 6-action JSON array
    # at episode start (matches training format). We cache the parsed actions and
    # replay them step-by-step.
    _episode_cache: dict = {"actions": [], "task_id": None}

    def _parse_action_array(text: str) -> list[Action]:
        """Try to extract a list[Action] from the model's response.

        Handles direct array, ```json fences, and regex fallback. Returns
        empty list if nothing parseable.
        """
        cleaned = text.strip()
        # Strip code fences
        if cleaned.startswith("```"):
            # ```json ... ``` or ``` ... ```
            cleaned = cleaned.lstrip("`").lstrip()
            if cleaned[:4].lower() == "json":
                cleaned = cleaned[4:].lstrip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()

        actions: list[Action] = []
        try:
            parsed = json.loads(cleaned)
        except Exception:
            # Regex fallback: find first [...] array
            m = _re.search(r"\[\s*\{.*?\}\s*\]", cleaned, _re.DOTALL)
            if not m:
                return []
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                return []

        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            try:
                actions.append(Action.model_validate(item))
            except Exception:
                continue
        return actions

    def policy(observation_dict: dict, rng: random.Random) -> Action:  # noqa: ARG001
        _ensure_loaded()
        assert model is not None and tokenizer is not None

        import torch

        current_task = observation_dict.get("task_id")
        current_step = observation_dict.get("step", 0)

        # Detect new episode: env step rolled back to 0, or different task,
        # or cache empty. On any of these, regenerate the trajectory.
        is_new_episode = (
            current_step == 0
            or current_task != _episode_cache.get("task_id")
            or not _episode_cache.get("actions")
        )

        if is_new_episode:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(observation_dict, indent=2)},
            ]

            # enable_thinking=False is the reliable Qwen3 toggle; fall back if
            # the tokenizer doesn't accept the kwarg.
            if getattr(tokenizer, "chat_template", None) is not None:
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
            else:
                prompt = system_prompt + "\n\nObservation:\n" + messages[1]["content"] + "\n\nAction JSON:"

            inputs = tokenizer(prompt, return_tensors="pt")
            if hasattr(model, "device") and model.device is not None:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            cached_actions = _parse_action_array(response)
            _episode_cache["actions"] = cached_actions
            _episode_cache["task_id"] = current_task

            if _DEBUG_ON and _debug_state["printed"] < _DEBUG_LIMIT:
                _debug_state["printed"] += 1
                print("=" * 30 + f" ADAPTER COMPLETION #{_debug_state['printed']} " + "=" * 30)
                print(f"[adapter] task={current_task} step={current_step} (new episode -> generating)")
                print(f"[adapter] raw_response (len={len(response)}):")
                print(response[:2000])
                print(f"[adapter] parsed {len(cached_actions)} actions: "
                      + ", ".join(a.command.value for a in cached_actions))
                print("=" * 90)

        # Use the action at the current step from the cached trajectory.
        actions = _episode_cache.get("actions", [])
        if current_step < len(actions):
            return actions[current_step]

        # Cache exhausted (model emitted < expected actions, or step beyond array).
        # Fall back to NOOP so the env naturally truncates without crashing.
        return Action(
            command=ActionCommand.NOOP,
            args={},
            justification="Cache exhausted — adapter emitted fewer actions than env steps",
            confidence=0.0,
        )

    return policy


def random_policy(
    observation_dict: dict,  # noqa: ARG001
    rng: random.Random,
) -> Action:
    """Uniform random over ActionCommand values."""
    cmd = rng.choice(list(ActionCommand))
    return Action(command=cmd, args={}, confidence=0.5)


def heuristic_policy(
    observation_dict: dict,
    rng: random.Random,  # noqa: ARG001
) -> Action:
    """Competent rule-based baseline that actually builds a valid agent spec.

    Fills each required field once (name, description, skill, prompt, model) then
    submits. Purpose: prove the environment is reachable with >0 reward so GRPO
    has learning signal to bootstrap from. Not optimal — judge-scored components
    (description_quality, workflow_clarity, etc.) will be mediocre.
    """
    step = observation_dict.get("step", 0)
    max_steps = observation_dict.get("max_steps", 7)
    task_id = observation_dict.get("task_id", "task")
    available_skills = observation_dict.get("available_skills") or []
    current_spec = observation_dict.get("current_spec") or {}

    # Submit on the final step regardless
    if step >= max_steps - 1:
        return Action(command=ActionCommand.SUBMIT, confidence=0.7)

    # Fill missing required fields in priority order
    if not current_spec.get("name"):
        return Action(
            command=ActionCommand.SET_NAME,
            args={"name": task_id.replace("_", "-")},
            confidence=0.6,
        )
    if not current_spec.get("description"):
        return Action(
            command=ActionCommand.SET_DESCRIPTION,
            args={"description": f"Agent that handles {task_id} tasks end-to-end."},
            confidence=0.6,
        )
    if not current_spec.get("skills") and available_skills:
        return Action(
            command=ActionCommand.ADD_SKILL,
            args={"skill": available_skills[0]},
            confidence=0.6,
        )
    prompt = current_spec.get("system_prompt", "")
    if len(prompt) < 50:
        return Action(
            command=ActionCommand.WRITE_PROMPT,
            args={
                "prompt": (
                    "You are a specialist agent. Read the task carefully, plan the "
                    "steps, execute each one, then verify the result before submitting."
                ),
                "mode": "replace",
            },
            confidence=0.6,
        )
    if not current_spec.get("model"):
        return Action(
            command=ActionCommand.SET_MODEL,
            args={"model": "sonnet"},
            confidence=0.6,
        )
    # All required fields filled — coast until the final SUBMIT step
    return Action(command=ActionCommand.NOOP, confidence=0.5)


POLICIES = {
    "random": random_policy,
    "heuristic": heuristic_policy,
}


# ---------------------------------------------------------------------------
# Rollout runner
# ---------------------------------------------------------------------------


def run_episode(
    env: Environment,
    policy_name: str,
    scenario_name: Optional[str] = None,
    rng: Optional[random.Random] = None,
    adapter_path: Optional[str | Path] = None,
    base_model: Optional[str] = None,
    cached_policy: Optional[Callable[[dict, random.Random], Action]] = None,
) -> Trajectory:
    """Run one episode to completion, return a Trajectory.

    If `cached_policy` is provided, it is used directly (avoids re-creating
    the adapter policy per episode, which would re-load the model each call).
    """
    if cached_policy is not None:
        policy = cached_policy
    elif policy_name == "adapter":
        if adapter_path is None or base_model is None:
            raise ValueError("policy='adapter' requires adapter_path and base_model")
        policy = make_adapter_policy(adapter_path=adapter_path, base_model=base_model)
    else:
        if policy_name not in POLICIES:
            raise ValueError(f"Unknown policy: {policy_name}. Valid: {list(POLICIES) + ['adapter']}")
        policy = POLICIES[policy_name]
    rng = rng or random.Random()

    obs = env.reset(scenario_name=scenario_name)
    traj = Trajectory(
        task_id=obs.task_id,
        scenario_name=scenario_name,
        difficulty=env._task.difficulty if env._task else None,
        metadata={"policy": policy_name},
    )

    while not (obs.done or obs.truncated):
        action = policy(obs.model_dump(), rng)
        obs = env.step(action)
        traj.append(TrajectoryStep(
            step=obs.step,
            action=action.model_dump(),
            observation=obs.model_dump(),
            reward=obs.reward,
            done=obs.done,
            reward_breakdown=obs.reward_breakdown,
        ))

    # DOMAIN: define success per theme. Default: positive final reward.
    traj.success = traj.total_reward > 0
    return traj


def collect(
    episodes: int,
    policy: str,
    output_dir: str | Path,
    scenario_name: Optional[str] = None,
    seed: Optional[int] = None,
    domain_randomise: bool = True,
    curriculum_phase: Optional[int] = None,
    reward_config: Optional[RewardConfig] = None,
    adapter_path: Optional[str | Path] = None,
    base_model: Optional[str] = None,
) -> TrajectoryDataset:
    """Collect N episodes, save to output_dir as a TrajectoryDataset."""
    rng = random.Random(seed)
    dataset = TrajectoryDataset()

    # Build adapter policy ONCE before the episode loop. This avoids reloading
    # the LoRA-adapted model from scratch on every episode (which would add
    # ~1-2 minutes per episode and make eval impractically slow).
    cached_policy: Optional[Callable[[dict, random.Random], Action]] = None
    if policy == "adapter":
        if adapter_path is None or base_model is None:
            raise ValueError("policy='adapter' requires adapter_path and base_model")
        cached_policy = make_adapter_policy(
            adapter_path=adapter_path,
            base_model=base_model,
        )

    for i in range(episodes):
        env = Environment(
            domain_randomise=domain_randomise,
            seed=rng.randint(0, 10_000_000),
            curriculum_phase=curriculum_phase or 1,
            reward_config=reward_config,
        )
        traj = run_episode(
            env,
            policy,
            scenario_name=scenario_name,
            rng=rng,
            adapter_path=adapter_path,
            base_model=base_model,
            cached_policy=cached_policy,
        )
        dataset.append(traj)
        print(
            f"[{i+1}/{episodes}] task={traj.task_id} "
            f"length={traj.length} reward={traj.total_reward:.3f} "
            f"success={traj.success}"
        )

    dataset.save_dir(output_dir)
    summary = dataset.summary()
    print(f"\n==> {episodes} episodes saved to {output_dir}")
    print(f"    mean_reward={summary['mean_reward']:.3f} "
          f"success_rate={summary['success_rate']:.1%} "
          f"mean_length={summary['mean_length']:.1f}")
    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect rollout trajectories.")
    parser.add_argument("--policy", choices=list(POLICIES) + ["adapter"], default="random")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--scenario-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-randomise", action="store_true")
    parser.add_argument("--curriculum-phase", type=int, default=None,
                        help="Curriculum phase (1-4) for task selection")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter directory (policy=adapter)")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base HF model id (policy=adapter)")
    parser.add_argument("--reward-mode",
                        choices=[m.value for m in RewardMode],
                        default=None,
                        help="Override reward mode (default: HYBRID). "
                             "Use 'additive' to expose anti-hack penalty surface "
                             "without gate masking.")
    args = parser.parse_args()

    reward_config = None
    if args.reward_mode is not None:
        reward_config = RewardConfig(mode=RewardMode(args.reward_mode))

    collect(
        episodes=args.episodes,
        policy=args.policy,
        output_dir=args.output_dir,
        scenario_name=args.scenario_name,
        seed=args.seed,
        domain_randomise=not args.no_randomise,
        curriculum_phase=args.curriculum_phase,
        reward_config=reward_config,
        adapter_path=args.adapter_path,
        base_model=args.base_model,
    )


if __name__ == "__main__":
    main()
