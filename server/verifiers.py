"""Hard verifiers for RLVR approach — fast, free, 100% of steps.

These are non-LLM checks that run before the judge. They prevent format hacks
and ensure basic validity of agent specifications.
"""

from __future__ import annotations

import yaml
from pydantic import BaseModel


class VerifierResult(BaseModel):
    """Result from a single hard verification check."""
    passed: bool
    score: float  # 0.0 or 1.0 (binary)
    errors: list[str] = []


class HardVerifiers:
    """Fast, free verification checks (RLVR approach).

    Runs 100% of steps, costs ~$0, catches format/structure errors before
    they reach the expensive judge.
    """

    @staticmethod
    def verify_yaml(spec: dict) -> VerifierResult:
        """Check if spec can be serialized to valid YAML."""
        try:
            yaml.safe_dump(spec)
            return VerifierResult(passed=True, score=1.0, errors=[])
        except Exception as e:
            return VerifierResult(
                passed=False,
                score=0.0,
                errors=[f"YAML serialization failed: {e!s}"]
            )

    @staticmethod
    def verify_required_fields(spec: dict) -> VerifierResult:
        """Check required fields are present and non-empty."""
        required = ["name", "description", "system_prompt"]
        errors: list[str] = []

        for field in required:
            if field not in spec:
                errors.append(f"Missing required field: {field}")
            elif not spec[field]:
                errors.append(f"Empty required field: {field}")

        return VerifierResult(
            passed=len(errors) == 0,
            score=1.0 if len(errors) == 0 else 0.0,
            errors=errors
        )

    @staticmethod
    def verify_prompt_length(spec: dict, min_length: int = 50) -> VerifierResult:
        """Check system prompt meets minimum length (anti-empty hack)."""
        prompt = spec.get("system_prompt", "")
        if len(prompt) < min_length:
            return VerifierResult(
                passed=False,
                score=0.0,
                errors=[f"Prompt too short: {len(prompt)} < {min_length}"]
            )
        return VerifierResult(passed=True, score=1.0, errors=[])

    @staticmethod
    def verify_model_type(spec: dict) -> VerifierResult:
        """Check model is one of the allowed values."""
        valid_models = ["haiku", "sonnet", "opus", "inherit"]
        model = spec.get("model", "sonnet")

        if model not in valid_models:
            return VerifierResult(
                passed=False,
                score=0.0,
                errors=[f"Invalid model: {model}. Must be one of {valid_models}"]
            )
        return VerifierResult(passed=True, score=1.0, errors=[])

    @staticmethod
    def verify_skills_format(spec: dict) -> VerifierResult:
        """Check skills is a list (or can be converted to one)."""
        skills = spec.get("skills")

        if skills is None:
            return VerifierResult(passed=True, score=1.0, errors=[])  # Optional field

        if not isinstance(skills, list):
            return VerifierResult(
                passed=False,
                score=0.0,
                errors=[f"skills must be a list, got {type(skills).__name__}"]
            )

        return VerifierResult(passed=True, score=1.0, errors=[])

    @classmethod
    def verify_all(cls, spec: dict) -> dict[str, VerifierResult]:
        """Run all hard verifiers.

        Returns:
            dict mapping verifier name to its result
        """
        return {
            "yaml_valid": cls.verify_yaml(spec),
            "has_required_fields": cls.verify_required_fields(spec),
            "prompt_length_ok": cls.verify_prompt_length(spec),
            "model_valid": cls.verify_model_type(spec),
            "skills_format_ok": cls.verify_skills_format(spec),
        }

    @classmethod
    def get_gate_results(cls, spec: dict) -> tuple[bool, list[str]]:
        """Check if all GATE components pass.

        Gate components are hard requirements — if any fail, the reward is 0.

        Returns:
            (all_passed, list_of_errors)
        """
        results = cls.verify_all(spec)

        # GATE components: these must ALL pass
        gate_components = ["yaml_valid", "has_required_fields", "prompt_length_ok"]

        all_passed = True
        all_errors: list[str] = []

        for component in gate_components:
            result = results[component]
            if not result.passed:
                all_passed = False
                all_errors.extend(result.errors)

        return all_passed, all_errors
