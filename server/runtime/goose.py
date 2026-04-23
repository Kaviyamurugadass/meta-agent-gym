"""Goose runtime integration for real execution (steps 3, 6, 9).

This module runs generated agents using the Goose framework to validate
they actually work. This is the 10% "real execution" tier in our three-tier
verification system.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from models import AgentSpec

logger = logging.getLogger("server.runtime.goose")


class GooseExecutionResult(BaseModel):
    """Result from running an agent via Goose."""

    success: bool
    output: str = ""
    error: str = ""
    tokens_used: int = 0
    duration: float = 0.0
    validation_passed: bool = False


class GooseRunner:
    """Run agents using Goose and collect metrics.

    This is the "real execution" tier (10% of steps at steps 3, 6, 9).
    Validates that generated agents actually work.

    NOTE: This is a stub implementation. Real Goose integration requires:
    - Goose binary in PATH
    - Test input files
    - Expected output validation
    """

    # Path to goose binary (can be overridden)
    GOOSE_PATH: str = "goose"

    # Default timeout for agent execution
    DEFAULT_TIMEOUT: int = 60

    def __init__(self, goose_path: Optional[str] = None):
        """Initialize Goose runner.

        Args:
            goose_path: Path to goose binary. If None, uses GOOSE_PATH.
        """
        self.goose_path = goose_path or self.GOOSE_PATH

    def is_available(self) -> bool:
        """Check if Goose is available in the environment."""
        try:
            result = subprocess.run(
                [self.goose_path, "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def run(
        self,
        agent_spec: AgentSpec,
        test_input: dict[str, Any],
        timeout: int | None = None,
    ) -> GooseExecutionResult:
        """Execute agent with test input.

        Args:
            agent_spec: The agent specification to run
            test_input: Test input data for the agent
            timeout: Execution timeout in seconds

        Returns:
            GooseExecutionResult with execution metrics
        """
        timeout = timeout or self.DEFAULT_TIMEOUT

        # Check if goose is available
        if not self.is_available():
            logger.warning("Goose not available, returning mock result")
            return self._mock_result(agent_spec, test_input)

        # Write agent spec to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            agent_file = Path(f.name)
            f.write(agent_spec.to_markdown())

        # Write test input to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_file = Path(f.name)
            json.dump(test_input, f)

        try:
            # Run goose
            cmd = [
                self.goose_path,
                "run",
                str(agent_file),
                "--test", str(test_file),
                "--timeout", str(timeout),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10,
            )

            return self._parse_result(result)

        except subprocess.TimeoutExpired:
            return GooseExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout}s",
            )
        except Exception as e:
            return GooseExecutionResult(
                success=False,
                error=f"Execution failed: {e!s}",
            )
        finally:
            # Cleanup temp files
            agent_file.unlink(missing_ok=True)
            test_file.unlink(missing_ok=True)

    def _parse_result(self, result: subprocess.CompletedProcess) -> GooseExecutionResult:
        """Parse goose execution result."""
        try:
            # Try to parse as JSON
            output_data = json.loads(result.stdout)
            return GooseExecutionResult(
                success=output_data.get("success", False),
                output=output_data.get("output", ""),
                error=output_data.get("error", ""),
                tokens_used=output_data.get("tokens", 0),
                duration=output_data.get("duration", 0.0),
                validation_passed=output_data.get("validation", False),
            )
        except json.JSONDecodeError:
            # Fall back to plain text
            return GooseExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
            )

    def _mock_result(self, agent_spec: AgentSpec, test_input: dict) -> GooseExecutionResult:
        """Return a mock result when Goose is not available.

        This allows development/testing without Goose installed.
        Does basic validation of the agent spec.
        """
        # Basic validation
        errors = []

        if not agent_spec.name:
            errors.append("Missing agent name")

        if not agent_spec.description:
            errors.append("Missing agent description")

        if len(agent_spec.system_prompt) < 50:
            errors.append("System prompt too short")

        if not agent_spec.skills:
            errors.append("No skills specified")

        # Check if required skills match test input expectations
        required_skills = test_input.get("required_skills", [])
        missing_skills = set(required_skills) - set(agent_spec.skills)
        if missing_skills:
            errors.append(f"Missing required skills: {missing_skills}")

        return GooseExecutionResult(
            success=len(errors) == 0,
            output="Mock execution completed" if len(errors) == 0 else "",
            error="; ".join(errors),
            validation_passed=len(errors) == 0,
        )


async def run_goose_validation(
    agent_spec: AgentSpec,
    test_input: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    """Async wrapper for Goose validation.

    This is called at steps 3, 6, 9 for real execution validation.

    Args:
        agent_spec: The agent to validate
        test_input: Test input data
        step: Current step number (for logging)

    Returns:
        Validation result dict
    """
    runner = GooseRunner()
    result = runner.run(agent_spec, test_input)

    logger.info(
        "goose_validation step=%d success=%s tokens=%d",
        step, result.success, result.tokens_used,
    )

    return {
        "step": step,
        "success": result.success,
        "validation_passed": result.validation_passed,
        "tokens_used": result.tokens_used,
        "duration": result.duration,
        "error": result.error,
    }
