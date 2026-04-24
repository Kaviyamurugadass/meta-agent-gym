"""Option B harness: run AGENT.md files through Goose + grade against expected output.

Standalone evaluation — does NOT couple to the training reward loop.
Uses the Goose CLI (configured with claude-code provider) to actually execute
generated agents against deterministic filesystem tasks, then checks whether
the output contains expected substrings.

Usage:
    # Smoke test with hand-written reference agents
    python -m evaluation.goose_execution --smoke

    # Run a specific AGENT.md against all tasks
    python -m evaluation.goose_execution --agent path/to/agent.md
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import AgentSpec, ModelType

FIXTURES = Path(__file__).parent / "fixtures"
INPUTS = FIXTURES / "inputs"
REFERENCE_AGENTS = FIXTURES / "reference_agents"


# ---------------------------------------------------------------------------
# Test tasks — each task points at a fixture file and specifies expected output
# ---------------------------------------------------------------------------

@dataclass
class TestTask:
    task_id: str
    prompt: str
    expected_substrings: list[str]
    reference_agent: str  # filename under reference_agents/
    max_turns: int = 8


def _build_tasks() -> list[TestTask]:
    return [
        TestTask(
            task_id="extract_price",
            prompt=(
                f"Read the file at '{(INPUTS / 'product.html').as_posix()}'. "
                "Find the product price. "
                "Output ONLY the price value with its currency symbol and nothing else."
            ),
            expected_substrings=["$19.99"],
            reference_agent="price_extractor.md",
        ),
        TestTask(
            task_id="count_rows",
            prompt=(
                f"Read the file at '{(INPUTS / 'data.csv').as_posix()}'. "
                "Count the number of data rows, excluding the header row. "
                "Output ONLY the count as a single integer."
            ),
            expected_substrings=["5"],
            reference_agent="csv_counter.md",
        ),
        TestTask(
            task_id="find_emails",
            prompt=(
                f"Read the file at '{(INPUTS / 'emails.txt').as_posix()}'. "
                "Find every email address. "
                "Output them one per line, nothing else."
            ),
            expected_substrings=[
                "alice@example.com",
                "bob@company.org",
                "charlie@test.io",
            ],
            reference_agent="email_finder.md",
        ),
    ]


TEST_TASKS = _build_tasks()


# ---------------------------------------------------------------------------
# AgentSpec → Goose recipe adapter
# ---------------------------------------------------------------------------

def agent_spec_to_recipe(spec: AgentSpec, task_prompt: str) -> dict:
    """Convert AgentSpec + a task prompt into a valid Goose recipe dict."""
    instructions = spec.system_prompt
    if spec.skills:
        instructions += f"\n\nAvailable capabilities: {', '.join(spec.skills)}"

    return {
        "version": "1.0.0",
        "title": spec.name or "generated-agent",
        "description": spec.description or "Generated meta-agent",
        "instructions": instructions,
        "prompt": task_prompt,
    }


def load_agent_md(path: Path) -> AgentSpec:
    """Parse AGENT.md (YAML frontmatter + body) into AgentSpec."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing YAML frontmatter")

    _, frontmatter_str, body = text.split("---", 2)
    frontmatter = yaml.safe_load(frontmatter_str) or {}

    model_str = str(frontmatter.get("model", "sonnet")).lower()
    try:
        model = ModelType(model_str)
    except ValueError:
        model = ModelType.SONNET

    allowed_tools_raw = frontmatter.get("allowed-tools")
    allowed_tools: Optional[list[str]] = None
    if isinstance(allowed_tools_raw, str):
        allowed_tools = [t.strip() for t in allowed_tools_raw.split(",") if t.strip()]
    elif isinstance(allowed_tools_raw, list):
        allowed_tools = list(allowed_tools_raw)

    return AgentSpec(
        name=str(frontmatter.get("name", "unnamed")),
        description=str(frontmatter.get("description", "")),
        skills=list(frontmatter.get("skills") or []),
        model=model,
        system_prompt=body.strip(),
        allowed_tools=allowed_tools,
    )


# ---------------------------------------------------------------------------
# Goose runner
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    task_id: str
    spec_name: str
    passed: bool
    output: str = ""
    error: Optional[str] = None
    duration_s: float = 0.0
    missing: list[str] = field(default_factory=list)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _goose_env() -> dict[str, str]:
    """Build a clean env for Goose — strips stale OPENAI_* vars that would
    otherwise override the persisted claude-code provider config.
    Force-set (not setdefault) so stale values in os.environ can't win."""
    env = dict(os.environ)
    for stale in ("OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_HOST",
                  "OPENAI_BASE_URL", "OPENAI_BASE_PATH"):
        env.pop(stale, None)
    env["GOOSE_PROVIDER"] = "claude-code"
    env["GOOSE_MODEL"] = "sonnet"
    env["CLAUDE_CODE_COMMAND"] = (
        r"C:\Users\Kaviya\AppData\Roaming\npm\node_modules\@anthropic-ai\claude-code\bin\claude.exe"
    )
    return env


def run_goose(recipe: dict, max_turns: int, timeout: int) -> tuple[str, str, int]:
    """Write recipe, invoke goose, return (stdout, stderr, returncode)."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    try:
        yaml.safe_dump(recipe, tmp, sort_keys=False, allow_unicode=True)
        tmp.close()
        result = subprocess.run(
            [
                "goose",
                "run",
                "--recipe",
                tmp.name,
                "--quiet",
                "--no-session",
                "--max-turns",
                str(max_turns),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_goose_env(),
        )
        return _strip_ansi(result.stdout), _strip_ansi(result.stderr), result.returncode
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _grade(output: str, task: TestTask) -> tuple[bool, list[str]]:
    low = output.lower()
    missing = [s for s in task.expected_substrings if s.lower() not in low]
    return (not missing), missing


def run_one(spec: AgentSpec, task: TestTask, timeout: int = 180) -> RunResult:
    recipe = agent_spec_to_recipe(spec, task.prompt)
    t0 = time.time()
    try:
        stdout, stderr, rc = run_goose(recipe, task.max_turns, timeout)
    except subprocess.TimeoutExpired:
        return RunResult(
            task_id=task.task_id,
            spec_name=spec.name,
            passed=False,
            error=f"timeout after {timeout}s",
            duration_s=time.time() - t0,
        )

    duration = time.time() - t0
    passed, missing = _grade(stdout, task)
    passed = passed and rc == 0
    err = stderr.strip()[:500] if rc != 0 and stderr.strip() else None
    return RunResult(
        task_id=task.task_id,
        spec_name=spec.name,
        passed=passed,
        output=stdout,
        error=err,
        duration_s=duration,
        missing=missing,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(results: list[RunResult]) -> None:
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r.passed)
    print(f"RESULT: {passed}/{len(results)} tasks passed\n")
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.task_id:20s} {r.duration_s:6.1f}s  ({r.spec_name})")
        if not r.passed:
            if r.error:
                print(f"         error: {r.error}")
            if r.missing:
                print(f"         missing: {r.missing}")


def cmd_smoke() -> int:
    """Run hand-written reference AGENT.md files against each task."""
    print(f"Smoke test — reference agents from {REFERENCE_AGENTS}\n")
    results: list[RunResult] = []
    for task in TEST_TASKS:
        ref_path = REFERENCE_AGENTS / task.reference_agent
        if not ref_path.exists():
            print(f"[SKIP] {task.task_id}: reference agent missing at {ref_path}")
            continue
        spec = load_agent_md(ref_path)
        print(f"Running {task.task_id} with spec '{spec.name}' ...", flush=True)
        result = run_one(spec, task)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  -> {status} ({result.duration_s:.1f}s)")
        if not result.passed:
            head = (result.output or "")[:200].replace("\n", " ")
            print(f"     output head: {head!r}")

    _print_summary(results)
    return 0 if all(r.passed for r in results) else 1


def cmd_agent(agent_path: Path) -> int:
    """Run a single AGENT.md against all tasks."""
    spec = load_agent_md(agent_path)
    print(f"Evaluating spec '{spec.name}' from {agent_path}\n")
    results: list[RunResult] = []
    for task in TEST_TASKS:
        print(f"Running {task.task_id} ...", flush=True)
        result = run_one(spec, task)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  -> {status} ({result.duration_s:.1f}s)")

    _print_summary(results)
    return 0 if all(r.passed for r in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--smoke", action="store_true",
                        help="Run hand-written reference agents against each task")
    parser.add_argument("--agent", type=Path, default=None,
                        help="Path to an AGENT.md file to evaluate")
    args = parser.parse_args()

    if args.smoke:
        return cmd_smoke()
    if args.agent:
        return cmd_agent(args.agent)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
