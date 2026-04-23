"""Adversarial designer — generates hard test cases targeting policy weaknesses.

Each strategy produces a TaskSpec with built-in traps that expose specific
failure modes in the trained policy:

    skill_trap           — counter-intuitive required skills
    model_mismatch       — misleading task complexity vs. model requirement
    over_engineering_bait — vague requirements tempt too many skills
    red_herring_trap     — plausible but wrong skills in problem statement
    vague_requirements   — minimal input forces clear spec from ambiguity
    boundary_case        — tests the 5-skill efficiency drop boundary
    cross_domain         — combines 3+ skill domains

CLI:
    python server/adversarial.py --count 14 --output adversarial.json
    python server/adversarial.py --strategies skill_trap,red_herring_trap --count 6
"""

from __future__ import annotations

import argparse
import json
import random
from enum import Enum
from pathlib import Path
from typing import Optional
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from models import TaskSpec
from server.skills import AVAILABLE_SKILLS, SKILL_CATEGORIES


class AdversarialStrategy(str, Enum):
    SKILL_TRAP = "skill_trap"
    MODEL_MISMATCH = "model_mismatch"
    OVER_ENGINEERING_BAIT = "over_engineering_bait"
    RED_HERRING_TRAP = "red_herring_trap"
    VAGUE_REQUIREMENTS = "vague_requirements"
    BOUNDARY_CASE = "boundary_case"
    CROSS_DOMAIN = "cross_domain"


# ---------------------------------------------------------------------------
# Strategy generators
# ---------------------------------------------------------------------------

_SKILL_TRAP_TEMPLATES: list[dict] = [
    {
        "id": "adv_skill_trap_001",
        "domain": "data",
        "problem_statement": (
            "Build an agent that reads a simple configuration file and extracts "
            "the database connection settings."
        ),
        "required_skills": ["json-parser"],
        "recommended_skills": [],
        "red_herrings": [
            "The config file is JSON, not plain text — file-reader alone won't parse it",
            "Don't add file-writer — this is read-only",
        ],
        "expected_findings": {"skill_count": 1, "trap": "json_not_file"},
    },
    {
        "id": "adv_skill_trap_002",
        "domain": "web",
        "problem_statement": (
            "Build an agent that monitors an API endpoint and alerts when "
            "the response status code changes from 200."
        ),
        "required_skills": ["http-client"],
        "recommended_skills": ["notifier"],
        "red_herrings": [
            "Don't add web-scraping — this is an API, not a web page",
            "HTML parsing is not needed for JSON API responses",
        ],
        "expected_findings": {"skill_count": 1, "trap": "api_not_web"},
    },
    {
        "id": "adv_skill_trap_003",
        "domain": "code",
        "problem_statement": (
            "Build an agent that finds all TODO comments in a Python codebase "
            "and lists them with file names and line numbers."
        ),
        "required_skills": ["pattern-matcher"],
        "recommended_skills": ["file-reader"],
        "red_herrings": [
            "Don't add code-reviewer — this is pattern matching, not code review",
            "Don't add test-generator — no tests needed for a search tool",
        ],
        "expected_findings": {"skill_count": 1, "trap": "search_not_review"},
    },
]

_MODEL_MISMATCH_TEMPLATES: list[dict] = [
    {
        "id": "adv_model_mismatch_001",
        "domain": "analysis",
        "problem_statement": (
            "Build a simple agent that counts words in a text file and returns "
            "the top 5 most frequent words."
        ),
        "required_skills": ["file-reader"],
        "recommended_skills": [],
        "red_herrings": [
            "Don't use opus — this is a basic counting task",
            "Haiku is sufficient for word frequency counting",
        ],
        "expected_findings": {"skill_count": 1, "model": "haiku", "trap": "easy_task"},
    },
    {
        "id": "adv_model_mismatch_002",
        "domain": "code",
        "problem_statement": (
            "Build an agent that analyzes a large codebase for security vulnerabilities, "
            "cross-references findings across multiple files, identifies attack chains, "
            "and generates a prioritized remediation plan with dependency analysis."
        ),
        "required_skills": ["code-reviewer", "pattern-matcher", "file-reader"],
        "recommended_skills": ["report-generator"],
        "red_herrings": [
            "This requires deep analysis — haiku is not sufficient",
            "Multiple files must be cross-referenced, not reviewed in isolation",
        ],
        "expected_findings": {"skill_count": 3, "model": "opus", "trap": "hard_task_easy_sound"},
    },
]

_OVER_ENGINEERING_BAIT_TEMPLATES: list[dict] = [
    {
        "id": "adv_over_engineering_bait_001",
        "domain": "data",
        "problem_statement": (
            "Build an agent that helps with data tasks."
        ),
        "required_skills": ["csv-handler"],
        "recommended_skills": [],
        "red_herrings": [
            "The problem is intentionally vague — resist adding extra skills",
            "One skill is enough for 'data tasks' without specifics",
        ],
        "expected_findings": {"skill_count": 1, "trap": "vague_bait"},
    },
    {
        "id": "adv_over_engineering_bait_002",
        "domain": "web",
        "problem_statement": (
            "Build a comprehensive web agent that can do everything needed "
            "for web-related tasks."
        ),
        "required_skills": ["web-scraping"],
        "recommended_skills": ["html-parser"],
        "red_herrings": [
            "Despite 'comprehensive', only web-scraping is actually required",
            "Adding http-client and json-parser would be over-engineering",
        ],
        "expected_findings": {"skill_count": 1, "trap": "comprehensive_bait"},
    },
]

_RED_HERRING_TRAP_TEMPLATES: list[dict] = [
    {
        "id": "adv_red_herring_trap_001",
        "domain": "web",
        "problem_statement": (
            "Build an agent that scrapes a website using Selenium with JavaScript rendering, "
            "stores results in a PostgreSQL database, and sends email notifications. "
            "The site has a simple static HTML product listing."
        ),
        "required_skills": ["web-scraping", "html-parser"],
        "recommended_skills": [],
        "red_herrings": [
            "The site is STATIC HTML — no Selenium or JavaScript needed",
            "No database required — file output is sufficient",
            "Email notifications are not in the required skills",
        ],
        "expected_findings": {"skill_count": 2, "trap": "ignore_selenium_db_email"},
    },
    {
        "id": "adv_red_herring_trap_002",
        "domain": "data",
        "problem_statement": (
            "Build an agent using machine learning and AI to predict future trends "
            "from a CSV file. The CSV has 3 columns: date, category, and count. "
            "Just output the monthly totals."
        ),
        "required_skills": ["csv-handler"],
        "recommended_skills": ["data-aggregator"],
        "red_herrings": [
            "No ML needed — just sum the counts by month",
            "'Predict future trends' is a red herring — the task says 'output monthly totals'",
            "Don't add json-parser or data-transformer for simple aggregation",
        ],
        "expected_findings": {"skill_count": 1, "trap": "ml_red_herring"},
    },
]

_VAGUE_REQUIREMENTS_TEMPLATES: list[dict] = [
    {
        "id": "adv_vague_requirements_001",
        "domain": "files",
        "problem_statement": "Build an agent that processes files.",
        "required_skills": ["file-reader", "file-writer"],
        "recommended_skills": [],
        "red_herrings": [
            "The vague statement forces you to write a clear, general-purpose file processor",
            "Description must include delegation guidance without knowing specific file types",
        ],
        "expected_findings": {"skill_count": 2, "trap": "vague_forces_clarity"},
    },
    {
        "id": "adv_vague_requirements_002",
        "domain": "code",
        "problem_statement": "Fix the code.",
        "required_skills": ["code-reviewer", "code-fixer"],
        "recommended_skills": [],
        "red_herrings": [
            "Minimal input — the agent must be a general-purpose code fixer",
            "Workflow clarity in the prompt is critical with such vague input",
        ],
        "expected_findings": {"skill_count": 2, "trap": "vague_forces_workflow"},
    },
]

_BOUNDARY_CASE_TEMPLATES: list[dict] = [
    {
        "id": "adv_boundary_case_001",
        "domain": "web",
        "problem_statement": (
            "Build an agent that scrapes multiple websites, parses HTML and JSON responses, "
            "makes HTTP requests with rate limiting, and validates the extracted data."
        ),
        "required_skills": [
            "web-scraping", "html-parser", "http-client",
            "json-parser", "data-validator",
        ],
        "recommended_skills": [],
        "red_herrings": [
            "Exactly 5 skills are required — adding more drops efficiency below 0.5",
            "Don't add report-generator or notifier",
        ],
        "expected_findings": {"skill_count": 5, "trap": "exact_5_boundary"},
    },
    {
        "id": "adv_boundary_case_002",
        "domain": "data",
        "problem_statement": (
            "Build a data pipeline agent that reads CSV files, parses JSON configs, "
            "transforms data formats, validates schemas, and aggregates results."
        ),
        "required_skills": [
            "csv-handler", "json-parser", "data-transformer",
            "data-validator", "data-aggregator",
        ],
        "recommended_skills": [],
        "red_herrings": [
            "5 skills exactly — one more would trigger the efficiency penalty",
            "Don't add file-writer for output",
        ],
        "expected_findings": {"skill_count": 5, "trap": "exact_5_boundary"},
    },
]

_CROSS_DOMAIN_TEMPLATES: list[dict] = [
    {
        "id": "adv_cross_domain_001",
        "domain": "general",
        "problem_statement": (
            "Build an agent that scrapes monitoring dashboards for metrics, "
            "analyzes application logs for error patterns, and generates "
            "an incident report with recommendations."
        ),
        "required_skills": ["web-scraping", "log-analyzer", "report-generator"],
        "recommended_skills": ["html-parser", "pattern-matcher"],
        "red_herrings": [
            "Three different domains: web, analysis, and output",
            "Don't default to a single-domain solution",
        ],
        "expected_findings": {"skill_count": 3, "trap": "three_domains"},
    },
    {
        "id": "adv_cross_domain_002",
        "domain": "general",
        "problem_statement": (
            "Build an agent that reviews code files for security issues, "
            "reads test coverage reports in CSV format, and sends "
            "notifications when critical vulnerabilities are found."
        ),
        "required_skills": ["code-reviewer", "csv-handler", "notifier"],
        "recommended_skills": ["pattern-matcher"],
        "red_herrings": [
            "Code, data, and output domains combined",
            "Don't treat this as purely a code review task",
        ],
        "expected_findings": {"skill_count": 3, "trap": "three_domains"},
    },
]

_ALL_TEMPLATES: dict[AdversarialStrategy, list[dict]] = {
    AdversarialStrategy.SKILL_TRAP: _SKILL_TRAP_TEMPLATES,
    AdversarialStrategy.MODEL_MISMATCH: _MODEL_MISMATCH_TEMPLATES,
    AdversarialStrategy.OVER_ENGINEERING_BAIT: _OVER_ENGINEERING_BAIT_TEMPLATES,
    AdversarialStrategy.RED_HERRING_TRAP: _RED_HERRING_TRAP_TEMPLATES,
    AdversarialStrategy.VAGUE_REQUIREMENTS: _VAGUE_REQUIREMENTS_TEMPLATES,
    AdversarialStrategy.BOUNDARY_CASE: _BOUNDARY_CASE_TEMPLATES,
    AdversarialStrategy.CROSS_DOMAIN: _CROSS_DOMAIN_TEMPLATES,
}


def _template_to_taskspec(t: dict) -> TaskSpec:
    return TaskSpec(
        task_id=t["id"],
        domain=t.get("domain", "general"),
        difficulty=t.get("difficulty", "hard"),
        problem_statement=t["problem_statement"],
        max_steps=t.get("max_steps", 15),
        required_skills=t.get("required_skills", []),
        recommended_skills=t.get("recommended_skills", []),
        user_preferences=t.get("user_preferences", {"language": "python"}),
        citations=t.get("citations", []),
        expected_findings=t.get("expected_findings", {}),
        red_herrings=t.get("red_herrings", []),
    )


class AdversarialDesigner:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._history: list[dict] = []
        self._counters: dict[str, int] = {}

    def generate(self, strategy: Optional[AdversarialStrategy] = None) -> TaskSpec:
        if strategy is None:
            strategy = self._rng.choice(list(AdversarialStrategy))
        templates = _ALL_TEMPLATES.get(strategy, [])
        if not templates:
            raise ValueError(f"No templates for strategy {strategy}")
        template = self._rng.choice(templates)
        return _template_to_taskspec(template)

    def generate_batch(self, n: int = 14) -> list[TaskSpec]:
        strategies = list(AdversarialStrategy)
        tasks: list[TaskSpec] = []
        for i in range(n):
            strategy = strategies[i % len(strategies)]
            tasks.append(self.generate(strategy))
        return tasks

    def get_all_scenarios(self) -> list[TaskSpec]:
        all_tasks: list[TaskSpec] = []
        for templates in _ALL_TEMPLATES.values():
            for t in templates:
                all_tasks.append(_template_to_taskspec(t))
        return all_tasks

    def record_result(
        self,
        task_id: str,
        success: bool,
        reward_breakdown: dict[str, float],
    ) -> None:
        strategy_name = task_id.split("_")[1] if "_" in task_id else "unknown"
        self._history.append({
            "task_id": task_id,
            "strategy": strategy_name,
            "success": success,
            "reward_breakdown": reward_breakdown,
        })

    def get_weaknesses(self) -> dict[str, float]:
        if not self._history:
            return {s.value: 0.0 for s in AdversarialStrategy}
        by_strategy: dict[str, list[bool]] = {}
        for entry in self._history:
            s = entry["strategy"]
            by_strategy.setdefault(s, []).append(entry["success"])
        result: dict[str, float] = {}
        for s in AdversarialStrategy:
            outcomes = by_strategy.get(s.value, [])
            result[s.value] = sum(outcomes) / len(outcomes) if outcomes else 0.0
        return result

    def save(self, path: str | Path) -> None:
        data = {
            "history": self._history,
            "counters": self._counters,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AdversarialDesigner":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        designer = cls()
        designer._history = data.get("history", [])
        designer._counters = data.get("counters", {})
        return designer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate adversarial test cases.")
    parser.add_argument(
        "--count", type=int, default=14,
        help="Number of adversarial tasks to generate",
    )
    parser.add_argument(
        "--strategies", type=str, default=None,
        help="Comma-separated strategies (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (.json). Prints to stdout if omitted.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    designer = AdversarialDesigner(seed=args.seed)

    if args.strategies:
        strategies = [
            AdversarialStrategy(s.strip())
            for s in args.strategies.split(",")
        ]
        tasks = [designer.generate(s) for s in strategies * ((args.count // len(strategies)) + 1)]
        tasks = tasks[:args.count]
    else:
        tasks = designer.generate_batch(args.count)

    for t in tasks:
        print(
            f"{t.task_id:35s} domain={t.domain:10s} "
            f"skills={len(t.required_skills)} "
            f"difficulty={t.difficulty}"
        )

    if args.output:
        payload = [t.model_dump() for t in tasks]
        Path(args.output).write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n==> {len(tasks)} adversarial tasks saved to {args.output}")


if __name__ == "__main__":
    main()
