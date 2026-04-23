"""Scenario library — one TaskSpec per curated problem.

Curriculum-based test cases for meta-agent training:
    Phase 1 (easy): Single skill tasks
    Phase 2 (medium): 2-3 skills
    Phase 3 (hard): 3-5 skills with tradeoffs
    Phase 4 (expert): 5+ skills with red herrings
"""

from __future__ import annotations

from models import TaskSpec


# ---------------------------------------------------------------------------
# Placeholder scenarios (for template compatibility)
# ---------------------------------------------------------------------------

PLACEHOLDER_SCENARIOS: list[TaskSpec] = [
    TaskSpec(
        task_id="placeholder_easy",
        domain="placeholder",
        difficulty="easy",
        problem_statement="Placeholder task for template compatibility.",
        max_steps=5,
        expected_findings={},
    ),
]


# ---------------------------------------------------------------------------
# Phase 1: Easy tasks (single skill)
# ---------------------------------------------------------------------------

PHASE_1_SCENARIOS: list[TaskSpec] = [
    TaskSpec(
        task_id="ws_easy_001",
        domain="web",
        difficulty="easy",
        problem_statement=(
            "Build an agent that extracts product prices from a single e-commerce page. "
            "The agent should identify the price element and return the price value."
        ),
        max_steps=7,
        required_skills=["web-scraping"],
        recommended_skills=[],
        user_preferences={
            "language": "python",
            "constraints": ["must_handle_errors", "respect_rate_limits"],
        },
        citations=["https://developer.mozilla.org/en-US/docs/Web/HTML"],
        expected_findings={
            "skill_count": 1,  # Number of required skills
        },
        red_herrings=[
            "Don't add selenium unless task explicitly requires JavaScript rendering",
            "Don't add rate limiting if task is single-page only",
        ],
    ),
    TaskSpec(
        task_id="da_easy_001",
        domain="data",
        difficulty="easy",
        problem_statement=(
            "Build an agent that reads a CSV file and counts the number of rows."
        ),
        max_steps=7,
        required_skills=["csv-handler"],
        recommended_skills=[],
        user_preferences={
            "language": "python",
        },
        citations=["https://docs.python.org/3/library/csv.html"],
        expected_findings={
            "skill_count": 1,
        },
        red_herrings=[
            "Don't add data transformation unless explicitly requested",
            "Simple row count is sufficient",
        ],
    ),
    TaskSpec(
        task_id="cr_easy_001",
        domain="code",
        difficulty="easy",
        problem_statement=(
            "Build an agent that reviews code for missing error handling."
        ),
        max_steps=7,
        required_skills=["code-reviewer"],
        recommended_skills=[],
        user_preferences={
            "focus": "error_handling",
        },
        citations=["https://peps.python.org/pep-0008/"],
        expected_findings={
            "skill_count": 1,
        },
        red_herrings=[
            "Don't suggest refactoring working code",
            "Focus only on error handling, not other issues",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Phase 2: Medium tasks (2-3 skills)
# ---------------------------------------------------------------------------

PHASE_2_SCENARIOS: list[TaskSpec] = [
    TaskSpec(
        task_id="ws_medium_001",
        domain="web",
        difficulty="medium",
        problem_statement=(
            "Build an agent that extracts product data from multiple pages of an e-commerce site. "
            "The agent should handle pagination and return structured data."
        ),
        max_steps=10,
        required_skills=["web-scraping", "html-parser"],
        recommended_skills=["http-client"],
        user_preferences={
            "language": "python",
            "constraints": ["handle_pagination", "rate_limiting"],
        },
        citations=["https://developer.mozilla.org/en-US/docs/Web/HTML"],
        expected_findings={
            "skill_count": 2,  # web-scraping, html-parser
        },
        red_herrings=[
            "Don't add selenium unless JavaScript is explicitly required",
        ],
    ),
    TaskSpec(
        task_id="da_medium_001",
        domain="data",
        difficulty="medium",
        problem_statement=(
            "Build an agent that analyzes CSV data, handles missing values, "
            "and generates summary statistics."
        ),
        max_steps=10,
        required_skills=["csv-handler", "data-transformer"],
        recommended_skills=["data-validator"],
        user_preferences={
            "language": "python",
            "output_format": "report",
        },
        citations=["https://pandas.pydata.org/docs/"],
        expected_findings={
            "skill_count": 2,
        },
        red_herrings=[
            "Don't add visualization unless explicitly requested",
            "Simple imputation is acceptable",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Phase 3: Hard tasks (3-5 skills)
# ---------------------------------------------------------------------------

PHASE_3_SCENARIOS: list[TaskSpec] = [
    TaskSpec(
        task_id="ws_hard_001",
        domain="web",
        difficulty="hard",
        problem_statement=(
            "Build an agent that scrapes data from multiple e-commerce sites, "
            "normalizes the data format, handles errors and rate limits, "
            "and stores results in JSON format."
        ),
        max_steps=15,
        required_skills=["web-scraping", "html-parser", "http-client", "json-parser"],
        recommended_skills=["data-validator"],
        user_preferences={
            "language": "python",
            "constraints": ["multi_site", "rate_limiting", "error_recovery"],
        },
        citations=["https://developer.mozilla.org/en-US/docs/Web/HTML"],
        expected_findings={
            "skill_count": 4,
        },
        red_herrings=[
            "Different sites may have different HTML structures — normalization is expected",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Phase 4: Expert tasks (5+ skills with red herrings)
# ---------------------------------------------------------------------------

PHASE_4_SCENARIOS: list[TaskSpec] = [
    TaskSpec(
        task_id="ws_expert_001",
        domain="web",
        difficulty="expert",
        problem_statement=(
            "Build a comprehensive web scraping agent that: extracts data from "
            "multiple sites, handles JavaScript rendering, respects robots.txt, "
            "implements rate limiting, validates data, handles errors gracefully, "
            "and generates a summary report."
        ),
        max_steps=20,
        required_skills=[
            "web-scraping",
            "html-parser",
            "http-client",
            "data-validator",
            "report-generator",
        ],
        recommended_skills=["json-parser", "notifier"],
        user_preferences={
            "language": "python",
            "constraints": ["javascript_rendering", "multi_site", "comprehensive"],
        },
        citations=["https://developer.mozilla.org/en-US/docs/Web/HTML"],
        expected_findings={
            "skill_count": 5,
        },
        red_herrings=[
            "Selenium/JavaScript rendering is REQUIRED for this task (unlike easier tasks)",
            "Rate limiting is required even for single-site scraping",
        ],
    ),
]


# ---------------------------------------------------------------------------
# All scenarios combined
# ---------------------------------------------------------------------------

SCENARIOS: list[TaskSpec] = [
    *PLACEHOLDER_SCENARIOS,  # Keep placeholders for template compatibility
    *PHASE_1_SCENARIOS,
    *PHASE_2_SCENARIOS,
    *PHASE_3_SCENARIOS,
    *PHASE_4_SCENARIOS,
]


def get_scenario(name: str) -> TaskSpec | None:
    """Lookup by task_id. Returns None if not found."""
    for scenario in SCENARIOS:
        if scenario.task_id == name:
            return scenario
    return None


def get_scenarios_by_phase(phase: int) -> list[TaskSpec]:
    """Get scenarios for a specific curriculum phase."""
    if phase == 1:
        return PHASE_1_SCENARIOS
    elif phase == 2:
        return PHASE_2_SCENARIOS
    elif phase == 3:
        return PHASE_3_SCENARIOS
    elif phase == 4:
        return PHASE_4_SCENARIOS
    return []


def get_scenarios_by_difficulty(difficulty: str) -> list[TaskSpec]:
    """Get scenarios by difficulty level."""
    return [s for s in SCENARIOS if s.difficulty == difficulty]
