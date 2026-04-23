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
            "skill_count": 1,
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
    TaskSpec(
        task_id="fi_easy_001",
        domain="files",
        difficulty="easy",
        problem_statement=(
            "Build an agent that reads a log file and displays the first 50 lines "
            "with line numbers. The agent should handle large files efficiently."
        ),
        max_steps=7,
        required_skills=["file-reader"],
        recommended_skills=[],
        user_preferences={
            "language": "python",
            "output_format": "numbered_lines",
        },
        citations=["https://docs.python.org/3/tutorial/inputoutput.html"],
        expected_findings={
            "skill_count": 1,
        },
        red_herrings=[
            "Don't load entire file into memory for large files",
            "Don't add log parsing unless explicitly requested",
        ],
    ),
    TaskSpec(
        task_id="fi_easy_002",
        domain="files",
        difficulty="easy",
        problem_statement=(
            "Build an agent that writes structured JSON output to a file "
            "with proper formatting and indentation."
        ),
        max_steps=7,
        required_skills=["file-writer"],
        recommended_skills=[],
        user_preferences={
            "language": "python",
            "output_format": "json",
        },
        citations=["https://docs.python.org/3/library/json.html"],
        expected_findings={
            "skill_count": 1,
        },
        red_herrings=[
            "Don't add file reading capabilities unless requested",
            "Simple write operation is sufficient",
        ],
    ),
    TaskSpec(
        task_id="an_easy_001",
        domain="analysis",
        difficulty="easy",
        problem_statement=(
            "Build an agent that analyzes server logs and identifies ERROR lines "
            "with their timestamps. The agent should count total errors and list unique error messages."
        ),
        max_steps=7,
        required_skills=["log-analyzer"],
        recommended_skills=[],
        user_preferences={
            "language": "python",
            "log_format": "standard",
        },
        citations=["https://docs.python.org/3/library/re.html"],
        expected_findings={
            "skill_count": 1,
        },
        red_herrings=[
            "Don't add pattern matching for complex multi-line errors",
            "Simple ERROR line extraction is sufficient",
        ],
    ),
    TaskSpec(
        task_id="ou_easy_001",
        domain="output",
        difficulty="easy",
        problem_statement=(
            "Build an agent that generates a markdown summary report "
            "from structured data. The report should include headers, tables, and key metrics."
        ),
        max_steps=7,
        required_skills=["report-generator"],
        recommended_skills=[],
        user_preferences={
            "language": "python",
            "output_format": "markdown",
        },
        citations=["https://www.markdownguide.org/"],
        expected_findings={
            "skill_count": 1,
        },
        red_herrings=[
            "Don't add data aggregation unless explicitly requested",
            "Focus on report formatting, not data processing",
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
            "skill_count": 2,
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
    TaskSpec(
        task_id="cr_medium_001",
        domain="code",
        difficulty="medium",
        problem_statement=(
            "Build an agent that reviews Python code for security anti-patterns "
            "and common vulnerabilities such as SQL injection, hardcoded credentials, "
            "and unsafe deserialization."
        ),
        max_steps=10,
        required_skills=["code-reviewer", "pattern-matcher"],
        recommended_skills=[],
        user_preferences={
            "language": "python",
            "focus": "security",
        },
        citations=["https://owasp.org/www-project-top-ten/"],
        expected_findings={
            "skill_count": 2,
        },
        red_herrings=[
            "Don't suggest performance optimizations",
            "Focus on security issues, not code style",
        ],
    ),
    TaskSpec(
        task_id="fi_medium_001",
        domain="files",
        difficulty="medium",
        problem_statement=(
            "Build an agent that reads a CSV file, transforms the data structure, "
            "and writes the results to a new JSON file with proper formatting."
        ),
        max_steps=10,
        required_skills=["file-reader", "file-writer", "data-transformer"],
        recommended_skills=["csv-handler"],
        user_preferences={
            "language": "python",
            "input_format": "csv",
            "output_format": "json",
        },
        citations=["https://docs.python.org/3/library/csv.html"],
        expected_findings={
            "skill_count": 3,
        },
        red_herrings=[
            "Don't add data validation unless explicitly requested",
            "Simple format conversion is sufficient",
        ],
    ),
    TaskSpec(
        task_id="an_medium_001",
        domain="analysis",
        difficulty="medium",
        problem_statement=(
            "Build an agent that analyzes application logs to detect recurring error patterns "
            "and generates frequency counts of each error type."
        ),
        max_steps=10,
        required_skills=["log-analyzer", "pattern-matcher"],
        recommended_skills=["data-aggregator"],
        user_preferences={
            "language": "python",
            "output_format": "frequency_table",
        },
        citations=["https://docs.python.org/3/library/collections.html"],
        expected_findings={
            "skill_count": 2,
        },
        red_herrings=[
            "Don't add real-time monitoring capabilities",
            "Historical analysis only, not streaming",
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
    TaskSpec(
        task_id="da_hard_001",
        domain="data",
        difficulty="hard",
        problem_statement=(
            "Build an agent that ingests CSV data from multiple sources, "
            "validates each against expected schemas, transforms to a unified format, "
            "and outputs consolidated JSON results."
        ),
        max_steps=15,
        required_skills=["csv-handler", "data-transformer", "data-validator", "json-parser"],
        recommended_skills=["data-aggregator"],
        user_preferences={
            "language": "python",
            "constraints": ["multi_source", "schema_validation"],
        },
        citations=["https://pandas.pydata.org/docs/"],
        expected_findings={
            "skill_count": 4,
        },
        red_herrings=[
            "Don't add database storage unless explicitly requested",
            "File-based output is sufficient",
        ],
    ),
    TaskSpec(
        task_id="cr_hard_001",
        domain="code",
        difficulty="hard",
        problem_statement=(
            "Build an agent that reads code files, identifies bugs and security issues, "
            "generates fixes for the problems found, and writes corresponding test cases."
        ),
        max_steps=15,
        required_skills=["code-reviewer", "code-fixer", "test-generator", "file-reader"],
        recommended_skills=["pattern-matcher"],
        user_preferences={
            "language": "python",
            "constraints": ["generate_fixes", "generate_tests"],
        },
        citations=["https://docs.python.org/3/library/unittest.html"],
        expected_findings={
            "skill_count": 4,
        },
        red_herrings=[
            "Don't rewrite the entire codebase — fix only identified issues",
            "Test cases should cover the specific bugs found",
        ],
    ),
    TaskSpec(
        task_id="fi_hard_001",
        domain="files",
        difficulty="hard",
        problem_statement=(
            "Build a file-processing agent that reads multiple log files, validates their format, "
            "extracts error patterns, and writes a consolidated error report."
        ),
        max_steps=15,
        required_skills=["file-reader", "file-writer", "data-validator", "log-analyzer"],
        recommended_skills=["pattern-matcher"],
        user_preferences={
            "language": "python",
            "constraints": ["multi_file", "format_validation"],
        },
        citations=["https://docs.python.org/3/library/pathlib.html"],
        expected_findings={
            "skill_count": 4,
        },
        red_herrings=[
            "Don't add real-time log monitoring",
            "Batch processing of existing files is sufficient",
        ],
    ),
    TaskSpec(
        task_id="an_hard_001",
        domain="analysis",
        difficulty="hard",
        problem_statement=(
            "Build an analysis agent that aggregates data from multiple log sources, "
            "identifies patterns across time windows, and generates a structured "
            "report with trend analysis and anomaly highlights."
        ),
        max_steps=15,
        required_skills=["log-analyzer", "pattern-matcher", "data-aggregator", "report-generator"],
        recommended_skills=["notifier"],
        user_preferences={
            "language": "python",
            "output_format": "structured_report",
            "constraints": ["time_window_analysis", "anomaly_detection"],
        },
        citations=["https://docs.python.org/3/library/datetime.html"],
        expected_findings={
            "skill_count": 4,
        },
        red_herrings=[
            "Don't add machine learning models for anomaly detection",
            "Statistical threshold-based detection is sufficient",
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
    TaskSpec(
        task_id="da_expert_001",
        domain="data",
        difficulty="expert",
        problem_statement=(
            "Build a comprehensive data pipeline agent that: ingests data from CSV and JSON sources, "
            "validates schemas, transforms to unified format, aggregates summary statistics, "
            "and detects anomalies with configurable thresholds. Must handle inconsistent formats "
            "and missing fields across sources."
        ),
        max_steps=20,
        required_skills=[
            "csv-handler",
            "json-parser",
            "data-transformer",
            "data-validator",
            "data-aggregator",
        ],
        recommended_skills=["report-generator", "notifier"],
        user_preferences={
            "language": "python",
            "constraints": ["multi_source", "schema_validation", "anomaly_detection"],
        },
        citations=["https://pandas.pydata.org/docs/", "https://docs.python.org/3/library/json.html"],
        expected_findings={
            "skill_count": 5,
        },
        red_herrings=[
            "Anomaly detection should use statistical thresholds, not ML models",
            "Don't add database connectors — file-based I/O only",
            "Real-time streaming is NOT required",
        ],
    ),
    TaskSpec(
        task_id="ou_expert_001",
        domain="output",
        difficulty="expert",
        problem_statement=(
            "Build a reporting agent that: scrapes dashboard data from internal web pages, "
            "aggregates metrics across services, generates formatted HTML reports, "
            "and sends notification alerts when metrics exceed configurable thresholds. "
            "Must handle multiple data formats and partial failures gracefully."
        ),
        max_steps=20,
        required_skills=[
            "report-generator",
            "notifier",
            "data-aggregator",
            "csv-handler",
            "html-parser",
        ],
        recommended_skills=["http-client", "json-parser"],
        user_preferences={
            "language": "python",
            "constraints": ["html_reports", "threshold_alerting", "multi_format"],
        },
        citations=["https://developer.mozilla.org/en-US/docs/Web/HTML"],
        expected_findings={
            "skill_count": 5,
        },
        red_herrings=[
            "Don't add a web server — this is a batch reporting tool",
            "Email notifications are sufficient, don't add Slack/SMS integrations",
            "HTML report generation is more important than real-time monitoring",
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


def register_scenarios(extra: list[TaskSpec]) -> None:
    """Register additional scenarios (e.g., adversarial tasks) at runtime."""
    SCENARIOS.extend(extra)
