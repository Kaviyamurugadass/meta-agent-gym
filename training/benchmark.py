"""Literature / expert benchmark — scripted "perfect agent" per scenario.

For each scenario, author the ideal action sequence a domain expert would
take. Benchmark runs this sequence against the env and reports match-ratio
against ground truth.

Why judges love this:
    - Proves the env isn't gameable (expert trajectory scores high)
    - Gives an upper bound for "Trained" column in README
    - Makes grading intent legible

TEMPLATE: `EXPERT_TRAJECTORIES` is a placeholder. On finale day, fill one
entry per scenario with the real action sequence.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add project root to path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from models import Action, ActionCommand
from training.reward_backend import LocalBackend


@dataclass
class BenchmarkResult:
    scenario_name: str
    total_reward: float
    steps_taken: int
    success: bool
    match_ratio: float  # fraction of expected_findings the trajectory discovered
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# Expert trajectories — META-AGENT: Ideal action sequences for each scenario
# ---------------------------------------------------------------------------

EXPERT_TRAJECTORIES: dict[str, list[Action]] = {
    # -------------------------------------------------------------------------
    # Placeholder trajectories (template compatibility)
    # -------------------------------------------------------------------------
    "placeholder_easy": [
        Action(command=ActionCommand.INSPECT, args={"target": "placeholder"}, confidence=0.9),
        Action(command=ActionCommand.NOOP),
        Action(command=ActionCommand.NOOP),
        Action(command=ActionCommand.SUBMIT, justification="placeholder", confidence=0.9),
    ],

    # -------------------------------------------------------------------------
    # Phase 1: Easy tasks (single skill)
    # -------------------------------------------------------------------------
    "ws_easy_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "product-price-scraper"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Extract product prices from e-commerce pages. Use when you need pricing data from online stores."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "web-scraping"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a web scraping specialist. When extracting prices:\n1. Identify the price element on the page\n2. Extract the price value\n3. Handle errors gracefully\n4. Return structured data with name and price"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "da_easy_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "csv-row-counter"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Read CSV files and count rows. Use for quick data analysis tasks."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "csv-handler"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "haiku"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a CSV specialist. When processing CSV files:\n1. Read the file using the csv library\n2. Count the number of data rows (excluding header)\n3. Handle encoding errors gracefully\n4. Return the count"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "cr_easy_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "error-handling-reviewer"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Review code for missing error handling. Use when validating Python code for production readiness."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "code-reviewer"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a code reviewer focused on error handling. When reviewing:\n1. Check for missing try/except blocks\n2. Verify file operations have error handling\n3. Look for unchecked return values\n4. Report specific line numbers with issues"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "fi_easy_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "log-file-reader"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Read log files and display the first 50 lines with line numbers. Use for quick inspection of large log files."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "file-reader"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "haiku"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a file reading specialist. When reading log files:\n1. Open the file efficiently using a buffer for large files\n2. Read the first 50 lines\n3. Prepend line numbers to each line\n4. Handle encoding errors gracefully"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "fi_easy_002": [
        Action(command=ActionCommand.SET_NAME, args={"name": "json-file-writer"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Write structured JSON data to files with proper formatting. Use for saving configuration or data output."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "file-writer"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "haiku"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a file writing specialist. When writing JSON files:\n1. Ensure proper JSON formatting with 2-space indentation\n2. Validate data is JSON-serializable before writing\n3. Create parent directories if needed\n4. Handle write errors gracefully"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "an_easy_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "error-log-analyzer"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Analyze server logs to find ERROR lines with timestamps. Use for quick error auditing of log files."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "log-analyzer"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "haiku"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a log analysis specialist. When analyzing logs:\n1. Scan for lines containing ERROR level\n2. Extract timestamps from each error line\n3. Count total errors found\n4. List unique error messages with their frequency"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "ou_easy_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "markdown-report-generator"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Generate markdown summary reports from structured data. Use for creating readable documentation from data."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "report-generator"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "haiku"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a report generation specialist. When creating markdown reports:\n1. Add appropriate headers and subheaders\n2. Format tabular data as markdown tables\n3. Include summary metrics at the top\n4. Keep formatting clean and consistent"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    # -------------------------------------------------------------------------
    # Phase 2: Medium tasks (2-3 skills)
    # -------------------------------------------------------------------------
    "ws_medium_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "multi-page-scraper"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Extract product data from multiple e-commerce pages with pagination. Use when scraping paginated catalogs."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "web-scraping"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "html-parser"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "http-client"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a web scraping specialist. When scraping multi-page sites:\n1. Handle pagination by detecting next links\n2. Parse HTML to extract product data\n3. Implement rate limiting\n4. Return structured JSON with all products"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "da_medium_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "csv-analyzer"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Analyze CSV data with missing value handling and summary statistics. Use for data quality assessment."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "csv-handler"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "data-transformer"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a data analyst. When analyzing CSV files:\n1. Load the CSV and detect missing values\n2. Calculate summary statistics (mean, median, std)\n3. Handle missing data appropriately\n4. Generate a clear summary report"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "cr_medium_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "security-reviewer"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Review Python code for security anti-patterns and common vulnerabilities. Use for security auditing."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "code-reviewer"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "pattern-matcher"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a security-focused code reviewer. When reviewing for security:\n1. Check for SQL injection vulnerabilities\n2. Look for hardcoded credentials and secrets\n3. Detect unsafe deserialization (pickle, eval, exec)\n4. Identify improper input validation\n5. Report findings with severity levels"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "fi_medium_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "csv-to-json-converter"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Read CSV files, transform data structures, and write JSON output. Use for data format migration."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "file-reader"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "file-writer"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "data-transformer"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a data conversion specialist. When converting CSV to JSON:\n1. Read the CSV file and parse headers\n2. Transform each row into a JSON object\n3. Handle data type conversion (numbers, dates)\n4. Write formatted JSON output with proper indentation"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "an_medium_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "error-pattern-detector"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Analyze application logs to detect recurring error patterns and generate frequency counts. Use for log-based troubleshooting."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "log-analyzer"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "pattern-matcher"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a log analysis specialist. When detecting error patterns:\n1. Parse log entries and identify error lines\n2. Group similar errors by message pattern\n3. Count frequency of each error type\n4. Rank by occurrence and report the most common patterns"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    # -------------------------------------------------------------------------
    # Phase 3: Hard tasks (3-5 skills)
    # -------------------------------------------------------------------------
    "ws_hard_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "multi-site-scraper"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Scrape data from multiple e-commerce sites with normalization and error recovery. Use for competitive price monitoring."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "web-scraping"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "html-parser"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "http-client"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "json-parser"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "data-validator"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a multi-site scraping specialist. When scraping multiple sites:\n1. Handle different HTML structures per site\n2. Normalize data to common format\n3. Implement robust error handling and retries\n4. Respect rate limits\n5. Output consolidated JSON"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "da_hard_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "multi-source-data-pipeline"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Ingest CSV data from multiple sources, validate schemas, transform to unified format, and output JSON. Use for data integration tasks."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "csv-handler"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "data-transformer"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "data-validator"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "json-parser"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a data pipeline engineer. When processing multi-source data:\n1. Read CSV files from each source\n2. Validate each against expected schema (columns, types)\n3. Transform to a unified column format\n4. Merge and output as consolidated JSON\n5. Log any validation failures"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "cr_hard_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "bug-fix-reviewer"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Read code files, identify bugs and security issues, generate fixes, and write test cases. Use for automated code remediation."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "code-reviewer"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "code-fixer"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "test-generator"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "file-reader"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are an automated code remediation specialist. When fixing code:\n1. Read and understand the codebase\n2. Identify bugs and security vulnerabilities\n3. Generate minimal, targeted fixes\n4. Write test cases that verify the fixes\n5. Ensure tests cover edge cases"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "fi_hard_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "log-file-processor"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Process multiple log files, validate format, extract error patterns, and write consolidated reports. Use for batch log analysis."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "file-reader"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "file-writer"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "data-validator"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "log-analyzer"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are a log file processing specialist. When processing logs:\n1. Read multiple log files from the directory\n2. Validate each file's format (timestamp, level, message)\n3. Extract and categorize error entries\n4. Consolidate findings into a single report\n5. Write the report to disk"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "an_hard_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "log-trend-analyzer"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Aggregate log data from multiple sources, identify time-window patterns, and generate structured trend reports. Use for operational analysis."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "log-analyzer"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "pattern-matcher"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "data-aggregator"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "report-generator"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are an operational analysis specialist. When analyzing log trends:\n1. Aggregate log entries from multiple sources\n2. Group errors by time window (hourly/daily)\n3. Identify recurring patterns and spikes\n4. Detect anomalies using statistical thresholds\n5. Generate a structured report with trends and highlights"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    # -------------------------------------------------------------------------
    # Phase 4: Expert tasks (5+ skills)
    # -------------------------------------------------------------------------
    "ws_expert_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "comprehensive-scraper"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Full-featured web scraper with JavaScript rendering, rate limiting, and reporting. Use for production-scale data extraction."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "web-scraping"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "html-parser"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "http-client"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "data-validator"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "report-generator"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "opus"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are an expert web scraping engineer. When building production scrapers:\n1. Check robots.txt before scraping\n2. Handle JavaScript rendering with selenium\n3. Implement rate limiting per domain\n4. Validate all extracted data\n5. Handle errors with retries and fallbacks\n6. Generate comprehensive summary report\n7. Store results in structured JSON format"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "da_expert_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "data-pipeline-orchestrator"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Comprehensive data pipeline that ingests CSV/JSON, validates schemas, transforms, aggregates, and detects anomalies. Use for enterprise data integration."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "csv-handler"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "json-parser"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "data-transformer"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "data-validator"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "data-aggregator"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "opus"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are an expert data pipeline engineer. When building data pipelines:\n1. Ingest data from CSV and JSON sources\n2. Validate schemas for each source (column names, types, ranges)\n3. Transform to unified format handling inconsistencies\n4. Aggregate summary statistics across sources\n5. Detect anomalies using configurable statistical thresholds\n6. Handle missing fields and format variations gracefully\n7. Output consolidated results with anomaly flags"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],

    "ou_expert_001": [
        Action(command=ActionCommand.SET_NAME, args={"name": "dashboard-reporting-engine"}, confidence=0.95),
        Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Reporting agent that scrapes dashboards, aggregates metrics, generates HTML reports, and sends threshold-based alerts. Use for automated monitoring and reporting."}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "report-generator"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "notifier"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "data-aggregator"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "csv-handler"}, confidence=0.95),
        Action(command=ActionCommand.ADD_SKILL, args={"skill": "html-parser"}, confidence=0.95),
        Action(command=ActionCommand.SET_MODEL, args={"model": "opus"}, confidence=0.95),
        Action(command=ActionCommand.WRITE_PROMPT, args={
            "prompt": "You are an expert reporting and monitoring engineer. When building reporting systems:\n1. Scrape dashboard pages for metrics data\n2. Parse HTML to extract structured metrics\n3. Aggregate metrics across services and time periods\n4. Generate formatted HTML reports with charts and tables\n5. Check metrics against configurable thresholds\n6. Send alert notifications when thresholds are exceeded\n7. Handle partial failures gracefully (skip unavailable services)"
        }, confidence=0.95),
        Action(command=ActionCommand.SUBMIT, confidence=0.95),
    ],
}


def compute_match_ratio(observations, expected_findings: dict[str, Any]) -> float:  # type: ignore[no-untyped-def]
    """DOMAIN: override with keyword/semantic match.

    Default: checks whether each expected finding KEY appears anywhere in the
    observation history's summary text. Good enough for placeholder; real
    domains should check for specific values, not keys.
    """
    if not expected_findings:
        return 0.0
    corpus = " ".join(
        str(obs.summary) + " " + str(obs.latest_output or "")
        for obs in observations
    ).lower()
    hits = sum(1 for key in expected_findings if str(key).lower() in corpus)
    return hits / len(expected_findings)


def run_benchmark(
    scenario_name: str,
    trajectory: list[Action] | None = None,
    base_url: str | None = None,
) -> BenchmarkResult:
    """Run expert trajectory against the env, return result."""
    trajectory = trajectory or EXPERT_TRAJECTORIES.get(scenario_name)
    if trajectory is None:
        raise ValueError(f"No expert trajectory defined for {scenario_name}")

    backend = LocalBackend()
    total_reward, observations = backend.score(trajectory, scenario_name=scenario_name)

    # Pull expected_findings from the scenario library
    from server.tasks.scenarios import get_scenario
    scenario = get_scenario(scenario_name)
    expected = scenario.expected_findings if scenario else {}

    match = compute_match_ratio(observations, expected)
    final_obs = observations[-1]

    return BenchmarkResult(
        scenario_name=scenario_name,
        total_reward=total_reward,
        steps_taken=final_obs.step,
        success=final_obs.done and total_reward > 0,
        match_ratio=match,
        details={
            "truncated": final_obs.truncated,
            "final_reward_breakdown": dict(final_obs.reward_breakdown),
            "violations_seen": sum(
                len(obs.rule_violations) for obs in observations
            ),
        },
    )


def run_all(scenarios: list[str] | None = None) -> list[BenchmarkResult]:
    scenarios = scenarios or list(EXPERT_TRAJECTORIES.keys())
    return [run_benchmark(s) for s in scenarios]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run expert benchmark.")
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Specific scenarios to run. Default: all with expert trajectories.",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_all(args.scenarios)
    for r in results:
        print(
            f"{r.scenario_name:30s} "
            f"reward={r.total_reward:7.3f}  "
            f"steps={r.steps_taken:3d}  "
            f"match={r.match_ratio:4.2f}  "
            f"success={r.success}"
        )

    if args.output:
        payload = [vars(r) for r in results]
        Path(args.output).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\n==> {len(results)} results → {args.output}")


if __name__ == "__main__":
    main()
