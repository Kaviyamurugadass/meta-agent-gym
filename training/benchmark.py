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
