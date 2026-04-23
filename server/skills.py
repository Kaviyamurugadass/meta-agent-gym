"""Skill registry for meta-agent environment.

Defines available skills, their categories, and mappings to task types.
Used for curriculum progression (1 skill → 2-3 skills → 3-5 skills → 5+ skills).
"""

from __future__ import annotations

# Common skills from skills.sh + categorization
AVAILABLE_SKILLS: dict[str, str] = {
    # Web & Data
    "web-scraping": "Extract data from websites",
    "http-client": "Make HTTP requests",
    "html-parser": "Parse HTML content",
    "json-parser": "Parse JSON data",
    "csv-handler": "Handle CSV files",

    # Data Processing
    "data-transformer": "Transform data structures",
    "data-validator": "Validate data against schemas",
    "data-aggregator": "Aggregate and summarize data",

    # Code
    "code-reviewer": "Review code for quality",
    "code-fixer": "Fix common code issues",
    "test-generator": "Generate test cases",

    # Files
    "file-reader": "Read file contents",
    "file-writer": "Write file contents",

    # Analysis
    "log-analyzer": "Analyze log files",
    "pattern-matcher": "Match patterns in data",

    # Output
    "report-generator": "Generate reports",
    "notifier": "Send notifications",
}


# Skill categories for curriculum progression
SKILL_CATEGORIES: dict[str, list[str]] = {
    "web": ["web-scraping", "http-client", "html-parser"],
    "data": ["json-parser", "csv-handler", "data-transformer", "data-validator"],
    "code": ["code-reviewer", "code-fixer", "test-generator"],
    "files": ["file-reader", "file-writer"],
    "analysis": ["log-analyzer", "pattern-matcher"],
    "output": ["report-generator", "notifier"],
}


# Suggest skills for task types (for curriculum)
TASK_SKILL_MAP: dict[str, list[str]] = {
    "web_scraping": ["web-scraping", "html-parser", "http-client"],
    "data_analysis": ["csv-handler", "data-transformer", "data-validator"],
    "code_review": ["code-reviewer", "file-reader", "pattern-matcher"],
    "testing": ["test-generator", "code-fixer", "file-reader"],
    "log_analysis": ["log-analyzer", "pattern-matcher", "file-reader"],
    "reporting": ["report-generator", "data-aggregator", "csv-handler"],
}


# Domain templates for generating system prompts
AGENT_TEMPLATES: dict[str, str] = {
    "web-scraper": """You are a web scraping specialist.

Best practices:
- Always check robots.txt
- Implement rate limiting
- Handle errors gracefully
- Validate extracted data
- Store results in structured format

When scraping:
1. Identify target structure
2. Select appropriate extraction method
3. Handle pagination
4. Store results
""",

    "data-analyst": """You are a data analysis specialist.

Best practices:
- Validate data before analysis
- Document assumptions
- Handle missing data
- Provide actionable insights

When analyzing:
1. Understand data structure
2. Clean and validate
3. Explore patterns
4. Generate insights
""",

    "code-reviewer": """You are a code review specialist.

Best practices:
- Focus on code quality
- Check for security issues
- Verify error handling
- Assess performance

Review checklist:
- Code is clear and readable
- Proper error handling
- No security vulnerabilities
- Good test coverage
""",

    "api-integrator": """You are an API integration specialist.

Best practices:
- Validate inputs
- Handle API errors
- Implement retry logic
- Cache responses

When integrating:
1. Read API documentation
2. Design error handling
3. Implement retry logic
4. Add caching
""",
}


def get_skills_for_domain(domain: str) -> list[str]:
    """Get recommended skills for a domain."""
    return SKILL_CATEGORIES.get(domain, [])


def get_skills_for_task_type(task_type: str) -> list[str]:
    """Get required skills for a task type."""
    return TASK_SKILL_MAP.get(task_type, [])


def get_template_for_domain(domain: str) -> str | None:
    """Get system prompt template for a domain."""
    # Map domain to template key
    template_map = {
        "web": "web-scraper",
        "data": "data-analyst",
        "code": "code-reviewer",
        "api": "api-integrator",
    }
    key = template_map.get(domain)
    return AGENT_TEMPLATES.get(key) if key else None


def get_curriculum_skills(phase: int) -> dict[str, list[str]]:
    """Get skill recommendations for each curriculum phase.

    Args:
        phase: 1 (1 skill), 2 (2-3 skills), 3 (3-5 skills), 4 (5+ skills)

    Returns:
        dict mapping domain to list of skills for that phase
    """
    if phase == 1:
        # Single skill tasks
        return {
            "web": ["web-scraping"],
            "data": ["csv-handler"],
            "code": ["code-reviewer"],
        }
    elif phase == 2:
        # 2-3 skills
        return {
            "web": ["web-scraping", "html-parser", "http-client"],
            "data": ["csv-handler", "data-transformer"],
            "code": ["code-reviewer", "pattern-matcher"],
        }
    elif phase == 3:
        # 3-5 skills
        return {
            "web": ["web-scraping", "html-parser", "http-client", "json-parser"],
            "data": ["csv-handler", "data-transformer", "data-validator", "report-generator"],
            "code": ["code-reviewer", "pattern-matcher", "file-reader", "test-generator"],
        }
    else:
        # 5+ skills
        return {
            "web": list(SKILL_CATEGORIES["web"]) + ["json-parser"],
            "data": list(SKILL_CATEGORIES["data"]) + ["file-reader", "report-generator"],
            "code": list(SKILL_CATEGORIES["code"]) + ["log-analyzer"],
        }
