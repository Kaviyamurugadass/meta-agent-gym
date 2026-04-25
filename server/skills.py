"""Skill registry for meta-agent environment.

Skills are sourced from the Agent Skills Open Standard at https://skills.sh/ —
the real ecosystem used by Claude Code, Goose, Cursor, Copilot, and others.
Each skill maps to a real installable package: `npx skills add <owner/repo>`.

This means every AGENT.md our model generates contains REAL, deployable skills
that users can install and use immediately.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Real skills from skills.sh (https://skills.sh/)
# Format: "skill-id": "description  [owner/repo]"
# The model learns to pick these; the output AGENT.md is immediately deployable.
# ---------------------------------------------------------------------------

AVAILABLE_SKILLS: dict[str, str] = {
    # --- Web & Scraping ---
    "browser-use": "Control a real browser to interact with any website [browser-use/browser-use]",
    "firecrawl": "Scrape and crawl websites into clean markdown [firecrawl/cli]",
    "firecrawl-scrape": "Scrape a single URL into structured data [firecrawl/cli]",
    "firecrawl-search": "Search the web and return structured results [firecrawl/cli]",
    "firecrawl-crawl": "Crawl entire sites and return all pages [firecrawl/cli]",
    "web-design-guidelines": "Apply best-practice web design patterns [vercel-labs/agent-skills]",

    # --- Frontend & UI ---
    "frontend-design": "Build production-grade UI components and pages [anthropics/skills]",
    "ui-ux-pro-max": "Apply advanced UI/UX principles and patterns [nextlevelbuilder/ui-ux-pro-max-skill]",
    "react:components": "Generate and manage React components [google-labs-code/stitch-skills]",
    "shadcn": "Use shadcn/ui component library [shadcn/ui]",
    "sleek-design-mobile-apps": "Design polished mobile app interfaces [sleekdotdesign/agent-skills]",

    # --- Backend & APIs ---
    "vercel-react-best-practices": "Follow Vercel + React production patterns [vercel-labs/agent-skills]",
    "supabase-postgres-best-practices": "Use Supabase with Postgres correctly [supabase/agent-skills]",
    "supabase": "Integrate Supabase auth, DB, and storage [supabase/agent-skills]",
    "neon-postgres": "Connect and query Neon serverless Postgres [neondatabase/agent-skills]",
    "firebase-basics": "Set up and use Firebase services [firebase/agent-skills]",
    "convex-quickstart": "Build with Convex reactive backend [get-convex/agent-skills]",

    # --- Code Quality & Review ---
    "systematic-debugging": "Debug issues step-by-step with root cause analysis [obra/superpowers]",
    "test-driven-development": "Write tests before code, TDD workflow [obra/superpowers]",
    "requesting-code-review": "Prepare and request effective code reviews [obra/superpowers]",
    "receiving-code-review": "Process and apply code review feedback [obra/superpowers]",
    "playwright-best-practices": "Write robust Playwright end-to-end tests [currents-dev/playwright-cli]",
    "webapp-testing": "Test web applications thoroughly [anthropics/skills]",

    # --- Documents & Files ---
    "pdf": "Read, parse and generate PDF documents [anthropics/skills]",
    "docx": "Create and edit Word documents [anthropics/skills]",
    "xlsx": "Read and write Excel spreadsheets [anthropics/skills]",
    "pptx": "Generate PowerPoint presentations [anthropics/skills]",

    # --- AI & Agents ---
    "mcp-builder": "Build Model Context Protocol servers [anthropics/skills]",
    "self-improving-agent": "Agent that critiques and improves its own outputs [charon-fan/agent-playbook]",
    "subagent-driven-development": "Break tasks into parallel subagents [obra/superpowers]",
    "dispatching-parallel-agents": "Coordinate multiple agents concurrently [obra/superpowers]",
    "skill-creator": "Create new Agent Skills for the skills.sh ecosystem [anthropics/skills]",

    # --- Planning & Writing ---
    "brainstorming": "Generate and explore creative ideas [obra/superpowers]",
    "writing-plans": "Create structured plans before executing [obra/superpowers]",
    "executing-plans": "Follow through on structured plans reliably [obra/superpowers]",
    "writing-skills": "Write clear, effective prose and documentation [obra/superpowers]",
    "doc-coauthoring": "Collaboratively write and edit documents [anthropics/skills]",

    # --- DevOps & Infrastructure ---
    "github-actions-docs": "Write and debug GitHub Actions workflows [xixu-me/skills]",
    "sentry-cli": "Monitor errors and performance with Sentry [sentry/dev]",
    "deploy-to-vercel": "Deploy projects to Vercel [vercel-labs/agent-skills]",

    # --- Marketing & SEO ---
    "seo-audit": "Audit and improve site SEO [coreyhaines31/marketingskills]",
    "copywriting": "Write compelling marketing copy [coreyhaines31/marketingskills]",
    "social-content": "Create social media content [coreyhaines31/marketingskills]",

    # --- Productivity ---
    "obsidian-markdown": "Manage knowledge in Obsidian vaults [kepano/obsidian-skills]",
    "lark-doc": "Create and manage Lark/Feishu documents [larksuite/cli]",
    "gws-gmail": "Read and send Gmail messages [googleworkspace/cli]",
    "gws-drive": "Manage Google Drive files [googleworkspace/cli]",
}


# ---------------------------------------------------------------------------
# Skill → skills.sh install reference (for AGENT.md output)
# ---------------------------------------------------------------------------

SKILL_INSTALL_MAP: dict[str, str] = {
    "browser-use": "browser-use/browser-use",
    "firecrawl": "firecrawl/cli",
    "firecrawl-scrape": "firecrawl/cli",
    "firecrawl-search": "firecrawl/cli",
    "firecrawl-crawl": "firecrawl/cli",
    "web-design-guidelines": "vercel-labs/agent-skills",
    "frontend-design": "anthropics/skills",
    "ui-ux-pro-max": "nextlevelbuilder/ui-ux-pro-max-skill",
    "react:components": "google-labs-code/stitch-skills",
    "shadcn": "shadcn/ui",
    "sleek-design-mobile-apps": "sleekdotdesign/agent-skills",
    "vercel-react-best-practices": "vercel-labs/agent-skills",
    "supabase-postgres-best-practices": "supabase/agent-skills",
    "supabase": "supabase/agent-skills",
    "neon-postgres": "neondatabase/agent-skills",
    "firebase-basics": "firebase/agent-skills",
    "convex-quickstart": "get-convex/agent-skills",
    "systematic-debugging": "obra/superpowers",
    "test-driven-development": "obra/superpowers",
    "requesting-code-review": "obra/superpowers",
    "receiving-code-review": "obra/superpowers",
    "playwright-best-practices": "currents-dev/playwright-cli",
    "webapp-testing": "anthropics/skills",
    "pdf": "anthropics/skills",
    "docx": "anthropics/skills",
    "xlsx": "anthropics/skills",
    "pptx": "anthropics/skills",
    "mcp-builder": "anthropics/skills",
    "self-improving-agent": "charon-fan/agent-playbook",
    "subagent-driven-development": "obra/superpowers",
    "dispatching-parallel-agents": "obra/superpowers",
    "skill-creator": "anthropics/skills",
    "brainstorming": "obra/superpowers",
    "writing-plans": "obra/superpowers",
    "executing-plans": "obra/superpowers",
    "writing-skills": "obra/superpowers",
    "doc-coauthoring": "anthropics/skills",
    "github-actions-docs": "xixu-me/skills",
    "sentry-cli": "sentry/dev",
    "deploy-to-vercel": "vercel-labs/agent-skills",
    "seo-audit": "coreyhaines31/marketingskills",
    "copywriting": "coreyhaines31/marketingskills",
    "social-content": "coreyhaines31/marketingskills",
    "obsidian-markdown": "kepano/obsidian-skills",
    "lark-doc": "larksuite/cli",
    "gws-gmail": "googleworkspace/cli",
    "gws-drive": "googleworkspace/cli",
}


# ---------------------------------------------------------------------------
# Skill categories for curriculum progression
# ---------------------------------------------------------------------------

SKILL_CATEGORIES: dict[str, list[str]] = {
    "web": ["browser-use", "firecrawl", "firecrawl-scrape", "firecrawl-search", "firecrawl-crawl", "web-design-guidelines"],
    "frontend": ["frontend-design", "ui-ux-pro-max", "react:components", "shadcn", "sleek-design-mobile-apps"],
    "backend": ["supabase", "supabase-postgres-best-practices", "neon-postgres", "firebase-basics", "convex-quickstart"],
    "code": ["systematic-debugging", "test-driven-development", "requesting-code-review", "playwright-best-practices", "webapp-testing"],
    "documents": ["pdf", "docx", "xlsx", "pptx", "doc-coauthoring"],
    "agents": ["mcp-builder", "self-improving-agent", "subagent-driven-development", "skill-creator"],
    "planning": ["brainstorming", "writing-plans", "executing-plans", "writing-skills"],
    "devops": ["github-actions-docs", "sentry-cli", "deploy-to-vercel"],
    "marketing": ["seo-audit", "copywriting", "social-content"],
    "productivity": ["obsidian-markdown", "lark-doc", "gws-gmail", "gws-drive"],
}


# ---------------------------------------------------------------------------
# Task-type → real skill suggestions
# ---------------------------------------------------------------------------

TASK_SKILL_MAP: dict[str, list[str]] = {
    "web_scraping": ["firecrawl", "browser-use", "firecrawl-crawl"],
    "web_search": ["firecrawl-search", "browser-use"],
    "frontend": ["frontend-design", "react:components", "shadcn"],
    "data_analysis": ["xlsx", "pdf", "doc-coauthoring"],
    "code_review": ["systematic-debugging", "requesting-code-review", "webapp-testing"],
    "testing": ["test-driven-development", "playwright-best-practices", "webapp-testing"],
    "documentation": ["writing-plans", "writing-skills", "doc-coauthoring"],
    "deployment": ["deploy-to-vercel", "github-actions-docs", "sentry-cli"],
    "backend": ["supabase", "neon-postgres", "convex-quickstart"],
    "agents": ["mcp-builder", "skill-creator", "self-improving-agent"],
}


# ---------------------------------------------------------------------------
# Domain templates for generating system prompts
# ---------------------------------------------------------------------------

AGENT_TEMPLATES: dict[str, str] = {
    "web-scraper": """You are a web data extraction specialist using browser-use and firecrawl.

Best practices:
- Use firecrawl for fast structured extraction
- Use browser-use when JavaScript rendering is required
- Respect robots.txt and rate limits
- Return clean, structured JSON

Workflow:
1. Determine if JS rendering is needed
2. Select firecrawl-scrape (static) or browser-use (dynamic)
3. Extract and validate the data
4. Return structured results
""",

    "frontend": """You are a frontend engineering specialist.

Best practices:
- Use shadcn/ui components for consistent design
- Follow React best practices for performance
- Ensure accessibility (WCAG 2.1)
- Mobile-first responsive design

Workflow:
1. Understand design requirements
2. Select appropriate components from shadcn
3. Implement with proper TypeScript types
4. Validate accessibility and responsiveness
""",

    "code-reviewer": """You are a code quality specialist with systematic debugging skills.

Best practices:
- Follow TDD principles: test before fixing
- Root-cause analysis before patching
- Check security vulnerabilities (OWASP top 10)
- Verify error handling completeness

Workflow:
1. Read and understand the code context
2. Identify issues systematically
3. Propose minimal, targeted fixes
4. Write regression tests for found bugs
""",

    "agent-builder": """You are an AI agent design specialist.

Best practices:
- Use the Agent Skills Open Standard (skills.sh)
- Choose the minimal set of skills needed
- Write clear, step-by-step system prompts
- Select the right model tier for the task

Workflow:
1. Understand the task requirements
2. Identify required skills from skills.sh
3. Design the system prompt with clear workflow
4. Validate against the task's success criteria
""",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_skills_for_domain(domain: str) -> list[str]:
    """Get recommended skills for a domain."""
    return SKILL_CATEGORIES.get(domain, [])


def get_skills_for_task_type(task_type: str) -> list[str]:
    """Get required skills for a task type."""
    return TASK_SKILL_MAP.get(task_type, [])


def get_template_for_domain(domain: str) -> str | None:
    """Get system prompt template for a domain."""
    template_map = {
        "web": "web-scraper",
        "frontend": "frontend",
        "code": "code-reviewer",
        "agents": "agent-builder",
    }
    key = template_map.get(domain)
    return AGENT_TEMPLATES.get(key) if key else None


def get_install_command(skill_id: str) -> str | None:
    """Return the `npx skills add` command for a skill."""
    repo = SKILL_INSTALL_MAP.get(skill_id)
    if repo:
        return f"npx skills add {repo}"
    return None


def get_curriculum_skills(phase: int) -> dict[str, list[str]]:
    """Get skill recommendations for each curriculum phase.

    Args:
        phase: 1 (1 skill), 2 (2-3 skills), 3 (3-5 skills), 4 (5+ skills)
    """
    if phase == 1:
        return {
            "web": ["firecrawl"],
            "frontend": ["frontend-design"],
            "code": ["systematic-debugging"],
            "documents": ["pdf"],
        }
    elif phase == 2:
        return {
            "web": ["firecrawl", "browser-use"],
            "frontend": ["frontend-design", "shadcn"],
            "code": ["systematic-debugging", "test-driven-development"],
            "documents": ["pdf", "xlsx"],
        }
    elif phase == 3:
        return {
            "web": ["firecrawl", "browser-use", "firecrawl-search"],
            "frontend": ["frontend-design", "shadcn", "react:components"],
            "code": ["systematic-debugging", "test-driven-development", "playwright-best-practices"],
            "backend": ["supabase", "neon-postgres", "convex-quickstart"],
        }
    else:
        return {
            "web": list(SKILL_CATEGORIES["web"]),
            "frontend": list(SKILL_CATEGORIES["frontend"]),
            "code": list(SKILL_CATEGORIES["code"]),
            "agents": list(SKILL_CATEGORIES["agents"]),
        }
