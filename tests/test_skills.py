"""Tests for skills registry."""

import pytest

from server import skills


def test_available_skills_not_empty():
    """Test that available skills are defined (skills.sh registry)."""
    assert len(skills.AVAILABLE_SKILLS) > 0
    assert "firecrawl" in skills.AVAILABLE_SKILLS
    assert "xlsx" in skills.AVAILABLE_SKILLS
    assert "systematic-debugging" in skills.AVAILABLE_SKILLS


def test_skill_categories():
    """Test skill categories are defined."""
    assert "web" in skills.SKILL_CATEGORIES
    assert "documents" in skills.SKILL_CATEGORIES
    assert "code" in skills.SKILL_CATEGORIES

    # Check web category has expected skills.sh entries
    assert "firecrawl" in skills.SKILL_CATEGORIES["web"]
    assert "browser-use" in skills.SKILL_CATEGORIES["web"]


def test_task_skill_map():
    """Test task to skill mapping."""
    assert "web_scraping" in skills.TASK_SKILL_MAP
    assert "data_analysis" in skills.TASK_SKILL_MAP
    assert "code_review" in skills.TASK_SKILL_MAP

    web_skills = skills.TASK_SKILL_MAP["web_scraping"]
    assert "firecrawl" in web_skills


def test_agent_templates():
    """Test agent templates are defined."""
    assert "web-scraper" in skills.AGENT_TEMPLATES
    assert "frontend" in skills.AGENT_TEMPLATES
    assert "code-reviewer" in skills.AGENT_TEMPLATES
    assert "agent-builder" in skills.AGENT_TEMPLATES

    # Template is non-empty
    assert len(skills.AGENT_TEMPLATES["web-scraper"]) > 0


def test_get_skills_for_domain():
    """Test getting skills for a domain."""
    web_skills = skills.get_skills_for_domain("web")
    assert isinstance(web_skills, list)
    assert len(web_skills) > 0
    assert "firecrawl" in web_skills


def test_get_skills_for_domain_unknown():
    """Test getting skills for unknown domain returns empty list."""
    unknown_skills = skills.get_skills_for_domain("unknown")
    assert unknown_skills == []


def test_get_skills_for_task_type():
    """Test getting skills for a task type."""
    ws_skills = skills.get_skills_for_task_type("web_scraping")
    assert isinstance(ws_skills, list)
    assert "firecrawl" in ws_skills


def test_get_template_for_domain():
    """Test getting template for a domain."""
    template = skills.get_template_for_domain("web")
    assert template is not None
    assert "scraping" in template.lower() or "specialist" in template.lower()


def test_get_template_for_domain_unknown():
    """Test getting template for unknown domain returns None."""
    template = skills.get_template_for_domain("unknown")
    assert template is None


def test_get_curriculum_skills_phase_1():
    """Test curriculum phase 1 (single skill)."""
    phase_skills = skills.get_curriculum_skills(1)

    assert "web" in phase_skills
    assert "documents" in phase_skills
    assert "code" in phase_skills

    # Phase 1 should have single skill tasks
    web_skills = phase_skills["web"]
    assert len(web_skills) == 1
    assert web_skills == ["firecrawl"]


def test_get_curriculum_skills_phase_2():
    """Test curriculum phase 2 (2-3 skills)."""
    phase_skills = skills.get_curriculum_skills(2)

    web_skills = phase_skills["web"]
    assert len(web_skills) >= 2
    assert "firecrawl" in web_skills
    assert "browser-use" in web_skills


def test_get_curriculum_skills_phase_4():
    """Test curriculum phase 4 (5+ skills)."""
    phase_skills = skills.get_curriculum_skills(4)

    web_skills = phase_skills["web"]
    # Phase 4 has at least 4 web skills (all web category skills plus json-parser)
    assert len(web_skills) >= 4
