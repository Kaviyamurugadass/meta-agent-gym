"""Tests for skills registry."""

import pytest

from server import skills


def test_available_skills_not_empty():
    """Test that available skills are defined."""
    assert len(skills.AVAILABLE_SKILLS) > 0
    assert "web-scraping" in skills.AVAILABLE_SKILLS
    assert "csv-handler" in skills.AVAILABLE_SKILLS
    assert "code-reviewer" in skills.AVAILABLE_SKILLS


def test_skill_categories():
    """Test skill categories are defined."""
    assert "web" in skills.SKILL_CATEGORIES
    assert "data" in skills.SKILL_CATEGORIES
    assert "code" in skills.SKILL_CATEGORIES

    # Check web category has expected skills
    assert "web-scraping" in skills.SKILL_CATEGORIES["web"]
    assert "http-client" in skills.SKILL_CATEGORIES["web"]


def test_task_skill_map():
    """Test task to skill mapping."""
    assert "web_scraping" in skills.TASK_SKILL_MAP
    assert "data_analysis" in skills.TASK_SKILL_MAP
    assert "code_review" in skills.TASK_SKILL_MAP

    web_skills = skills.TASK_SKILL_MAP["web_scraping"]
    assert "web-scraping" in web_skills


def test_agent_templates():
    """Test agent templates are defined."""
    assert "web-scraper" in skills.AGENT_TEMPLATES
    assert "data-analyst" in skills.AGENT_TEMPLATES
    assert "code-reviewer" in skills.AGENT_TEMPLATES

    # Template is non-empty
    assert len(skills.AGENT_TEMPLATES["web-scraper"]) > 0


def test_get_skills_for_domain():
    """Test getting skills for a domain."""
    web_skills = skills.get_skills_for_domain("web")
    assert isinstance(web_skills, list)
    assert len(web_skills) > 0
    assert "web-scraping" in web_skills


def test_get_skills_for_domain_unknown():
    """Test getting skills for unknown domain returns empty list."""
    unknown_skills = skills.get_skills_for_domain("unknown")
    assert unknown_skills == []


def test_get_skills_for_task_type():
    """Test getting skills for a task type."""
    ws_skills = skills.get_skills_for_task_type("web_scraping")
    assert isinstance(ws_skills, list)
    assert "web-scraping" in ws_skills


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
    assert "data" in phase_skills
    assert "code" in phase_skills

    # Phase 1 should have single skill tasks
    web_skills = phase_skills["web"]
    assert len(web_skills) == 1
    assert web_skills == ["web-scraping"]


def test_get_curriculum_skills_phase_2():
    """Test curriculum phase 2 (2-3 skills)."""
    phase_skills = skills.get_curriculum_skills(2)

    web_skills = phase_skills["web"]
    assert len(web_skills) >= 2
    assert "web-scraping" in web_skills
    assert "html-parser" in web_skills


def test_get_curriculum_skills_phase_4():
    """Test curriculum phase 4 (5+ skills)."""
    phase_skills = skills.get_curriculum_skills(4)

    web_skills = phase_skills["web"]
    # Phase 4 has at least 4 web skills (all web category skills plus json-parser)
    assert len(web_skills) >= 4
