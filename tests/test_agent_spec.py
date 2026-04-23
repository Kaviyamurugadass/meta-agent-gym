"""Tests for AgentSpec model."""

import pytest

from models import AgentSpec, ModelType


def test_agent_spec_creation():
    """Test creating a basic agent spec."""
    spec = AgentSpec(
        name="test-agent",
        description="A test agent",
        skills=["web-scraping", "http-client"],
        model=ModelType.SONNET,
        system_prompt="You are a test agent.",
    )

    assert spec.name == "test-agent"
    assert spec.description == "A test agent"
    assert spec.skills == ["web-scraping", "http-client"]
    assert spec.model == ModelType.SONNET
    assert spec.system_prompt == "You are a test agent."
    assert spec.user_invocable is True


def test_agent_spec_to_markdown():
    """Test converting agent spec to AGENT.md format."""
    spec = AgentSpec(
        name="web-scraper",
        description="Scrapes web data",
        skills=["web-scraping", "html-parser"],
        model=ModelType.SONNET,
        system_prompt="You scrape websites.",
        memory="project",
    )

    markdown = spec.to_markdown()

    assert "---" in markdown
    assert "name: web-scraper" in markdown
    assert "description: Scrapes web data" in markdown
    assert "- web-scraping" in markdown
    assert "- html-parser" in markdown
    assert "model: sonnet" in markdown
    assert "memory: project" in markdown
    assert "You scrape websites." in markdown


def test_agent_spec_to_dict():
    """Test converting agent spec to dict."""
    spec = AgentSpec(
        name="test-agent",
        description="Test",
        skills=["code-reviewer"],
        model=ModelType.HAIKU,
        system_prompt="Review code.",
    )

    spec_dict = spec.to_dict()

    assert spec_dict["name"] == "test-agent"
    assert spec_dict["skills"] == ["code-reviewer"]
    assert spec_dict["model"] == "haiku"


def test_agent_spec_defaults():
    """Test default values for optional fields."""
    spec = AgentSpec(
        name="minimal",
        description="Minimal spec",
        system_prompt="Do work.",
    )

    assert spec.model == ModelType.SONNET  # Default
    assert spec.skills == []  # Default empty list
    assert spec.user_invocable is True  # Default
    assert spec.allowed_tools is None
    assert spec.memory is None
    assert spec.max_turns is None
