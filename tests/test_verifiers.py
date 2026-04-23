"""Tests for HardVerifiers module."""

import pytest

from server.verifiers import HardVerifiers, VerifierResult


def test_verify_yaml_valid():
    """Test YAML verification with valid input."""
    spec = {
        "name": "test",
        "description": "test",
        "system_prompt": "A" * 100,
    }

    result = HardVerifiers.verify_yaml(spec)

    assert result.passed is True
    assert result.score == 1.0
    assert result.errors == []


def test_verify_yaml_invalid():
    """Test YAML verification with invalid input (non-serializable)."""
    # Use a non-serializable object
    spec = {
        "name": "test",
        "callback": lambda x: x,  # Functions can't be YAML serialized
    }

    result = HardVerifiers.verify_yaml(spec)

    assert result.passed is False
    assert result.score == 0.0
    assert len(result.errors) > 0


def test_verify_required_fields_all_present():
    """Test required fields verification with all fields present."""
    spec = {
        "name": "test-agent",
        "description": "A test agent",
        "system_prompt": "You are a test agent with sufficient content.",
    }

    result = HardVerifiers.verify_required_fields(spec)

    assert result.passed is True
    assert result.score == 1.0
    assert result.errors == []


def test_verify_required_fields_missing_name():
    """Test required fields verification with missing name."""
    spec = {
        "description": "A test agent",
        "system_prompt": "You are a test agent.",
    }

    result = HardVerifiers.verify_required_fields(spec)

    assert result.passed is False
    assert result.score == 0.0
    assert "Missing required field: name" in result.errors


def test_verify_required_fields_empty():
    """Test required fields verification with empty fields."""
    spec = {
        "name": "",
        "description": "",
        "system_prompt": "",
    }

    result = HardVerifiers.verify_required_fields(spec)

    assert result.passed is False
    assert result.score == 0.0
    assert len(result.errors) == 3


def test_verify_prompt_length_valid():
    """Test prompt length verification with valid prompt."""
    spec = {"system_prompt": "A" * 100}

    result = HardVerifiers.verify_prompt_length(spec, min_length=50)

    assert result.passed is True
    assert result.score == 1.0
    assert result.errors == []


def test_verify_prompt_length_too_short():
    """Test prompt length verification with short prompt."""
    spec = {"system_prompt": "Short"}

    result = HardVerifiers.verify_prompt_length(spec, min_length=50)

    assert result.passed is False
    assert result.score == 0.0
    assert "Prompt too short" in result.errors[0]


def test_verify_model_type_valid():
    """Test model type verification with valid model."""
    spec = {"model": "sonnet"}

    result = HardVerifiers.verify_model_type(spec)

    assert result.passed is True
    assert result.score == 1.0


def test_verify_model_type_invalid():
    """Test model type verification with invalid model."""
    spec = {"model": "invalid-model"}

    result = HardVerifiers.verify_model_type(spec)

    assert result.passed is False
    assert result.score == 0.0
    assert "Invalid model" in result.errors[0]


def test_verify_skills_format_valid():
    """Test skills format verification with valid list."""
    spec = {"skills": ["web-scraping", "http-client"]}

    result = HardVerifiers.verify_skills_format(spec)

    assert result.passed is True
    assert result.score == 1.0


def test_verify_skills_format_invalid():
    """Test skills format verification with non-list."""
    spec = {"skills": "web-scraping"}

    result = HardVerifiers.verify_skills_format(spec)

    assert result.passed is False
    assert result.score == 0.0
    assert "must be a list" in result.errors[0]


def test_verify_skills_format_none():
    """Test skills format verification with None (optional field)."""
    spec = {}

    result = HardVerifiers.verify_skills_format(spec)

    assert result.passed is True
    assert result.score == 1.0


def test_verify_all():
    """Test running all verifiers."""
    spec = {
        "name": "test-agent",
        "description": "Test",
        "system_prompt": "A" * 100,
        "model": "sonnet",
        "skills": ["web-scraping"],
    }

    results = HardVerifiers.verify_all(spec)

    assert "yaml_valid" in results
    assert "has_required_fields" in results
    assert "prompt_length_ok" in results
    assert "model_valid" in results
    assert "skills_format_ok" in results

    # All should pass with valid spec
    for result in results.values():
        assert result.passed is True


def test_get_gate_results_all_pass():
    """Test gate results when all pass."""
    spec = {
        "name": "test-agent",
        "description": "Test",
        "system_prompt": "A" * 100,
    }

    passed, errors = HardVerifiers.get_gate_results(spec)

    assert passed is True
    assert errors == []


def test_get_gate_results_fail():
    """Test gate results when one fails."""
    spec = {
        "name": "test-agent",
        "description": "",  # Empty description
        "system_prompt": "Short",  # Too short
    }

    passed, errors = HardVerifiers.get_gate_results(spec)

    assert passed is False
    assert len(errors) > 0
