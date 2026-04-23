"""Observation quality guardrails — fails the build if domain observations are too weak.

The template ships with `placeholder_*` scenarios whose observations are
intentionally metadata-only. These tests **skip** for any scenario whose
task_id starts with `placeholder_` and **activate** the moment you rename
to a domain-specific task_id on finale day.

What "strong" means here:
  1. `Observation.latest_output` is populated (not None, not empty dict)
  2. `Observation.summary` has domain context beyond "Step N/M"
  3. Observation does NOT leak ground-truth values from `expected_findings`
  4. Changing the action between two steps produces a visible observation delta

If any check fails after you rename your scenarios, revisit
`server/environment.py::_build_observation` and add real decision-relevant
signal to `obs.latest_output`.
"""

from __future__ import annotations

import json
import re

import pytest

from models import Action, ActionCommand
from server.environment import Environment
from server.tasks.scenarios import SCENARIOS


def _is_placeholder(task_id: str) -> bool:
    """Placeholder scenarios skip these checks — they're template filler."""
    return task_id.startswith("placeholder_")


# ---------------------------------------------------------------------------
# Check 1: latest_output carries domain signal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.task_id)
def test_observation_latest_output_has_domain_signal(scenario):
    """`latest_output` must be populated with domain-relevant fields.

    Weak: None, empty dict, or dict with only "status": "ok" style fluff.
    Strong: metrics, log summaries, state snapshots, alerts, etc.
    """
    if _is_placeholder(scenario.task_id):
        pytest.skip(f"{scenario.task_id} is a placeholder — rename on finale day to activate this check")

    env = Environment(domain_randomise=False, seed=42)
    env.reset(scenario_name=scenario.task_id)
    # Take two steps so _build_observation has had a chance to populate state
    obs = env.step(Action(command=ActionCommand.INSPECT, args={"target": "any"}))
    obs = env.step(Action(command=ActionCommand.NOOP))

    assert obs.latest_output is not None, (
        f"\n{scenario.task_id}: latest_output is None after step. "
        f"\nFix: override _build_observation in environment.py and set "
        f"obs.latest_output to a dict containing at least one domain-signal "
        f"field (metrics, log excerpt, current state snapshot, etc.)."
    )
    assert len(obs.latest_output) >= 1, (
        f"\n{scenario.task_id}: latest_output is an empty dict. "
        f"\nAdd domain-specific fields (metrics, state snapshot, log excerpt, etc.)."
    )


# ---------------------------------------------------------------------------
# Check 2: summary has more than "Step N/M"
# ---------------------------------------------------------------------------


_MINIMAL_SUMMARY_RE = re.compile(r"^\s*Step\s+\d+/\d+\s*$")


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.task_id)
def test_observation_summary_has_domain_context(scenario):
    """`summary` must include domain context, not just 'Step N/M'."""
    if _is_placeholder(scenario.task_id):
        pytest.skip(f"{scenario.task_id} is a placeholder")

    env = Environment(domain_randomise=False, seed=42)
    env.reset(scenario_name=scenario.task_id)
    obs = env.step(Action(command=ActionCommand.INSPECT, args={"target": "any"}))

    assert not _MINIMAL_SUMMARY_RE.match(obs.summary), (
        f"\n{scenario.task_id}: summary is only 'Step N/M' (metadata). "
        f"\nAdd domain context — e.g. 'range=[1,16] last=lower' for guessing, "
        f"'cpu=85% latency=300ms alerts=2' for SRE."
    )


# ---------------------------------------------------------------------------
# Check 3: no ground-truth leakage from expected_findings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.task_id)
def test_observation_does_not_leak_ground_truth(scenario):
    """Values from `expected_findings` must NOT appear verbatim in observations.

    If they do, the agent can read the answer directly — env becomes trivial.
    """
    if _is_placeholder(scenario.task_id):
        pytest.skip(f"{scenario.task_id} is a placeholder")

    env = Environment(domain_randomise=False, seed=42)
    obs_reset = env.reset(scenario_name=scenario.task_id)
    obs_step = env.step(Action(command=ActionCommand.NOOP))

    combined = (
        json.dumps(obs_reset.model_dump(), default=str).lower()
        + " "
        + json.dumps(obs_step.model_dump(), default=str).lower()
    )

    for key, value in scenario.expected_findings.items():
        # Skip keys starting with _ (internal, e.g. _target, _initial_range in examples)
        if key.startswith("_"):
            continue
        if not isinstance(value, (str, int, float)):
            continue
        v_str = str(value).lower().strip()
        # Only flag if the value is specific enough to be a leak.
        # Digits-only strings ≤3 chars are too generic (could match step counts, etc.)
        if len(v_str) < 4 and v_str.isdigit():
            continue
        # Only flag strings that are >=4 chars (avoid false positives on tiny values)
        if len(v_str) < 4:
            continue

        assert v_str not in combined, (
            f"\n{scenario.task_id}: expected_findings['{key}'] value '{value}' "
            f"appears verbatim in an observation — this leaks ground truth. "
            f"\nMove it to State.hidden_truth instead (POMDP discipline)."
        )


# ---------------------------------------------------------------------------
# Check 4: different actions produce different observations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.task_id)
def test_different_actions_produce_visible_delta(scenario):
    """After two DIFFERENT actions, observations must differ somewhere.

    If they don't, the agent can't distinguish the effect of its actions —
    training signal is zero.
    """
    if _is_placeholder(scenario.task_id):
        pytest.skip(f"{scenario.task_id} is a placeholder")

    # Run two episodes with the SAME scenario but DIFFERENT first actions
    env_a = Environment(domain_randomise=False, seed=7)
    env_a.reset(scenario_name=scenario.task_id)
    obs_a = env_a.step(Action(command=ActionCommand.INSPECT, args={"target": "x"}))

    env_b = Environment(domain_randomise=False, seed=7)
    env_b.reset(scenario_name=scenario.task_id)
    obs_b = env_b.step(Action(command=ActionCommand.NOOP))

    # Observations should differ in at least one field beyond `step`
    a_signal = {"summary": obs_a.summary, "latest_output": obs_a.latest_output}
    b_signal = {"summary": obs_b.summary, "latest_output": obs_b.latest_output}

    assert a_signal != b_signal, (
        f"\n{scenario.task_id}: INSPECT vs NOOP produce identical observations. "
        f"\nThe agent cannot distinguish its actions. Either:"
        f"\n  (a) _execute_action does not actually change state (Step 5a)"
        f"\n  (b) _build_observation does not expose the state change (Step 5b)"
    )


# ---------------------------------------------------------------------------
# Sanity: at least one placeholder exists so template-as-shipped passes
# ---------------------------------------------------------------------------


def test_template_ships_with_placeholder_scenarios():
    """Template contract: at least one `placeholder_*` scenario exists in the box.

    This prevents accidentally shipping a template where ALL scenarios have
    been renamed — which would make the observation-quality tests fire in the
    template repo itself.
    """
    placeholders = [s for s in SCENARIOS if _is_placeholder(s.task_id)]
    assert placeholders, (
        "No placeholder scenarios found in server/tasks/scenarios.py. "
        "If this is a domain fill, that's expected — but be aware that "
        "deleting all placeholders means observation-quality tests will "
        "fail the build until domain observations are strong."
    )


# ---------------------------------------------------------------------------
# Report: show how many scenarios are being enforced
# ---------------------------------------------------------------------------


def test_report_scenario_enforcement_status():
    """Informational — prints which scenarios are enforced vs skipped.

    This is a regular test that always passes; its purpose is to make the
    status visible in test output. Run `pytest -s` to see.
    """
    enforced = [s.task_id for s in SCENARIOS if not _is_placeholder(s.task_id)]
    skipped = [s.task_id for s in SCENARIOS if _is_placeholder(s.task_id)]
    print(
        f"\n[observation quality] "
        f"{len(enforced)} scenario(s) enforced, {len(skipped)} placeholder(s) skipped"
    )
    if enforced:
        print(f"  enforced: {', '.join(enforced)}")
    if skipped:
        print(f"  skipped:  {', '.join(skipped)}")
    # Always passes — informational only
    assert True
