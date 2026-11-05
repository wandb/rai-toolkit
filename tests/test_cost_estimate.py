# SPDX-FileCopyrightText: 2026 Harsh Raj Singhania
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Offline coverage for assessment cost estimates."""

import json
from types import SimpleNamespace

from rai_toolkit.assessment import AssessmentResult
from rai_toolkit.evaluation.cost_estimate import (
    MODEL_PRICING,
    estimate_assessment_run_cost,
)


def _items(n=1, scores=1):
    return SimpleNamespace(
        items=[
            SimpleNamespace(scores={"judge": object()} if scores else {})
            for _ in range(n)
        ]
    )


def _result(cost_estimate):
    return AssessmentResult(
        model_name="offline-stub",
        preset="general",
        run_id="cost-test",
        started_at="2026-01-01T00:00:00+00:00",
        duration_seconds=0.0,
        overall_score=1.0,
        overall_passed=True,
        evaluation_overall_score=1.0,
        evaluation_overall_passed=True,
        score_breakdown={},
        verdict_rationale=[],
        frameworks=[],
        policy_violations=[],
        redteam_summary=None,
        evaluation_summary={},
        content_hash="cost-test",
        cost_estimate=cost_estimate,
    )


def test_empty_evaluation_has_no_estimate():
    assert estimate_assessment_run_cost(SimpleNamespace(items=[]), "general") is None
    assert estimate_assessment_run_cost(SimpleNamespace(items=None), "general") is None


def test_known_model_uses_repository_price_table():
    estimate = estimate_assessment_run_cost(
        _items(), "general", judge_model="gpt-4o-mini"
    )
    pricing = MODEL_PRICING["gpt-4o-mini"]
    expected = round(1 * (800 * pricing["prompt"] + 150 * pricing["completion"]), 4)
    assert estimate["status"] == "available"
    assert estimate["requested_judge_model"] == "gpt-4o-mini"
    assert estimate["judge_model_for_pricing"] == "gpt-4o-mini"
    assert estimate["estimated_usd_upper_bound"] == expected
    assert estimate["assumed_llm_calls_upper_bound"] == 1
    assert estimate["assumed_tokens_per_call"] == {"prompt": 800, "completion": 150}
    assert estimate["preset"] == "general"


def test_unknown_model_is_unavailable_without_substitute_price(monkeypatch):
    monkeypatch.delenv("RAI_JUDGE_MODEL", raising=False)
    estimate = estimate_assessment_run_cost(
        _items(n=2, scores=1), "general", judge_model="unknown-custom-model"
    )
    assert estimate == {
        "preset": "general",
        "requested_judge_model": "unknown-custom-model",
        "assumed_llm_calls_upper_bound": 2,
        "assumed_tokens_per_call": {"prompt": 800, "completion": 150},
        "status": "unavailable",
        "reason": "unknown_model_pricing",
        "judge_model_for_pricing": None,
        "estimated_usd_upper_bound": None,
        "note": (
            "Pricing is unavailable for requested judge model 'unknown-custom-model': "
            "it is not in the local price table. No substitute price was "
            "used, and this is not a $0 estimate."
        ),
    }
    assert estimate["estimated_usd_upper_bound"] != 0


def test_explicit_argument_precedes_environment(monkeypatch):
    monkeypatch.setenv("RAI_JUDGE_MODEL", "gpt-4o")
    estimate = estimate_assessment_run_cost(
        _items(), "general", judge_model="gpt-4o-mini"
    )
    assert estimate["requested_judge_model"] == "gpt-4o-mini"
    assert estimate["judge_model_for_pricing"] == "gpt-4o-mini"
    assert estimate["status"] == "available"


def test_environment_selects_model_when_argument_empty(monkeypatch):
    monkeypatch.setenv("RAI_JUDGE_MODEL", "gpt-4o")
    estimate = estimate_assessment_run_cost(_items(), "general", judge_model="  ")
    pricing = MODEL_PRICING["gpt-4o"]
    expected = round(800 * pricing["prompt"] + 150 * pricing["completion"], 4)
    assert estimate["requested_judge_model"] == "gpt-4o"
    assert estimate["estimated_usd_upper_bound"] == expected


def test_default_model_when_unset(monkeypatch):
    monkeypatch.delenv("RAI_JUDGE_MODEL", raising=False)
    estimate = estimate_assessment_run_cost(_items(), "general")
    assert estimate["requested_judge_model"] == "gpt-4o-mini"
    assert estimate["status"] == "available"


def test_json_preserves_unavailable_state():
    estimate = estimate_assessment_run_cost(
        _items(), "general", judge_model="unknown-custom-model"
    )
    encoded = json.dumps(_result(estimate).to_dict())
    restored = json.loads(encoded)["cost_estimate"]
    assert restored["status"] == "unavailable"
    assert restored["reason"] == "unknown_model_pricing"
    assert restored["requested_judge_model"] == "unknown-custom-model"
    assert restored["judge_model_for_pricing"] is None
    assert restored["estimated_usd_upper_bound"] is None


def test_summary_and_html_explain_unavailable_pricing():
    estimate = estimate_assessment_run_cost(
        _items(), "general", judge_model="unknown-custom-model"
    )
    result = _result(estimate)
    summary = result.format_summary()
    html = result.to_html()
    assert "pricing unavailable for requested model unknown-custom-model" in summary
    assert "unknown_model_pricing" in summary
    assert "$" not in summary.split("Cost estimate", 1)[1].split("\n", 1)[0]
    assert "pricing unavailable for requested model" in html
    assert "unknown-custom-model" in html
    assert "unknown_model_pricing" in html
    assert "~$" not in html
