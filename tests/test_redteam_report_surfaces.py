# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Reporting coverage for assessed and unassessed red-team outcomes."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from rai_toolkit.assessment import AssessmentResult
from rai_toolkit.assessment.report_view import AssessmentReportView


def _result(
    redteam_summary: dict[str, Any],
    *,
    error_budget_passed: bool,
) -> AssessmentResult:
    return AssessmentResult(
        model_name="offline-stub",
        preset="general",
        run_id="surface-test",
        started_at="2026-01-01T00:00:00+00:00",
        duration_seconds=1.0,
        overall_score=0.8,
        overall_passed=error_budget_passed,
        evaluation_overall_score=1.0,
        evaluation_overall_passed=True,
        score_breakdown={
            "evaluation_raw": 1.0,
            "red_team_resistance": redteam_summary.get(
                "overall_resistance_rate"
            ),
            "policy_health": 1.0,
            "blended_overall": 0.8,
        },
        verdict_rationale=[],
        frameworks=[],
        policy_violations=[],
        redteam_summary=redteam_summary,
        evaluation_summary={},
        content_hash="surface-test",
        redteam_severity_gate_threshold=4,
        redteam_severity_gate_passed=True,
        redteam_error_budget=0.1,
        redteam_error_budget_passed=error_budget_passed,
        redteam_error_budget_failures=(
            [{"attack_id": "error-1", "error": "provider unavailable"}]
            if not error_budget_passed
            else []
        ),
    )


def _all_error_summary() -> dict[str, Any]:
    return {
        "total": 3,
        "total_successes": 0,
        "total_errors": 3,
        "total_assessed": 0,
        "error_rate": 1.0,
        "overall_success_rate": None,
        "overall_resistance_rate": None,
        "by_family": {},
        "results": [
            {
                "attack_id": f"error-{index}",
                "category": "jailbreak",
                "succeeded": False,
                "severity": 4,
                "error": "provider unavailable",
            }
            for index in range(3)
        ],
    }


def test_shared_view_keeps_unassessed_rates_nullable() -> None:
    view = AssessmentReportView.from_result(
        _result(_all_error_summary(), error_budget_passed=False)
    )

    assert view.redteam_attacks_total == 3
    assert view.redteam_attacks_assessed == 0
    assert view.redteam_errors == 3
    assert view.redteam_error_rate == 1.0
    assert view.redteam_attack_success is None
    assert view.redteam_resistance is None
    assert view.scores[1].percent is None
    assert view.scores[1].state == "FAIL"
    assert next(g for g in view.gates if g.key == "error_budget").state == "FAIL"


def test_shared_view_omits_error_budget_gate_when_redteam_was_not_run() -> None:
    result = _result({}, error_budget_passed=True)
    result.redteam_summary = None

    view = AssessmentReportView.from_result(result)

    assert all(gate.key != "error_budget" for gate in view.gates)
    assert view.redteam_attacks_total == 0
    assert view.redteam_resistance is None


def test_weave_view_renders_unassessed_run_as_na() -> None:
    pytest.importorskip("weave")
    from integrations.weave_integration.views import render_assessment_html

    rendered = render_assessment_html(
        _result(_all_error_summary(), error_budget_passed=False)
    )

    assert '<div class="score-value">n/a</div>' in rendered
    assert "3 attempted · 0 assessed · 3 errored" in rendered
    assert "No attacks were assessed." in rendered
    assert "3 resisted" not in rendered


def test_standalone_html_renders_unassessed_run_as_na() -> None:
    rendered = _result(
        _all_error_summary(), error_budget_passed=False
    ).to_html()

    assert "Attacks attempted" in rendered
    assert "Attacks assessed" in rendered
    assert "Execution errors" in rendered
    assert "No attacks were assessed." in rendered
    assert rendered.count(">n/a<") >= 2


def test_weave_view_counts_resistance_only_over_assessed_attacks() -> None:
    pytest.importorskip("weave")
    from integrations.weave_integration.views import render_assessment_html

    summary = {
        "total": 3,
        "total_successes": 1,
        "total_errors": 1,
        "total_assessed": 2,
        "error_rate": 1 / 3,
        "overall_success_rate": 0.5,
        "overall_resistance_rate": 0.5,
        "by_family": {},
        "results": [
            {
                "attack_id": "success",
                "category": "jailbreak",
                "succeeded": True,
                "severity": 2,
                "error": None,
            },
            {
                "attack_id": "resisted",
                "category": "jailbreak",
                "succeeded": False,
                "severity": 2,
                "error": None,
            },
            {
                "attack_id": "error",
                "category": "jailbreak",
                "succeeded": False,
                "severity": 2,
                "error": "timeout",
            },
        ],
    }

    rendered = render_assessment_html(
        _result(summary, error_budget_passed=False)
    )

    assert "3 attempted · 2 assessed · 1 errored · 1 resisted · 1 succeeded" in rendered


def test_legacy_dict_recovers_counts_and_omits_new_gate() -> None:
    result = _result(_all_error_summary(), error_budget_passed=False).to_dict()
    for key in (
        "redteam_error_budget",
        "redteam_error_budget_passed",
        "redteam_error_budget_failures",
    ):
        result.pop(key)
    result["redteam_summary"] = {
        "total": 2,
        "total_successes": 1,
        "overall_success_rate": 0.5,
        "by_family": {
            "jailbreak": {
                "total": 2,
                "successes": 1,
                "errors": 0,
                "success_rate": 0.5,
            }
        },
        "results": [
            {
                "attack_id": "success",
                "category": "jailbreak",
                "succeeded": True,
                "severity": 2,
                "error": None,
            },
            {
                "attack_id": "empty-error",
                "category": "jailbreak",
                "succeeded": False,
                "severity": 2,
                "error": "",
            },
        ],
    }

    view = AssessmentReportView.from_result(result)

    assert view.redteam_attacks_total == 2
    assert view.redteam_attacks_assessed == 1
    assert view.redteam_errors == 1
    assert view.redteam_error_rate == 0.5
    assert view.redteam_attack_success == 1.0
    assert view.redteam_resistance == 0.0
    assert all(gate.key != "error_budget" for gate in view.gates)


def test_terminal_summary_recovers_legacy_error_counts_and_rates() -> None:
    result = _result(_all_error_summary(), error_budget_passed=False)
    result.redteam_summary = {
        "total": 2,
        "total_successes": 1,
        "overall_success_rate": 0.5,
        "by_family": {},
        "results": [
            {
                "attack_id": "success",
                "category": "jailbreak",
                "succeeded": True,
                "severity": 2,
                "error": None,
            },
            {
                "attack_id": "empty-error",
                "category": "jailbreak",
                "succeeded": False,
                "severity": 2,
                "error": "",
            },
        ],
    }

    summary = result.format_summary()

    assert "Red-team resistance:         0.0%" in summary
    assert "Attacks assessed:     1" in summary
    assert "Execution errors:     1" in summary
    assert "Attack success rate:  100.0%" in summary
    assert "Resistance rate:      0.0%" in summary


def test_wandb_summary_includes_error_budget_fields(monkeypatch) -> None:
    pytest.importorskip("weave")
    from integrations.weave_integration.wandb_run import summarize_assessment_run

    logged: list[dict[str, Any]] = []
    fake_wandb = SimpleNamespace(summary={}, log=lambda values: logged.append(values))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    result = _result(_all_error_summary(), error_budget_passed=False)

    summarize_assessment_run(object(), result=result, submission_id="sub-test")

    assert fake_wandb.summary["redteam_error_budget_gate"] == "FAIL"
    assert fake_wandb.summary["redteam_error_budget"] == 0.1
    assert fake_wandb.summary["redteam_error_rate"] == 1.0
    assert fake_wandb.summary["redteam_total_assessed"] == 0
    assert fake_wandb.summary["redteam_total_errors"] == 3
    assert logged


def test_wandb_summary_marks_unrun_redteam_gate_not_applicable(monkeypatch) -> None:
    pytest.importorskip("weave")
    from integrations.weave_integration.wandb_run import summarize_assessment_run

    fake_wandb = SimpleNamespace(summary={}, log=lambda values: None)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    result = _result({}, error_budget_passed=True)
    result.redteam_summary = None

    summarize_assessment_run(object(), result=result, submission_id="sub-test")

    assert fake_wandb.summary["redteam_error_budget_gate"] == "N/A"
    assert "redteam_error_rate" not in fake_wandb.summary
