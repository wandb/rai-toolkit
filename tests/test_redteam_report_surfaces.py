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
    redteam_summary: dict[str, Any] | None,
    *,
    error_budget_passed: bool | None,
    source_coverage_passed: bool | None = None,
    source_failures: list[dict[str, str]] | None = None,
    verdict_rationale: list[str] | None = None,
) -> AssessmentResult:
    resistance = (
        redteam_summary.get("overall_resistance_rate")
        if redteam_summary is not None
        else None
    )
    redteam_component = (
        0.0 if source_coverage_passed is False else resistance or 0.0
    )
    return AssessmentResult(
        model_name="offline-stub",
        preset="general",
        run_id="surface-test",
        started_at="2026-01-01T00:00:00+00:00",
        duration_seconds=1.0,
        overall_score=0.8,
        overall_passed=(
            error_budget_passed is not False
            and source_coverage_passed is not False
        ),
        evaluation_overall_score=1.0,
        evaluation_overall_passed=True,
        score_breakdown={
            "evaluation_raw": 1.0,
            "red_team_resistance": resistance,
            "red_team_composite_component": redteam_component,
            "policy_health": 1.0,
            "blended_overall": 0.8,
        },
        verdict_rationale=verdict_rationale or [],
        frameworks=[],
        policy_violations=[],
        redteam_summary=redteam_summary,
        evaluation_summary={},
        content_hash="surface-test",
        redteam_severity_gate_threshold=4,
        redteam_severity_gate_passed=(
            True
            if redteam_summary is not None
            and int(redteam_summary.get("total_assessed") or 0) > 0
            else None
        ),
        redteam_error_budget=0.1,
        redteam_error_budget_passed=error_budget_passed,
        redteam_error_budget_failures=(
            [{"attack_id": "error-1", "error": "provider unavailable"}]
            if error_budget_passed is False
            else []
        ),
        redteam_source_coverage_passed=source_coverage_passed,
        redteam_source_failures=source_failures or [],
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


def _assessed_summary() -> dict[str, Any]:
    return {
        "total": 1,
        "total_successes": 0,
        "total_errors": 0,
        "total_assessed": 1,
        "error_rate": 0.0,
        "overall_success_rate": 0.0,
        "overall_resistance_rate": 1.0,
        "by_family": {},
        "results": [
            {
                "attack_id": "resisted",
                "category": "jailbreak",
                "succeeded": False,
                "severity": 2,
                "error": None,
            }
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
    assert next(g for g in view.gates if g.key == "source_coverage").state == "N/A"
    assert view.redteam_source_failures == []


def test_shared_view_marks_redteam_gates_not_applicable_when_not_run() -> None:
    result = _result(None, error_budget_passed=None)

    view = AssessmentReportView.from_result(result)

    assert next(g for g in view.gates if g.key == "severity").state == "N/A"
    assert next(g for g in view.gates if g.key == "error_budget").state == "N/A"
    assert next(g for g in view.gates if g.key == "source_coverage").state == "N/A"
    assert view.redteam_attacks_total == 0
    assert view.redteam_resistance is None
    assert view.scores[1].percent is None
    assert view.scores[1].state == "N/A"


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


def test_report_surfaces_render_skipped_redteam_as_not_applicable() -> None:
    result = _result(None, error_budget_passed=None)

    standalone = result.to_html()
    assert "red-team severity gate: sev ≥ 4: N/A" in standalone
    assert "red-team error budget: errors ≤ 10%: N/A" in standalone
    assert "red-team source coverage: N/A" in standalone
    assert '<div class="value">n/a</div>' in standalone
    assert '<div class="value">80.0%</div>' not in standalone

    pytest.importorskip("weave")
    from integrations.weave_integration.views import render_assessment_html

    weave_view = render_assessment_html(result)
    assert "red-team severity gate (sev ≥ 4) N/A" in weave_view
    assert "red-team error budget (errors ≤ 10%) N/A" in weave_view
    assert "red-team source coverage" not in weave_view
    assert '<div class="score-value">n/a</div>' in weave_view
    assert '<div class="score-value">80.0%</div>' not in weave_view


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


def test_shared_and_local_surfaces_report_source_coverage_failure() -> None:
    source = "<private-adapter>"
    error = "dependency missing: install the optional extra <pyrit>"
    result = _result(
        _assessed_summary(),
        error_budget_passed=True,
        source_coverage_passed=False,
        source_failures=[{"source": source, "error": error}],
    )

    view = AssessmentReportView.from_result(result)

    assert next(g for g in view.gates if g.key == "source_coverage").state == "FAIL"
    assert view.scores[1].state == "FAIL"
    assert view.scores[1].note.startswith("requested source coverage failed")
    assert len(view.redteam_source_failures) == 1
    assert view.redteam_source_failures[0].source == source
    assert view.redteam_source_failures[0].error == error

    terminal = result.format_summary()
    assert "Red-team source coverage:     [FAIL]" in terminal
    assert f"{source}: {error}" in terminal

    standalone = result.to_html()
    assert "red-team source coverage: FAIL" in standalone
    assert "Requested source failures (1)" in standalone
    assert "&lt;private-adapter&gt;" in standalone
    assert "dependency missing: install the optional extra &lt;pyrit&gt;" in standalone


def test_weave_surfaces_do_not_export_source_failure_metadata() -> None:
    pytest.importorskip("weave")
    from integrations.weave_integration.views import (
        _compact_result_view,
        render_assessment_html,
    )

    source = "private-adapter-sentinel"
    error = "private-source-error-sentinel"
    result = _result(
        _assessed_summary(),
        error_budget_passed=True,
        source_coverage_passed=False,
        source_failures=[{"source": source, "error": error}],
        verdict_rationale=[
            "Evaluation evidence remains available for review.",
            "Red-team source coverage gate failed: "
            f"{source} could not run because {error}."
        ],
    )

    rendered = render_assessment_html(result)
    rendered_lower = rendered.lower()
    compacted_text = repr(_compact_result_view(result))

    assert "source coverage" not in rendered_lower
    assert source not in rendered
    assert error not in rendered
    assert "redteam_source_coverage_passed" not in compacted_text
    assert "redteam_source_failures" not in compacted_text
    assert source not in compacted_text
    assert error not in compacted_text
    assert "Evaluation evidence remains available for review." in rendered
    assert "Evaluation evidence remains available for review." in compacted_text
    assert "A local-only evidence gate did not pass." in rendered
    assert "A local-only evidence gate did not pass." in compacted_text
    assert "VERDICT: FAIL" in rendered


def test_weave_compact_failure_returns_only_safe_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("weave")
    from integrations.weave_integration.views import _compact_result_view

    source = "private-adapter-sentinel"
    error = "private-source-error-sentinel"
    result = _result(
        _assessed_summary(),
        error_budget_passed=True,
        source_coverage_passed=False,
        source_failures=[{"source": source, "error": error}],
    )

    def fail_serialization() -> dict[str, Any]:
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(result, "to_dict", fail_serialization)

    compacted = _compact_result_view(result)
    compacted_text = repr(compacted)

    assert compacted == {
        "verdict": "FAIL",
        "evaluation_score": 1.0,
        "composite_score": 0.8,
        "result": None,
    }
    assert source not in compacted_text
    assert error not in compacted_text


def test_zero_attack_source_failure_does_not_report_clean_error_rate() -> None:
    empty_summary = {
        "total": 0,
        "total_successes": 0,
        "total_errors": 0,
        "total_assessed": 0,
        "error_rate": 0.0,
        "overall_success_rate": None,
        "overall_resistance_rate": None,
        "by_family": {},
        "results": [],
    }
    result = _result(
        empty_summary,
        error_budget_passed=None,
        source_coverage_passed=False,
        source_failures=[
            {"source": "pyrit", "error": "The source returned no report."}
        ],
    )

    standalone = result.to_html()

    assert "n/a (no attacks attempted)" in standalone
    assert ">0 (0.0%)<" not in standalone


def test_legacy_dict_recovers_counts_and_omits_new_gate() -> None:
    result = _result(_all_error_summary(), error_budget_passed=False).to_dict()
    for key in (
        "redteam_error_budget",
        "redteam_error_budget_passed",
        "redteam_error_budget_failures",
        "redteam_source_coverage_passed",
        "redteam_source_failures",
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
    assert all(gate.key != "source_coverage" for gate in view.gates)
    assert view.redteam_source_failures == []


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

    assert fake_wandb.summary["redteam_severity_gate"] == "N/A"
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
    result = _result(None, error_budget_passed=None)

    summarize_assessment_run(object(), result=result, submission_id="sub-test")

    assert fake_wandb.summary["redteam_severity_gate"] == "N/A"
    assert fake_wandb.summary["redteam_error_budget_gate"] == "N/A"
    assert "redteam_error_rate" not in fake_wandb.summary


def test_wandb_summary_does_not_export_source_failure_metadata(monkeypatch) -> None:
    pytest.importorskip("weave")
    from integrations.weave_integration.wandb_run import summarize_assessment_run

    logged: list[dict[str, Any]] = []
    fake_wandb = SimpleNamespace(summary={}, log=lambda values: logged.append(values))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    source = "private-adapter-sentinel"
    error = "private-source-error-sentinel"
    result = _result(
        _assessed_summary(),
        error_budget_passed=True,
        source_coverage_passed=False,
        source_failures=[{"source": source, "error": error}],
    )

    summarize_assessment_run(object(), result=result, submission_id="sub-test")

    exported = repr((fake_wandb.summary, logged))
    assert all("source" not in key for key in fake_wandb.summary)
    assert source not in exported
    assert error not in exported
