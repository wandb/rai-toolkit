# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Regression coverage for explicitly requested red-team sources."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from rai_toolkit.assessment.assessor import Assessor
from rai_toolkit.compliance.frameworks import ComplianceProfile, Framework
from rai_toolkit.evaluation.pipeline import EvaluationResults
from rai_toolkit.models.base import BaseModel, ModelResponse
from rai_toolkit.redteam.attacks import AttackCategory
from rai_toolkit.redteam.runner import (
    AttackResult,
    AttackRunner,
    RedTeamReport,
    _aggregate,
)


class StubModel(BaseModel):
    name = "stub-model"

    async def predict(
        self,
        input_text: str,
        context: str = "",
        **kwargs: Any,
    ) -> ModelResponse:
        return ModelResponse(output="safe response")


def _attack(
    attack_id: str,
    *,
    succeeded: bool = False,
    error: str | None = None,
) -> AttackResult:
    return AttackResult(
        attack_id=attack_id,
        category=AttackCategory.JAILBREAK,
        succeeded=succeeded,
        model_output="unsafe response" if succeeded else "safe response",
        prompt="attack prompt",
        severity=2,
        error=error,
    )


def _report(
    results: list[AttackResult],
    *,
    duration: float = 0.1,
) -> RedTeamReport:
    return RedTeamReport(
        model_name="stub-model",
        results=results,
        by_family=_aggregate(results),
        total_duration_s=duration,
        generated_at=1.0,
    )


def _passing_evaluation() -> EvaluationResults:
    profile = ComplianceProfile(
        name="test profile",
        framework=Framework.MIT_AI_RISK,
        categories=[],
        industry="general",
    )
    return EvaluationResults(
        name="passing evaluation",
        profile=profile,
        model_name="stub-model",
        items=[],
        summary={},
        overall_score=1.0,
        overall_passed=True,
        timestamp="2026-01-01T00:00:00+00:00",
    )


def _assessor(
    *,
    sources: list[str],
    run_redteam: bool = True,
    max_error_rate: float = 0.10,
) -> Assessor:
    return Assessor(
        model=StubModel(),
        preset="general",
        datasets=["test-dataset"],
        run_redteam=run_redteam,
        extra_redteam_sources=sources,
        redteam_max_error_rate=max_error_rate,
    )


def _stub_assessment_phases(
    monkeypatch: pytest.MonkeyPatch,
    assessor: Assessor,
) -> None:
    monkeypatch.setattr(assessor, "_load_datasets", list)

    async def passing_evaluation(*args: Any, **kwargs: Any) -> EvaluationResults:
        return _passing_evaluation()

    monkeypatch.setattr(assessor, "_run_evaluation", passing_evaluation)
    monkeypatch.setattr(assessor, "_run_policy_checks", lambda evaluation: ([], []))
    monkeypatch.setattr(assessor, "_assess_frameworks", lambda *args: [])
    monkeypatch.setattr(
        "rai_toolkit.assessment.assessor._warn_if_missing_llm_keys",
        lambda *args, **kwargs: None,
    )


def _stub_builtin_report(
    monkeypatch: pytest.MonkeyPatch,
    report: RedTeamReport,
) -> None:
    async def run_all(_runner: AttackRunner) -> RedTeamReport:
        return report

    monkeypatch.setattr(AttackRunner, "run_all", run_all)


def test_source_names_are_normalized_deduplicated_and_dispatched_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assessor = _assessor(
        sources=[" PyRiT ", "pyrit", " GARAK ", "garak", "   ", ""],
    )
    _stub_builtin_report(monkeypatch, _report([_attack("built-in")], duration=0.2))
    calls: list[str] = []

    async def run_source(source: str) -> RedTeamReport:
        calls.append(source)
        return _report([_attack(f"{source}-attack")], duration=0.3)

    monkeypatch.setattr(assessor, "_run_extra_redteam_source", run_source)

    report, failures = asyncio.run(assessor._run_redteam())

    assert assessor.extra_redteam_sources == ["pyrit", "garak", "<empty>"]
    assert calls == ["pyrit", "garak", "<empty>"]
    assert failures == []
    assert [result.attack_id for result in report.results] == [
        "built-in",
        "pyrit-attack",
        "garak-attack",
        "<empty>-attack",
    ]
    assert report.total_duration_s == pytest.approx(1.1)


def test_arbitrary_source_exception_is_sanitized_without_synthetic_attack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assessor = _assessor(sources=["pyrit"])
    _stub_builtin_report(monkeypatch, _report([_attack("built-in")]))

    async def fail_source(source: str) -> RedTeamReport:
        raise RuntimeError("secret-token-and-private-path")

    monkeypatch.setattr(assessor, "_run_extra_redteam_source", fail_source)

    report, failures = asyncio.run(assessor._run_redteam())

    assert [result.attack_id for result in report.results] == ["built-in"]
    assert report.total == 1
    assert len(failures) == 1
    assert failures[0].source == "pyrit"
    assert failures[0].error == (
        "RuntimeError: source execution failed. Check local logs for details."
    )
    assert "secret-token" not in failures[0].error


@pytest.mark.parametrize(
    ("source", "returned_report", "expected_error"),
    [
        ("pyrit", None, "The source returned no report."),
        ("garak", _report([]), "The source returned a report with no attacks."),
        ("custom-adapter", {}, "The source returned an invalid report."),
    ],
)
def test_invalid_source_reports_are_visible_failures(
    source: str,
    returned_report: Any,
    expected_error: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assessor = _assessor(sources=[source])
    _stub_builtin_report(monkeypatch, _report([_attack("built-in")]))

    async def run_source(_source: str) -> Any:
        return returned_report

    monkeypatch.setattr(assessor, "_run_extra_redteam_source", run_source)

    report, failures = asyncio.run(assessor._run_redteam())

    assert report.total == 1
    assert [(failure.source, failure.error) for failure in failures] == [
        (source, expected_error)
    ]


def test_mixed_source_results_preserve_successful_rows_and_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assessor = _assessor(sources=["pyrit", "garak"])
    _stub_builtin_report(monkeypatch, _report([_attack("built-in")], duration=0.2))

    async def run_source(source: str) -> RedTeamReport:
        if source == "garak":
            raise TimeoutError("internal endpoint detail")
        return _report(
            [_attack("pyrit-resisted"), _attack("pyrit-succeeded", succeeded=True)],
            duration=0.7,
        )

    monkeypatch.setattr(assessor, "_run_extra_redteam_source", run_source)

    report, failures = asyncio.run(assessor._run_redteam())

    assert [result.attack_id for result in report.results] == [
        "built-in",
        "pyrit-resisted",
        "pyrit-succeeded",
    ]
    assert report.total_duration_s == pytest.approx(0.9)
    assert len(failures) == 1
    assert failures[0].source == "garak"
    assert "internal endpoint detail" not in failures[0].error


def test_unknown_source_becomes_a_controlled_source_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assessor = _assessor(sources=["custom-adapter"])
    _stub_builtin_report(monkeypatch, _report([_attack("built-in")]))

    report, failures = asyncio.run(assessor._run_redteam())

    assert report.total == 1
    assert len(failures) == 1
    assert failures[0].source == "custom-adapter"
    assert failures[0].error == (
        "Unknown red-team source 'custom-adapter'. Expected 'pyrit' or 'garak'."
    )


def test_missing_optional_adapters_produce_actionable_source_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from integrations.garak_integration import adapter as garak_adapter
    from integrations.pyrit_integration import adapter as pyrit_adapter

    assessor = _assessor(sources=["pyrit", "garak"])
    _stub_builtin_report(monkeypatch, _report([_attack("built-in")]))
    monkeypatch.setattr(pyrit_adapter, "PYRIT_INSTALLED", False)
    monkeypatch.setattr(
        pyrit_adapter,
        "_PYRIT_IMPORT_ERROR",
        ModuleNotFoundError("private-import-location"),
    )
    monkeypatch.setattr(garak_adapter, "GARAK_INSTALLED", False)

    report, failures = asyncio.run(assessor._run_redteam())

    assert report.total == 1
    assert [failure.source for failure in failures] == ["pyrit", "garak"]
    assert "rai-toolkit[pyrit]" in failures[0].error
    assert "private-import-location" not in failures[0].error
    assert "rai-toolkit[garak]" in failures[1].error


@pytest.mark.parametrize("source", ["pyrit", "garak"])
def test_assessor_sanitizes_before_source_level_tracing_wrapper(
    source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if source == "pyrit":
        from integrations.pyrit_integration import adapter

        installed_name = "PYRIT_INSTALLED"
        runner_name = "run_pyrit_attacks"
    else:
        from integrations.garak_integration import adapter

        installed_name = "GARAK_INSTALLED"
        runner_name = "run_garak_probes"

    assessor = _assessor(sources=[source])
    _stub_builtin_report(monkeypatch, _report([_attack("built-in")]))
    calls: list[str] = []

    async def raw_runner(model: BaseModel) -> RedTeamReport:
        calls.append("raw")
        raise RuntimeError("private-source-exception-sentinel")

    async def traced_wrapper(model: BaseModel) -> RedTeamReport:
        calls.append("traced")
        raise AssertionError("source-level tracing wrapper should be bypassed")

    traced_wrapper.__wrapped__ = raw_runner  # type: ignore[attr-defined]
    monkeypatch.setattr(adapter, installed_name, True)
    monkeypatch.setattr(adapter, runner_name, traced_wrapper)

    report, failures = asyncio.run(assessor._run_redteam())

    assert report.total == 1
    assert calls == ["raw"]
    assert len(failures) == 1
    assert failures[0].source == source
    assert failures[0].error == (
        "RuntimeError: source execution failed. Check local logs for details."
    )
    assert "private-source-exception-sentinel" not in failures[0].error


def test_nonempty_all_error_source_counts_as_covered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assessor = _assessor(sources=["pyrit"])
    _stub_builtin_report(monkeypatch, _report([]))

    async def run_source(source: str) -> RedTeamReport:
        return _report([_attack("pyrit-error", error="timeout")])

    monkeypatch.setattr(assessor, "_run_extra_redteam_source", run_source)

    report, failures = asyncio.run(assessor._run_redteam())

    assert failures == []
    assert report.total == 1
    assert report.total_assessed == 0
    assert report.total_errors == 1


def test_full_assessment_fails_closed_and_sanitizes_source_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assessor = _assessor(sources=["pyrit"])
    _stub_assessment_phases(monkeypatch, assessor)
    _stub_builtin_report(monkeypatch, _report([_attack("built-in-resisted")]))

    async def fail_source(source: str) -> RedTeamReport:
        raise ConnectionError("https://secret.example/api?token=private")

    monkeypatch.setattr(assessor, "_run_extra_redteam_source", fail_source)

    result = asyncio.run(assessor.run())
    serialized = result.to_dict()
    serialized_text = json.dumps(serialized)

    assert result.overall_passed is False
    assert result.redteam_source_coverage_passed is False
    assert result.redteam_source_failures == [
        {
            "source": "pyrit",
            "error": (
                "ConnectionError: source execution failed. "
                "Check local logs for details."
            ),
        }
    ]
    assert result.redteam_summary is not None
    assert result.redteam_summary["total"] == 1
    assert result.redteam_summary["overall_resistance_rate"] == 1.0
    assert result.score_breakdown["red_team_resistance"] == 1.0
    assert result.score_breakdown["red_team_composite_component"] == 0.0
    assert result.overall_score == pytest.approx(0.8)
    assert "secret.example" not in serialized_text
    assert "token=private" not in serialized_text
    assert any(
        "Red-team source coverage gate failed" in line
        for line in result.verdict_rationale
    )


def test_full_assessment_passes_source_gate_when_every_source_returns_attacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assessor = _assessor(sources=["pyrit", "garak"])
    _stub_assessment_phases(monkeypatch, assessor)
    _stub_builtin_report(monkeypatch, _report([_attack("built-in")]))

    async def run_source(source: str) -> RedTeamReport:
        return _report([_attack(f"{source}-resisted")])

    monkeypatch.setattr(assessor, "_run_extra_redteam_source", run_source)

    result = asyncio.run(assessor.run())

    assert result.overall_passed is True
    assert result.redteam_source_coverage_passed is True
    assert result.redteam_source_failures == []
    assert result.redteam_summary is not None
    assert result.redteam_summary["total"] == 3
    assert result.score_breakdown["red_team_resistance"] == 1.0
    assert result.score_breakdown["red_team_composite_component"] == 1.0
    assert result.overall_score == pytest.approx(1.0)
    assert any(
        "including requested source coverage" in line
        for line in result.verdict_rationale
    )


def test_all_error_source_passes_coverage_but_fails_attack_error_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assessor = _assessor(sources=["pyrit"], max_error_rate=1.0)
    _stub_assessment_phases(monkeypatch, assessor)
    _stub_builtin_report(monkeypatch, _report([]))

    async def run_source(source: str) -> RedTeamReport:
        return _report([_attack("pyrit-error", error="timeout")])

    monkeypatch.setattr(assessor, "_run_extra_redteam_source", run_source)

    result = asyncio.run(assessor.run())

    assert result.redteam_source_coverage_passed is True
    assert result.redteam_source_failures == []
    assert result.redteam_error_budget_passed is False
    assert result.overall_passed is False
    assert any(
        "all 1 attempted attack(s) ended in execution errors" in line
        for line in result.verdict_rationale
    )


def test_disabled_redteam_leaves_source_gate_not_applicable_and_does_not_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assessor = _assessor(sources=["pyrit"], run_redteam=False)
    _stub_assessment_phases(monkeypatch, assessor)

    async def unexpected_source_call(source: str) -> RedTeamReport:
        raise AssertionError(f"unexpected source dispatch: {source}")

    monkeypatch.setattr(
        assessor,
        "_run_extra_redteam_source",
        unexpected_source_call,
    )

    result = asyncio.run(assessor.run())

    assert result.overall_passed is True
    assert result.redteam_summary is None
    assert result.redteam_source_coverage_passed is None
    assert result.redteam_source_failures == []
    assert result.redteam_error_budget_passed is None
    assert result.redteam_severity_gate_passed is None
