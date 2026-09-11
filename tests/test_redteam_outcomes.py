# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Tests for assessed, resisted, and unassessed red-team outcomes."""

from __future__ import annotations

from rai_toolkit.redteam import AttackOutcome
from rai_toolkit.redteam.attacks import AttackCategory
from rai_toolkit.redteam.runner import AttackResult, RedTeamReport, _aggregate


def _result(
    attack_id: str,
    *,
    succeeded: bool,
    error: str | None = None,
    category: AttackCategory = AttackCategory.JAILBREAK,
) -> AttackResult:
    return AttackResult(
        attack_id=attack_id,
        category=category,
        succeeded=succeeded,
        model_output="model output",
        prompt="attack prompt",
        severity=4,
        error=error,
    )


def _report(results: list[AttackResult]) -> RedTeamReport:
    return RedTeamReport(
        model_name="test-model",
        results=results,
        by_family=_aggregate(results),
        total_duration_s=0.25,
        generated_at=1.0,
    )


def test_attack_outcome_values_are_stable() -> None:
    assert AttackOutcome.SUCCEEDED.value == "succeeded"
    assert AttackOutcome.RESISTED.value == "resisted"
    assert AttackOutcome.UNASSESSED_ERROR.value == "unassessed_error"


def test_mixed_results_use_only_assessed_attacks_for_rates() -> None:
    report = _report(
        [
            _result("succeeded", succeeded=True),
            _result("resisted", succeeded=False),
            _result("errored", succeeded=False, error="provider unavailable"),
        ]
    )

    assert report.total == 3
    assert report.total_assessed == 2
    assert report.total_errors == 1
    assert report.total_successes == 1
    assert report.error_rate == 1 / 3
    assert report.overall_success_rate == 0.5
    assert report.overall_resistance_rate == 0.5

    family = report.by_family[AttackCategory.JAILBREAK]
    assert family.total == 3
    assert family.assessed == 2
    assert family.errors == 1
    assert family.successes == 1
    assert family.success_rate == 0.5
    assert family.resistance_rate == 0.5


def test_all_errors_produce_no_assessment_rates() -> None:
    report = _report(
        [
            _result("error-with-message", succeeded=False, error="timeout"),
            _result("empty-error-message", succeeded=False, error=""),
            _result("another-error", succeeded=False, error="connection reset"),
        ]
    )

    assert report.total == 3
    assert report.total_assessed == 0
    assert report.total_errors == 3
    assert report.total_successes == 0
    assert report.error_rate == 1.0
    assert report.overall_success_rate is None
    assert report.overall_resistance_rate is None

    family = report.by_family[AttackCategory.JAILBREAK]
    assert family.assessed == 0
    assert family.success_rate is None
    assert family.resistance_rate is None


def test_error_takes_precedence_over_inconsistent_succeeded_flag() -> None:
    result = _result("inconsistent", succeeded=True, error="request failed")
    report = _report([result])

    assert result.assessed is False
    assert result.outcome is AttackOutcome.UNASSESSED_ERROR
    assert report.total_successes == 0
    assert report.total_assessed == 0
    assert report.by_family[AttackCategory.JAILBREAK].successes == 0


def test_assessed_result_outcomes_distinguish_success_and_resistance() -> None:
    succeeded = _result("succeeded", succeeded=True)
    resisted = _result("resisted", succeeded=False)

    assert succeeded.assessed is True
    assert succeeded.outcome is AttackOutcome.SUCCEEDED
    assert resisted.assessed is True
    assert resisted.outcome is AttackOutcome.RESISTED


def test_empty_report_has_zero_error_rate_but_no_assessment_rates() -> None:
    report = _report([])

    assert report.total_errors == 0
    assert report.total_assessed == 0
    assert report.error_rate == 0.0
    assert report.overall_success_rate is None
    assert report.overall_resistance_rate is None
    assert "Execution errors:       n/a (no attacks run)" in report.format_summary()
    assert "Execution errors:       0/0 (0.0%)" not in report.format_summary()


def test_serialization_adds_outcomes_and_assessment_counts() -> None:
    report = _report(
        [
            _result("succeeded", succeeded=True),
            _result("resisted", succeeded=False),
            _result("errored", succeeded=True, error="transport error"),
        ]
    )

    data = report.to_dict()

    assert data["total"] == 3
    assert data["total_successes"] == 1
    assert data["total_errors"] == 1
    assert data["total_assessed"] == 2
    assert data["error_rate"] == 1 / 3
    assert data["overall_success_rate"] == 0.5
    assert data["overall_resistance_rate"] == 0.5

    family = data["by_family"]["jailbreak"]
    assert family["total"] == 3
    assert family["assessed"] == 2
    assert family["successes"] == 1
    assert family["errors"] == 1
    assert family["success_rate"] == 0.5
    assert family["resistance_rate"] == 0.5

    assert [row["outcome"] for row in data["results"]] == [
        "succeeded",
        "resisted",
        "unassessed_error",
    ]


def test_summary_uses_assessed_denominators() -> None:
    report = _report(
        [
            _result("succeeded", succeeded=True),
            _result("resisted", succeeded=False),
            _result("errored", succeeded=False, error="timeout"),
        ]
    )

    summary = report.format_summary()

    assert "Attacks assessed:       2/3" in summary
    assert "Execution errors:       1/3 (33.3%)" in summary
    assert "Attack success rate:    50.0% (1/2 assessed)" in summary
    assert "1/2 assessed attacks succeeded (50%); 1/3 errors" in summary


def test_summary_renders_unassessed_run_as_not_available() -> None:
    report = _report(
        [
            _result("first", succeeded=False, error="timeout"),
            _result("second", succeeded=False, error=""),
        ]
    )

    summary = report.format_summary()

    assert "Attack success rate:    n/a (no attacks assessed)" in summary
    assert "Model resistance rate:  n/a (no attacks assessed)" in summary
    assert "jailbreak             n/a (no attacks assessed; 2/2 errors)" in summary
    assert "Model resistance rate:  100.0%" not in summary
