# SPDX-FileCopyrightText: 2026 Jah-yee
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Tests for unassessed items handling in ComplianceReport."""

from __future__ import annotations

import json as json_module
from dataclasses import dataclass
from typing import Any

import pytest

from rai_toolkit.evaluation.pipeline import EvaluationItem, EvaluationResults
from rai_toolkit.evaluation.report import ComplianceReport
from rai_toolkit.compliance.frameworks import Framework


@dataclass
class FakeProfile:
    framework: Framework = Framework.MIT_AI_RISK
    industry: str | None = "Test"


def _item(input_: str, scores: list[tuple[str, float, bool, bool]]) -> EvaluationItem:
    from rai_toolkit.scorers.base import ScorerResult
    return EvaluationItem(
        input=input_,
        model_output="output",
        scores={
            name: ScorerResult(
                score=score,
                passed=passed,
                category="MIT-1.1",
                explanation="test",
                assessed=assessed,
            )
            for name, score, passed, assessed in scores
        },
    )


def _results(items: list[EvaluationItem], summary: dict[str, Any]) -> EvaluationResults:
    return EvaluationResults(
        name="test-run",
        profile=FakeProfile(),
        model_name="test-model",
        items=items,
        summary=summary,
        overall_score=0.8,
        overall_passed=True,
        timestamp="2026-10-06T00:00:00Z",
        metadata={},
    )


class TestAllUnassessed:
    def test_summary_na_no_failure(self) -> None:
        # total_items=0: nothing was measured by the aggregator
        r = _results([_item("i1", [("s", 0.0, False, False)])],
                     {"MIT-1.1": {"pass_rate": None, "unassessed_items": 1, "passed_items": 0, "total_items": 0, "failed_items": 0}})
        s = ComplianceReport(r).to_summary()
        assert "MIT-1.1: N/A (1 un-assessed) [N/A]" in s
        assert "Items with failures" not in s

    def test_dict_failed_items_empty(self) -> None:
        # total_items=0: nothing was measured by the aggregator
        r = _results([_item("i1", [("s", 0.0, False, False)])],
                     {"MIT-1.1": {"pass_rate": None, "unassessed_items": 1, "passed_items": 0, "total_items": 0, "failed_items": 0}})
        d = ComplianceReport(r).to_dict()
        assert d["failed_items"] == []


class TestMixedPassUnassessed:
    def test_summary_shows_unassessed(self) -> None:
        # total_items=1: only the measured (assessed) item counts; unassessed is separate
        r = _results(
            [_item("i1", [("s", 1.0, True, True)]), _item("i2", [("s", 0.0, False, False)])],
            {"MIT-1.1": {"mean_score": 1.0, "pass_rate": 1.0, "passed_items": 1, "total_items": 1, "failed_items": 0, "unassessed_items": 1}})
        s = ComplianceReport(r).to_summary()
        assert "MIT-1.1: 100.0% (1/1 passed, 1 un-assessed) [PASS]" in s
        assert "Items with failures" not in s

    def test_dict_no_failures(self) -> None:
        # total_items=1: only the measured (assessed) item counts
        r = _results(
            [_item("i1", [("s", 1.0, True, True)]), _item("i2", [("s", 0.0, False, False)])],
            {"MIT-1.1": {"mean_score": 1.0, "pass_rate": 1.0, "passed_items": 1, "total_items": 1, "failed_items": 0, "unassessed_items": 1}})
        d = ComplianceReport(r).to_dict()
        assert d["failed_items"] == []


class TestFailurePlusUnassessed:
    def test_summary_counts_one_failure(self) -> None:
        # total_items=1: only the measured (assessed) item; i2 is unassessed and not counted in measured total
        r = _results(
            [_item("i1", [("s", 0.0, False, True)]), _item("i2", [("s", 0.0, False, False)])],
            {"MIT-1.1": {"mean_score": 0.0, "pass_rate": 0.0, "passed_items": 0, "total_items": 1, "failed_items": 1, "unassessed_items": 1}})
        s = ComplianceReport(r).to_summary()
        assert "MIT-1.1: 0.0% (0/1 passed, 1 un-assessed) [FAIL]" in s
        assert "Items with failures: 1/2" in s

    def test_dict_one_failure_row(self) -> None:
        # total_items=1: only the measured item; i2 (unassessed) is not in measured total
        r = _results(
            [_item("i1", [("s", 0.0, False, True)]), _item("i2", [("s", 0.0, False, False)])],
            {"MIT-1.1": {"mean_score": 0.0, "pass_rate": 0.0, "passed_items": 0, "total_items": 1, "failed_items": 1, "unassessed_items": 1}})
        d = ComplianceReport(r).to_dict()
        assert len(d["failed_items"]) == 1


class TestRowWithBothFailureAndUnassessed:
    def test_dict_nested_only_genuine_failure(self) -> None:
        # total_items=1: genuine is assessed (passed=False) so it counts as a failure; unass is not assessed
        r = _results(
            [_item("i1", [("genuine", 0.0, False, True), ("unass", 0.0, False, False)])],
            {"MIT-1.1": {"mean_score": 0.0, "pass_rate": 0.0, "passed_items": 0, "total_items": 1, "failed_items": 1, "unassessed_items": 1}})
        d = ComplianceReport(r).to_dict()
        assert len(d["failed_items"]) == 1  # genuine assessed failure counted once


class TestRowWithPassAndUnassessed:
    def test_not_counted_as_failure(self) -> None:
        # total_items=1: one assessed result (pass=passed), one unassessed (not counted in total)
        r = _results(
            [_item("i1", [("pass", 1.0, True, True), ("unass", 0.0, False, False)])],
            {"MIT-1.1": {"mean_score": 1.0, "pass_rate": 1.0, "passed_items": 1, "total_items": 1, "failed_items": 0, "unassessed_items": 1}})
        d = ComplianceReport(r).to_dict()
        assert d["failed_items"] == []


class TestFullyAssessedUnchanged:
    def test_passing_unchanged(self) -> None:
        r = _results([_item("i1", [("s", 1.0, True, True)])],
                     {"MIT-1.1": {"mean_score": 1.0, "pass_rate": 1.0, "passed_items": 1, "total_items": 1, "failed_items": 0, "unassessed_items": 0}})
        s = ComplianceReport(r).to_summary()
        assert "MIT-1.1: 100.0% (1/1 passed) [PASS]" in s
        assert "un-assessed" not in s
        assert "Items with failures" not in s

    def test_failing_unchanged(self) -> None:
        r = _results([_item("i1", [("s", 0.0, False, True)])],
                     {"MIT-1.1": {"mean_score": 0.0, "pass_rate": 0.0, "passed_items": 0, "total_items": 1, "failed_items": 1, "unassessed_items": 0}})
        s = ComplianceReport(r).to_summary()
        assert "MIT-1.1: 0.0% (0/1 passed) [FAIL]" in s
        assert "Items with failures: 1/1" in s
        assert "un-assessed" not in s


class TestTwoAssessedFailures:
    def test_two_assessed_failures_both_in_failed_scores(self) -> None:
        # Two different scorer failures on the same item: both should appear in failed_scores
        r = _results(
            [_item("i1", [("s1", 0.0, False, True), ("s2", 0.0, False, True)])],
            {"MIT-1.1": {"mean_score": 0.0, "pass_rate": 0.0, "passed_items": 0, "total_items": 1, "failed_items": 1, "unassessed_items": 0}})
        rep = ComplianceReport(r)
        d = rep.to_dict()
        assert len(d["failed_items"]) == 1  # counted once
        scores = d["failed_items"][0]["failed_scores"]
        assert "s1" in scores
        assert "s2" in scores
        # to_summary: row counted once, not twice
        s = rep.to_summary()
        assert "Items with failures: 1/1" in s

    def test_json_serializes_failed_items(self) -> None:
        # to_json should include failed_items with all required fields
        r = _results(
            [_item("i1", [("s", 0.0, False, True)])],
            {"MIT-1.1": {"mean_score": 0.0, "pass_rate": 0.0, "passed_items": 0, "total_items": 1, "failed_items": 1, "unassessed_items": 0}})
        rep = ComplianceReport(r)
        json_str = rep.to_json()
        data = json_module.loads(json_str)
        assert "failed_items" in data
        assert len(data["failed_items"]) == 1
        assert "failed_scores" in data["failed_items"][0]
        assert "s" in data["failed_items"][0]["failed_scores"]

    def test_input_preserved_after_to_summary(self) -> None:
        r = _results(
            [_item("i1", [("s", 0.0, False, True)])],
            {"MIT-1.1": {"mean_score": 0.0, "pass_rate": 0.0, "passed_items": 0, "total_items": 1, "failed_items": 1, "unassessed_items": 0}})
        rep = ComplianceReport(r)
        rep.to_summary()
        assert "i1" in rep.to_dict()["failed_items"][0]["input"]

    def test_input_preserved_after_to_dict(self) -> None:
        r = _results(
            [_item("i2", [("s", 0.0, False, True)])],
            {"MIT-1.1": {"mean_score": 0.0, "pass_rate": 0.0, "passed_items": 0, "total_items": 1, "failed_items": 1, "unassessed_items": 0}})
        rep = ComplianceReport(r)
        d = rep.to_dict()
        assert d["failed_items"][0]["input"] == "i2"

    def test_input_preserved_after_to_json(self) -> None:
        r = _results(
            [_item("i3", [("s", 0.0, False, True)])],
            {"MIT-1.1": {"mean_score": 0.0, "pass_rate": 0.0, "passed_items": 0, "total_items": 1, "failed_items": 1, "unassessed_items": 0}})
        rep = ComplianceReport(r)
        json_str = rep.to_json()
        data = json_module.loads(json_str)
        assert data["failed_items"][0]["input"] == "i3"


class TestFailureCountTextWithUnassessed:
    def test_failure_count_not_inflated_by_unassessed_row(self) -> None:
        # Row with pass + unassessed: should not appear in failure count
        r = _results(
            [_item("i1", [("pass", 1.0, True, True), ("unass", 0.0, False, False)])],
            {"MIT-1.1": {"mean_score": 1.0, "pass_rate": 1.0, "passed_items": 1, "total_items": 1, "failed_items": 0, "unassessed_items": 1}})
        s = ComplianceReport(r).to_summary()
        assert "MIT-1.1: 100.0% (1/1 passed, 1 un-assessed) [PASS]" in s
        assert "Items with failures" not in s

    def test_failure_count_not_inflated_by_unassessed_failure_row(self) -> None:
        # Row with failure + unassessed: only the assessed failure counts
        r = _results(
            [_item("i1", [("fail", 0.0, False, True), ("unass", 0.0, False, False)])],
            {"MIT-1.1": {"mean_score": 0.0, "pass_rate": 0.0, "passed_items": 0, "total_items": 1, "failed_items": 1, "unassessed_items": 1}})
        s = ComplianceReport(r).to_summary()
        assert "MIT-1.1: 0.0% (0/1 passed, 1 un-assessed) [FAIL]" in s
        # Total rows = 1, one assessed failure counted once
        assert "Items with failures: 1/1" in s
