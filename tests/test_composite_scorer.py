# SPDX-FileCopyrightText: 2026 Yusef Syed
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

from __future__ import annotations

import asyncio
import threading
from contextvars import ContextVar
from typing import Any

import pytest

from rai_toolkit.scorers.base import BaseScorer, ScorerResult
from rai_toolkit.scorers.composite import CompositeScorer


class FakeScorer(BaseScorer):
    def __init__(self, result: ScorerResult, name: str) -> None:
        super().__init__(name=name, category=result.category)
        self.result = result
        self.sync_calls = 0

    def score(
        self, output: str, input: str = "", context: str = "", **kwargs: Any
    ) -> ScorerResult:
        self.sync_calls += 1
        return self.result


class AsyncOnlyScorer(FakeScorer):
    def __init__(self, result: ScorerResult, name: str) -> None:
        super().__init__(result, name)
        self.async_calls = 0

    def score(
        self, output: str, input: str = "", context: str = "", **kwargs: Any
    ) -> ScorerResult:
        raise AssertionError("CompositeScorer.score_async() must not call score()")

    async def score_async(
        self, output: str, input: str = "", context: str = "", **kwargs: Any
    ) -> ScorerResult:
        self.async_calls += 1
        return self.result


def _result(
    score: float,
    category: str,
    *,
    passed: bool = True,
    assessed: bool = True,
) -> ScorerResult:
    return ScorerResult(
        score=score,
        passed=passed,
        category=category,
        explanation=category,
        assessed=assessed,
    )


@pytest.mark.asyncio
async def test_score_async_uses_child_async_override_and_renormalizes_weights() -> None:
    assessed = AsyncOnlyScorer(_result(0.8, "assessed"), "async")
    unassessed = FakeScorer(
        _result(0.0, "unassessed", passed=False, assessed=False), "unassessed"
    )
    composite = CompositeScorer(
        [assessed, unassessed], weights={"assessed": 2.0, "unassessed": 98.0}
    )

    result = await composite.score_async("output")

    assert assessed.async_calls == 1
    assert result.score == 0.8
    assert result.passed
    assert result.assessed
    assert result.details["scorers_assessed"] == 1
    assert result.details["scorers_unassessed"] == 1


def test_score_excludes_unassessed_results_from_unweighted_aggregation_and_fail_fast() -> None:
    passing = FakeScorer(_result(0.75, "passing"), "passing")
    unassessed_failure = FakeScorer(
        _result(0.0, "not-applicable", passed=False, assessed=False), "unassessed"
    )
    composite = CompositeScorer([passing, unassessed_failure], fail_fast=True)

    result = composite.score("output")

    assert passing.sync_calls == 1
    assert unassessed_failure.sync_calls == 1
    assert result.score == 0.75
    assert result.passed
    assert result.assessed
    assert "All 1 assessed checks passed" == result.explanation


def test_fail_fast_still_applies_to_assessed_failures() -> None:
    passing = FakeScorer(_result(1.0, "passing"), "passing")
    failing = FakeScorer(_result(0.0, "failing", passed=False), "failing")
    composite = CompositeScorer([passing, failing], fail_fast=True)

    result = composite.score("output")

    assert not result.passed
    assert result.assessed
    assert result.details["scorers_passed"] == 1
    assert result.details["scorers_failed"] == 1


@pytest.mark.asyncio
async def test_all_unassessed_results_are_explicitly_unassessed_with_reason() -> None:
    first = FakeScorer(_result(0.2, "first", assessed=False), "first")
    second = FakeScorer(_result(0.9, "second", assessed=False), "second")
    composite = CompositeScorer([first, second])

    result = await composite.score_async("output")

    assert not result.assessed
    assert not result.passed
    assert result.score == 0.0
    assert result.explanation == "Un-assessed: no child scorers produced a signal."


def test_sync_all_unassessed_results_preserve_coverage_details() -> None:
    first = FakeScorer(_result(0.2, "first", assessed=False), "first")
    second = FakeScorer(_result(0.9, "second", assessed=False), "second")
    composite = CompositeScorer([first, second])

    result = composite.score("output")

    assert not result.assessed
    assert not result.passed
    assert result.details["scorers_run"] == 2
    assert result.details["scorers_assessed"] == 0
    assert result.details["scorers_unassessed"] == 2
    assert result.details["scorers_passed"] == 0
    assert result.details["scorers_failed"] == 0
    assert all(not child["assessed"] for child in result.details["individual_results"])


@pytest.mark.asyncio
async def test_async_children_preserve_order_context_and_execution_thread() -> None:
    caller_thread = threading.get_ident()
    trace = ContextVar("trace", default="missing")
    calls: list[tuple[str, int, str]] = []

    class SyncChild(FakeScorer):
        def score(self, **kwargs: Any) -> ScorerResult:
            calls.append((self.name, threading.get_ident(), trace.get()))
            return super().score(**kwargs)

    class AsyncChild(AsyncOnlyScorer):
        async def score_async(self, **kwargs: Any) -> ScorerResult:
            await asyncio.sleep(0)
            calls.append((self.name, threading.get_ident(), trace.get()))
            return await super().score_async(**kwargs)

    scorers = [
        SyncChild(_result(1.0, "first"), "first"),
        AsyncChild(_result(1.0, "second"), "second"),
        SyncChild(_result(1.0, "third"), "third"),
    ]
    token = trace.set("row-context")
    try:
        result = await CompositeScorer(scorers).score_async("output")
    finally:
        trace.reset(token)

    assert [name for name, _, _ in calls] == ["first", "second", "third"]
    assert [value for _, _, value in calls] == ["row-context"] * 3
    assert calls[0][1] != caller_thread
    assert calls[1][1] == caller_thread
    assert calls[2][1] != caller_thread
    assert result.passed


@pytest.mark.asyncio
async def test_async_sync_child_error_stops_later_children() -> None:
    class FailingChild(FakeScorer):
        def score(self, **kwargs: Any) -> ScorerResult:
            raise ValueError("child failed")

    failing = FailingChild(_result(1.0, "first"), "first")
    later = FakeScorer(_result(1.0, "second"), "second")
    with pytest.raises(ValueError, match="child failed"):
        await CompositeScorer([failing, later]).score_async("output")
    assert later.sync_calls == 0
