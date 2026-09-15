# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import asyncio

import pytest

from rai_toolkit.scorers import ToolCallAccuracyScorer


def criteria(expected, forbidden=None):
    block = {"expected": expected}
    if forbidden is not None:
        block["forbidden"] = forbidden
    return {"tool_calls": block}


def test_exact_match_and_reordered_nested_keys():
    scorer = ToolCallAccuracyScorer()
    result = scorer.score(
        "quoted lookup_record() is not evidence",
        tool_calls=[{"id": "1", "name": "lookup_record", "arguments": {"b": {"x": 2, "y": [1, True]}, "a": 1.0}}],
        trace_complete=True,
        success_criteria=criteria([{"name": "lookup_record", "arguments": {"a": 1, "b": {"y": [1, True], "x": 2}}}]),
    )
    assert result.assessed and result.passed and result.score == 1
    assert result.details["matched_actual_ids"] == ["1"]


def test_missing_and_unexpected_calls_use_f1():
    result = ToolCallAccuracyScorer().score(
        "",
        tool_calls=[{"id": "1", "name": "other", "arguments": {}}],
        trace_complete=True,
        success_criteria=criteria([{"name": "lookup", "arguments": {}}]),
    )
    assert result.assessed and result.score == 0 and not result.passed
    assert result.details["argument_mismatches"] == []
    assert result.details["unexpected_actual_ids"] == ["1"]


def test_forbidden_calls_fail_even_at_zero_threshold():
    result = ToolCallAccuracyScorer(threshold=0).score(
        "",
        tool_calls=[{"id": "1", "name": "send_email", "arguments": {}}],
        trace_complete=True,
        success_criteria=criteria([], ["send_email"]),
    )
    assert result.assessed and result.score == 0 and not result.passed
    assert result.details["forbidden_actual_ids"] == ["1"]


def test_empty_expected_is_real_assessed_pass():
    result = ToolCallAccuracyScorer().score("", tool_calls=[], trace_complete=True, success_criteria=criteria([]))
    assert result.assessed and result.passed and result.score == 1


def test_missing_or_invalid_evidence_is_unassessed():
    scorer = ToolCallAccuracyScorer()
    assert not scorer.score("", success_criteria=criteria([])).assessed
    assert scorer.score("", success_criteria=criteria([])).details["skipped"] == "missing_tool_trace"
    assert not scorer.score("", tool_calls=[], trace_complete=False, success_criteria=criteria([])).assessed
    assert not scorer.score("", tool_calls=[{"id": "1", "name": "x", "arguments": "{}"}], trace_complete=True, success_criteria=criteria([])).assessed


def test_async_and_batch_delegate_to_same_contract():
    scorer = ToolCallAccuracyScorer()
    kwargs = {"output": "", "tool_calls": [], "trace_complete": True, "success_criteria": criteria([])}
    assert asyncio.run(scorer.score_async(**kwargs)).passed
    assert scorer.score_batch([kwargs])[0].passed


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True])
def test_threshold_is_strictly_validated(threshold):
    with pytest.raises(ValueError):
        ToolCallAccuracyScorer(threshold=threshold)
