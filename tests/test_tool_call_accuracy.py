# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import asyncio
import json
from copy import deepcopy

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
    assert result.details["matched_actual_indexes"] == [0]
    assert result.details["matched_calls"] == [
        {
            "expected_index": 0,
            "actual_index": 0,
            "actual_id": "1",
            "name": "lookup_record",
            "expected_arguments": {"a": 1, "b": {"y": [1, True], "x": 2}},
            "actual_arguments": {"b": {"x": 2, "y": [1, True]}, "a": 1.0},
        }
    ]


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


def test_argument_mismatch_preserves_original_evidence_without_mutating_inputs():
    tool_calls = [{"id": "1", "name": "lookup", "arguments": {"record_id": "actual"}}]
    success_criteria = criteria([{"name": "lookup", "arguments": {"record_id": "expected"}}])
    original_calls = deepcopy(tool_calls)
    original_criteria = deepcopy(success_criteria)

    result = ToolCallAccuracyScorer().score(
        "",
        tool_calls=tool_calls,
        trace_complete=True,
        success_criteria=success_criteria,
    )

    assert result.details["argument_mismatches"] == [
        {
            "expected_index": 0,
            "actual_index": 0,
            "actual_id": "1",
            "name": "lookup",
            "expected_arguments": {"record_id": "expected"},
            "actual_arguments": {"record_id": "actual"},
        }
    ]
    assert tool_calls == original_calls
    assert success_criteria == original_criteria
    json.dumps(result.details, allow_nan=False)


def test_missing_call_has_only_expected_evidence():
    result = ToolCallAccuracyScorer().score(
        "",
        tool_calls=[],
        trace_complete=True,
        success_criteria=criteria([{"name": "lookup", "arguments": {"id": 1}}]),
    )
    assert result.details["missing_expected_calls"] == [
        {"expected_index": 0, "name": "lookup", "arguments": {"id": 1}}
    ]
    assert "actual_id" not in result.details["missing_expected_calls"][0]


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


def test_missing_expected_key_is_missing_not_invalid_criteria():
    result = ToolCallAccuracyScorer().score(
        "",
        tool_calls=[],
        trace_complete=True,
        success_criteria={"tool_calls": {}},
    )
    assert not result.assessed
    assert result.details["skipped"] == "missing_tool_call_criteria"


@pytest.mark.parametrize(
    ("source", "expected_location"),
    [
        ("actual", "tool_calls[0].arguments.value.nested_key"),
        (
            "expected",
            "success_criteria.tool_calls.expected[0].arguments.value.nested_key",
        ),
    ],
)
def test_non_finite_arguments_report_safe_location_and_reason(source, expected_location):
    tool_calls = [{"id": "1", "name": "lookup", "arguments": {}}]
    expected = [{"name": "lookup", "arguments": {}}]
    target = tool_calls[0] if source == "actual" else expected[0]
    target["arguments"] = {"value": {"nested_key": float("nan")}}

    result = ToolCallAccuracyScorer().score(
        "",
        tool_calls=tool_calls,
        trace_complete=True,
        success_criteria=criteria(expected),
    )

    assert not result.assessed
    assert result.details["validation_location"] == expected_location
    assert result.details["validation_reason"] == "non_finite_number"
    assert "nan" not in json.dumps(result.details, allow_nan=False).lower()


def test_large_integer_is_valid_json_and_matches_exactly():
    value = 10**400
    result = ToolCallAccuracyScorer().score(
        "",
        tool_calls=[{"id": "1", "name": "lookup", "arguments": {"value": value}}],
        trace_complete=True,
        success_criteria=criteria([{"name": "lookup", "arguments": {"value": value}}]),
    )
    assert result.assessed and result.passed


def test_repeated_calls_match_one_to_one_and_pair_diagnostics_deterministically():
    result = ToolCallAccuracyScorer().score(
        "",
        tool_calls=[
            {"id": "first", "name": "lookup", "arguments": {"id": 1}},
            {"id": "second", "name": "lookup", "arguments": {"id": 3}},
        ],
        trace_complete=True,
        success_criteria=criteria(
            [
                {"name": "lookup", "arguments": {"id": 1}},
                {"name": "lookup", "arguments": {"id": 2}},
            ]
        ),
    )
    assert result.score == 0.5
    assert result.details["matched_actual_ids"] == ["first"]
    assert result.details["argument_mismatches"][0]["actual_id"] == "second"


def test_async_and_batch_delegate_to_same_contract():
    scorer = ToolCallAccuracyScorer()
    kwargs = {"output": "", "tool_calls": [], "trace_complete": True, "success_criteria": criteria([])}
    assert asyncio.run(scorer.score_async(**kwargs)).passed
    assert scorer.score_batch([kwargs])[0].passed


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True, 10**400])
def test_threshold_is_strictly_validated(threshold):
    with pytest.raises(ValueError):
        ToolCallAccuracyScorer(threshold=threshold)
