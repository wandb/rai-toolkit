# SPDX-FileCopyrightText: 2026 Kunyang Zhang
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Offline tests for `OutputFormatScorer` structural checks."""

from __future__ import annotations

import pytest

from rai_toolkit.scorers import OutputFormatScorer


def test_valid_json_passes() -> None:
    result = OutputFormatScorer().score('{"answer": 42}')

    assert result.score == 1.0
    assert result.passed
    assert result.assessed
    assert result.explanation == "Valid JSON with all required keys"


def test_valid_json_missing_a_required_key_is_a_measured_failure() -> None:
    result = OutputFormatScorer(required_keys=["answer", "sources"]).score('{"answer": 42}')

    assert result.score == 0.5
    assert not result.passed
    assert result.assessed
    assert result.details["missing_keys"] == ["sources"]


def test_invalid_json_is_a_measured_failure() -> None:
    result = OutputFormatScorer().score("not json at all")

    assert result.score == 0.0
    assert not result.passed
    assert result.assessed
    assert "Invalid JSON" in result.explanation


def test_valid_xml_passes() -> None:
    result = OutputFormatScorer(expected_format="xml").score("<answer><value>42</value></answer>")

    assert result.score == 1.0
    assert result.passed
    assert result.assessed


def test_invalid_xml_is_a_measured_failure() -> None:
    result = OutputFormatScorer(expected_format="xml").score("<answer>")

    assert result.score == 0.0
    assert not result.passed
    assert result.assessed
    assert "Invalid XML" in result.explanation


@pytest.mark.parametrize("expected_format", ["yaml", "JSON", "markdown", ""])
def test_unsupported_format_is_unassessed(expected_format: str) -> None:
    """A configuration mistake used to report a full pass for every output.

    `expected_format` is case-sensitive and only `json` and `xml` are implemented,
    so `expected_format="JSON"` is a typo that has to surface as a coverage gap.
    """
    result = OutputFormatScorer(expected_format=expected_format).score("whatever")

    assert not result.assessed
    assert result.score == 0.0
    assert not result.passed
    assert result.category == "MIT-7.1"
    assert result.details["skipped"] == "unsupported_format"
    assert result.details["scorer_name"] == "OutputFormatScorer"
    assert expected_format in result.explanation


def test_unsupported_format_never_claims_a_pass() -> None:
    scorer = OutputFormatScorer(expected_format="JSON")

    results = [scorer.score(text) for text in ['{"answer": 42}', "not json at all", ""]]

    assert all(not result.assessed and not result.passed for result in results)


def test_unsupported_format_is_reported_by_the_pipeline_as_a_coverage_gap() -> None:
    """`assessed=False` is what keeps the placeholder out of aggregates."""
    from rai_toolkit.scorers import ScoreNormalizer

    result = OutputFormatScorer(expected_format="yaml").score("whatever")

    assessed = [r for r in [result] if r.assessed]
    assert ScoreNormalizer.aggregate_scores(assessed) == 0.0
    assert result.details["skipped"]
