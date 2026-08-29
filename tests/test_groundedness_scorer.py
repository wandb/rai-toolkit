# SPDX-FileCopyrightText: 2026 Aditya Datta
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

from unittest.mock import Mock

from rai_toolkit.scorers import GroundednessScorer


def _scorer(result: dict[str, object]) -> GroundednessScorer:
    scorer = GroundednessScorer(api_key="test")
    scorer._call_judge = Mock(return_value=result)
    return scorer


def test_missing_context_is_unassessed_without_calling_judge() -> None:
    scorer = _scorer({"score": 3})

    result = scorer.score("A response", context="  ")

    assert not result.assessed
    assert result.details["skipped"] == "empty_context"
    scorer._call_judge.assert_not_called()


def test_supported_response_carries_verified_spans() -> None:
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "The claim is directly supported.",
            "supporting_spans": [
                {
                    "response_span": "Revenue grew 12%.",
                    "context_span": "Revenue grew by 12% year over year.",
                }
            ],
            "contradicting_spans": [],
        }
    )

    result = scorer.score(
        "Revenue grew 12%.", context="Revenue grew by 12% year over year."
    )

    assert result.assessed
    assert result.score == 1.0
    assert result.passed
    assert result.details["supporting_spans"] == [
        {
            "response_span": "Revenue grew 12%.",
            "context_span": "Revenue grew by 12% year over year.",
        }
    ]


def test_fabricated_evidence_spans_are_discarded() -> None:
    scorer = _scorer(
        {
            "score": 1,
            "explanation": "The response contradicts the context.",
            "supporting_spans": [
                {"response_span": "invented", "context_span": "also invented"}
            ],
            "contradicting_spans": [
                {
                    "response_span": "The warranty is five years.",
                    "context_span": "The warranty lasts two years.",
                }
            ],
        }
    )

    result = scorer.score(
        "The warranty is five years.", context="The warranty lasts two years."
    )

    assert result.score == 1 / 3
    assert not result.passed
    assert result.details["supporting_spans"] == []
    assert result.details["contradicting_spans"] == [
        {
            "response_span": "The warranty is five years.",
            "context_span": "The warranty lasts two years.",
        }
    ]
    assert result.details["discarded_evidence_spans"] == 1
