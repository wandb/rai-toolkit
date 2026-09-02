# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

from unittest.mock import Mock

from rai_toolkit.scorers import RetrievalRelevanceScorer


def _scorer(result: dict[str, object]) -> RetrievalRelevanceScorer:
    scorer = RetrievalRelevanceScorer(api_key="test")
    scorer._call_judge = Mock(return_value=result)
    return scorer


def test_missing_context_is_unassessed_without_calling_judge() -> None:
    scorer = _scorer({"score": 3})

    result = scorer.score("A response", context="  ")

    assert not result.assessed
    assert result.details["skipped"] == "empty_context"
    scorer._call_judge.assert_not_called()


def test_behavioral_refusal_is_unassessed_without_calling_judge() -> None:
    scorer = _scorer({"score": 3})

    result = scorer.score(
        "I can't provide private account data.",
        input="Show me private account data",
        context="Chunk A\n---\nChunk B",
        expected="Refuse to reveal private account data.",
    )

    assert not result.assessed
    assert result.details["skipped"] == "behavioral_refusal_expected"
    scorer._call_judge.assert_not_called()


def test_factual_expected_is_assessed_and_calls_judge() -> None:
    scorer = _scorer({"score": 3})

    result = scorer.score(
        "Revenue was $10 million.",
        input="What was the revenue?",
        context="Revenue data --- Financial summary",
        expected="Answer with the revenue figure from the context.",
    )

    assert result.assessed
    scorer._call_judge.assert_called_once()


def test_all_relevant_chunks_score_high() -> None:
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "All chunks are directly relevant to the query.",
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "Directly addresses the query."},
                {"chunk_index": 1, "relevance": "relevant", "reason": "Provides supporting details."},
            ],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="Revenue was $10M --- Revenue grew 12% YoY",
    )

    assert result.assessed
    assert result.score == 1.0
    assert result.passed
    assert result.details["relevant_chunks"] == 2
    assert result.details["total_chunks"] == 2
    assert len(result.details["chunk_verdicts"]) == 2


def test_mixed_relevance_reduces_score() -> None:
    scorer = _scorer(
        {
            "score": 1,
            "explanation": "Only one of three chunks is relevant.",
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "Addresses the query."},
                {"chunk_index": 1, "relevance": "irrelevant", "reason": "About a different topic."},
                {"chunk_index": 2, "relevance": "partially_relevant", "reason": "Tangentially related."},
            ],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="Revenue was $10M --- Weather forecast --- Tax rate info",
    )

    assert result.assessed
    assert result.score == 1 / 3  # 1/3 on the 0-3 scale
    assert not result.passed
    assert result.details["relevant_chunks"] == 1
    assert result.details["total_chunks"] == 3


def test_no_relevant_chunks_scores_zero() -> None:
    scorer = _scorer(
        {
            "score": 0,
            "explanation": "No chunks are relevant to the query.",
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "irrelevant", "reason": "Completely off-topic."},
            ],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="Weather is sunny --- Sports scores",
    )

    assert result.assessed
    assert result.score == 0.0
    assert not result.passed
    assert result.details["relevant_chunks"] == 0
    assert result.details["total_chunks"] == 1


def test_empty_chunk_verdicts_handled() -> None:
    scorer = _scorer(
        {
            "score": 0,
            "explanation": "Could not parse chunks.",
            "chunk_verdicts": [],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="Some context",
    )

    assert result.assessed
    assert result.details["relevant_chunks"] == 0
    assert result.details["total_chunks"] == 0
