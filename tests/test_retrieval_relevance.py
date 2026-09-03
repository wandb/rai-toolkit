# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import json
import re
from unittest.mock import Mock

from rai_toolkit.prompts.judge_prompts import JUDGE_PROMPTS
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


def test_delimiter_only_context_is_unassessed_without_calling_judge() -> None:
    scorer = _scorer({"score": 3})

    result = scorer.score("A response", input="What is the revenue?", context=" --- --- ")

    assert not result.assessed
    assert result.details["skipped"] == "empty_context"
    scorer._call_judge.assert_not_called()


def test_blank_query_is_unassessed_without_calling_judge() -> None:
    scorer = _scorer({"score": 3})

    result = scorer.score("A response", input="   ", context="Chunk A\n---\nChunk B")

    assert not result.assessed
    assert result.details["skipped"] == "empty_query"
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
    scorer = _scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "Directly relevant."},
                {"chunk_index": 1, "relevance": "relevant", "reason": "Supports the query."},
            ],
        }
    )

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
                {"chunk_index": 1, "relevance": "irrelevant", "reason": "Unrelated to the query."},
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
    assert result.details["total_chunks"] == 2


def test_success_details_omit_raw_judge_response() -> None:
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "All relevant.",
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "On topic."},
            ],
        }
    )

    result = scorer.score("Answer", input="Question", context="Chunk A")

    assert result.assessed
    assert "judge_response" not in result.details


def test_total_chunks_come_from_the_context_not_the_judge() -> None:
    # A judge that invents extra verdicts (here: chunk_index 7) must not be
    # able to inflate the reported chunk counts.
    scorer = _scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "On topic."},
                {"chunk_index": 7, "relevance": "relevant", "reason": "Invented chunk."},
            ],
        }
    )

    result = scorer.score("Answer", input="Question", context="Chunk A\n---\nChunk B")

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["total_chunks"] == 2
    assert result.details["missing_chunk_indexes"] == [1]
    assert result.details["discarded_verdicts"] == 1
    assert "judge_response" in result.details


def test_missing_chunk_verdicts_are_a_parse_failure() -> None:
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

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["total_chunks"] == 1
    assert result.details["covered_chunks"] == 0
    assert result.details["missing_chunk_indexes"] == [0]
    assert "judge_response" in result.details


def test_partial_verdict_coverage_is_a_parse_failure() -> None:
    scorer = _scorer(
        {
            "score": 2,
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "On topic."},
                {"chunk_index": 1, "relevance": "irrelevant", "reason": "Off topic."},
            ],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="Revenue was $10M --- Weather forecast --- Tax rate info",
    )

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["missing_chunk_indexes"] == [2]
    assert result.details["discarded_verdicts"] == 0


def test_duplicate_chunk_indexes_keep_the_first_verdict() -> None:
    scorer = _scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "First verdict."},
                {"chunk_index": 0, "relevance": "irrelevant", "reason": "Duplicate."},
                {"chunk_index": 1, "relevance": "relevant", "reason": "On topic."},
            ],
        }
    )

    result = scorer.score("Answer", input="Question", context="Chunk A\n---\nChunk B")

    assert result.assessed
    assert result.details["relevant_chunks"] == 2  # the first verdict for chunk 0 wins
    assert result.details["total_chunks"] == 2
    assert result.details["discarded_verdicts"] == 1
    assert [v["chunk_index"] for v in result.details["chunk_verdicts"]] == [0, 1]


def test_unknown_relevance_label_is_a_parse_failure() -> None:
    scorer = _scorer(
        {
            "score": 2,
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "kinda relevant", "reason": "Off-scale label."},
            ],
        }
    )

    result = scorer.score("Answer", input="Question", context="Chunk A")

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["discarded_verdicts"] == 1
    assert result.details["missing_chunk_indexes"] == [0]


def test_non_dict_verdicts_are_discarded() -> None:
    scorer = _scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                "relevant",
                42,
                {"chunk_index": 0, "relevance": "relevant", "reason": "On topic."},
            ],
        }
    )

    result = scorer.score("Answer", input="Question", context="Chunk A")

    assert result.assessed
    assert result.details["relevant_chunks"] == 1
    assert result.details["discarded_verdicts"] == 2


def test_unparseable_overall_score_is_a_parse_failure() -> None:
    for bad_score in ("NaN", "Infinity", 5, -1, 3.5, "high", None):
        scorer = _scorer(
            {
                "score": bad_score,
                "chunk_verdicts": [
                    {"chunk_index": 0, "relevance": "relevant", "reason": "On topic."},
                ],
            }
        )

        result = scorer.score("Answer", input="Question", context="Chunk A")

        assert not result.assessed, f"score={bad_score!r} should be a parse failure"
        assert result.details["skipped"] == "judge_parse_failure"


def test_template_json_example_is_parseable() -> None:
    # The template goes through a single .format() call, so any literal brace
    # in the JSON example must be doubled exactly once. A doubled-twice brace
    # renders as "{{" and the judge is shown a JSON example it cannot imitate.
    template = JUDGE_PROMPTS["RetrievalRelevanceScorer"]["template"]
    rendered = template.format(output="o", input="i", context="c")
    block = rendered.split("Respond in JSON format:", 1)[1]

    # Placeholders are not valid JSON; fill them in the way a judge would,
    # and drop the "..." continuation entry (with its trailing comma).
    json_like = (
        block.replace("<0-3>", "2")
        .replace("<brief reasoning about overall retrieval quality>", "ok")
        .replace("relevant|partially_relevant|irrelevant", "relevant")
        .replace("<one short sentence>", "because")
        .replace("...", "")
    )
    json_like = re.sub(r",(\s*\])", r"\1", json_like)

    payload = json.loads(json_like)
    assert payload["score"] == 2
    assert payload["chunk_verdicts"][0]["chunk_index"] == 0
    assert payload["chunk_verdicts"][0]["relevance"] == "relevant"
