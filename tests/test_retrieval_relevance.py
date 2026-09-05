# SPDX-FileCopyrightText: 2026 Zeming-Yuan
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


def test_expected_containing_declined_is_still_assessed() -> None:
    # "decline" is a behavioral-refusal marker for other judges, but this
    # scorer grades the retriever: factual expected text containing
    # "declined" must reach the judge, not be skipped.
    scorer = _scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "Has the figure."},
            ],
        }
    )

    result = scorer.score(
        "Revenue declined 5 percent year over year.",
        input="What happened to revenue?",
        context="[fin-q4] Revenue declined 5 percent, driven by renewals.",
        expected="Revenue declined 5 percent.",
    )

    assert result.assessed
    scorer._call_judge.assert_called_once()
    assert "skipped" not in result.details


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


def test_duplicate_chunk_indexes_are_a_parse_failure() -> None:
    # Two verdicts for the same chunk make the chunk ambiguous; "first wins"
    # would make the grade depend on the judge's verdict order (reversing the
    # two index-0 verdicts flips a perfect pass into a one-third score), so
    # the row is un-assessed instead of silently picking one.
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

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["duplicate_chunk_indexes"] == [0]
    assert result.details["total_chunks"] == 2
    assert "judge_response" in result.details


def test_duplicate_with_unrecognized_label_is_still_a_parse_failure() -> None:
    # A duplicated in-range index is ambiguous even when the second verdict's
    # label is off-scale: either the label check or the order check decides,
    # and both are judge noise rather than a graded chunk.
    scorer = _scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "On topic."},
                {"chunk_index": 0, "relevance": "kinda relevant", "reason": "Off-scale."},
            ],
        }
    )

    result = scorer.score("Answer", input="Question", context="Chunk A")

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["duplicate_chunk_indexes"] == [0]
    assert result.details["discarded_verdicts"] == 1


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


def test_non_integer_chunk_indexes_are_discarded_not_coerced() -> None:
    # int(False) == 0, int(0.9) == 0, and int("0") == 0: coercion would let a
    # sloppy judge claim chunk 0 with any of these. chunk_index must be a real
    # integer, so all three verdicts are discarded and chunk 0 goes missing.
    scorer = _scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": False, "relevance": "relevant", "reason": "Bool index."},
                {"chunk_index": 0.9, "relevance": "relevant", "reason": "Float index."},
                {"chunk_index": "0", "relevance": "relevant", "reason": "String index."},
            ],
        }
    )

    result = scorer.score("Answer", input="Question", context="Chunk A")

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["discarded_verdicts"] == 3
    assert result.details["missing_chunk_indexes"] == [0]


def test_native_labelled_blocks_are_chunks() -> None:
    # The toolkit's reference RAG apps emit "[source-id] text" blocks joined by
    # blank lines (demo_app/finance_advisor.py). Two native-format blocks must
    # count as two chunks, not one.
    scorer = _scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "On topic."},
                {"chunk_index": 1, "relevance": "relevant", "reason": "On topic."},
            ],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="[fin-1] Revenue was $10 million.\n\n[fin-2] Revenue grew 12% YoY.",
    )

    assert result.assessed
    assert result.details["total_chunks"] == 2
    assert result.details["relevant_chunks"] == 2
    assert [v["chunk_index"] for v in result.details["chunk_verdicts"]] == [0, 1]


def test_inline_brackets_do_not_split_chunks() -> None:
    # A label only counts at the start of a line: bracketed text inside a
    # passage must not become a new chunk boundary.
    scorer = _scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "On topic."},
                {"chunk_index": 1, "relevance": "relevant", "reason": "On topic."},
            ],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="[fin-1] See [appendix] for the full breakdown.\n\n[fin-2] Second block.",
    )

    assert result.assessed
    assert result.details["total_chunks"] == 2


def test_caller_context_before_labelled_blocks_is_not_a_chunk() -> None:
    # The reference RAG apps can supply caller context ahead of the retrieved
    # labelled blocks (demo_app/triage.py). That prefix is not a retrieved
    # chunk: it must neither count as one nor reach the judge, or the judge's
    # chunk indexes would shift relative to the chunks the scorer counts.
    scorer = _scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "On topic."},
                {"chunk_index": 1, "relevance": "relevant", "reason": "On topic."},
            ],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context=(
            "The year in review: revenue rose sharply across regions.\n\n"
            "[fin-1] Revenue was $10 million.\n\n[fin-2] Revenue grew 12% YoY."
        ),
    )

    assert result.assessed
    assert result.details["total_chunks"] == 2
    judge_prompt = scorer._call_judge.call_args[0][1]
    assert "The year in review" not in judge_prompt
    assert "[fin-1] Revenue was $10 million." in judge_prompt
    assert "[fin-2] Revenue grew 12% YoY." in judge_prompt


def test_markdown_link_at_line_start_is_not_a_source_label() -> None:
    # "[docs](https://example.com) useful material" starts a line with
    # bracketed text but is a Markdown link, not a "[source-id]" label: the
    # parser must not count it as one labelled chunk and discard the verdict
    # for the second section. Both sections stay chunks.
    scorer = _scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "On topic."},
                {"chunk_index": 1, "relevance": "irrelevant", "reason": "Unrelated."},
            ],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="[docs](https://example.com) useful material\n---\nUnrelated chunk",
    )

    assert result.assessed
    assert result.details["total_chunks"] == 2
    assert result.details["discarded_verdicts"] == 0


def test_labelled_context_takes_precedence_over_delimiters() -> None:
    # When line-start labels are present, the contract is the labelled-block
    # format; a literal "---" inside a passage does not split anything.
    scorer = _scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "On topic."},
                {"chunk_index": 1, "relevance": "relevant", "reason": "On topic."},
            ],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="[fin-1] Revenue was $10M --- audited figure.\n\n[fin-2] Growth note.",
    )

    assert result.assessed
    assert result.details["total_chunks"] == 2


def test_single_labelled_block_is_one_chunk() -> None:
    scorer = _scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "relevant", "reason": "On topic."},
            ],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="[fin-1] Revenue was $10 million.",
    )

    assert result.assessed
    assert result.details["total_chunks"] == 1


def test_contradictory_overall_score_is_derived_from_verdicts() -> None:
    # The judge claims a perfect score while marking every chunk irrelevant.
    # The returned score must follow the validated verdicts, not the judge's
    # own number; the contradictory value is kept in details for audit.
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "Perfect retrieval.",
            "chunk_verdicts": [
                {"chunk_index": 0, "relevance": "irrelevant", "reason": "Off-topic."},
                {"chunk_index": 1, "relevance": "irrelevant", "reason": "Unrelated."},
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
    assert result.details["raw_score"] == 0
    assert result.details["judge_score"] == 3.0
    assert result.details["relevant_chunks"] == 0
    assert result.details["total_chunks"] == 2
    # The shown explanation must follow the validated verdicts too: no
    # "Perfect retrieval." when both chunks were judged irrelevant. The
    # judge's own prose stays in details for audit.
    assert "Perfect retrieval." not in result.explanation
    assert "0 relevant" in result.explanation
    assert result.details["judge_explanation"] == "Perfect retrieval."


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
