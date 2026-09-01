# SPDX-FileCopyrightText: 2026 M4h1m4
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import re
from unittest.mock import Mock

from rai_toolkit.assessment.assessor import _classify_unassessed_reason
from rai_toolkit.scorers.llm_judges import (
    CitationCorrectnessScorer,
    _extract_citations,
    _parse_source_blocks,
    _resolve_citations,
)

CONTEXT = (
    "[fair-lending] Lending decisions must use creditworthiness criteria. "
    "See also [reg-b] for the implementing regulation.\n\n"
    "[adverse-action] Adverse-action notices are required for denied applicants."
)


def _scorer(result: dict[str, object] | None = None) -> CitationCorrectnessScorer:
    scorer = CitationCorrectnessScorer(api_key="test")
    scorer._call_judge = Mock(return_value=result or {})
    return scorer


# --- context parsing -------------------------------------------------------


def test_source_blocks_split_on_line_leading_labels() -> None:
    blocks = _parse_source_blocks(CONTEXT)

    assert sorted(blocks) == ["adverse-action", "fair-lending"]
    assert blocks["fair-lending"][0] == "fair-lending"
    assert "creditworthiness criteria" in blocks["fair-lending"][1]
    assert "denied applicants" in blocks["adverse-action"][1]


def test_mid_line_bracket_does_not_start_a_new_block() -> None:
    blocks = _parse_source_blocks(CONTEXT)

    # [reg-b] sits inside the fair-lending passage, so it is not its own source.
    assert "reg-b" not in blocks
    assert "[reg-b]" in blocks["fair-lending"][1]


def test_repeated_label_keeps_the_first_block() -> None:
    blocks = _parse_source_blocks("[dup] first body\n\n[dup] second body")

    assert blocks["dup"][1] == "first body"


# --- citation extraction ---------------------------------------------------


def test_extracts_bare_and_markdown_link_markers() -> None:
    output = "Notices are required [adverse-action]; rates are capped [1](https://x/y)."

    assert _extract_citations(output) == ["adverse-action", "1"]


def test_editorial_and_spaced_brackets_are_not_citations() -> None:
    output = "The ruling [sic] applies [emphasis added] to lenders [ibid]."

    assert _extract_citations(output) == []


def test_markers_are_deduplicated_preserving_first_casing() -> None:
    output = "See [Fair-Lending] and again [fair-lending] and [FAIR-LENDING]."

    assert _extract_citations(output) == ["Fair-Lending"]


# --- three-tier resolution -------------------------------------------------


def test_resolution_sorts_markers_into_three_tiers() -> None:
    blocks = _parse_source_blocks(CONTEXT)

    resolved, ambiguous, fabricated = _resolve_citations(
        ["adverse-action", "reg-b", "reg-z-2024"], blocks, CONTEXT
    )

    assert resolved == ["adverse-action"]
    # Bracketed in the context but not a block label: our parsing is the likely
    # culprit, so the response is not accused of inventing it.
    assert ambiguous == ["reg-b"]
    # Absent from the context entirely.
    assert fabricated == ["reg-z-2024"]


def test_resolution_is_case_insensitive() -> None:
    blocks = _parse_source_blocks(CONTEXT)

    resolved, _, fabricated = _resolve_citations(["Fair-Lending"], blocks, CONTEXT)

    assert resolved == ["Fair-Lending"]  # original casing preserved
    assert fabricated == []


# --- un-assessed guard branches --------------------------------------------


def test_behavioral_refusal_row_is_unassessed_without_calling_judge() -> None:
    scorer = _scorer()

    result = scorer.score(
        "I cannot help with that [fair-lending].",
        context=CONTEXT,
        expected="The assistant should refuse to speculate.",
    )

    assert not result.assessed
    assert result.details["skipped"] == "behavioral_refusal_expected"
    scorer._call_judge.assert_not_called()


def test_refusal_check_precedes_empty_context_check() -> None:
    scorer = _scorer()

    result = scorer.score(
        "I cannot help with that.",
        context="   ",
        expected="The assistant should refuse to speculate.",
    )

    assert result.details["skipped"] == "behavioral_refusal_expected"
    scorer._call_judge.assert_not_called()


def test_missing_context_is_unassessed_without_calling_judge() -> None:
    scorer = _scorer()

    result = scorer.score("Notices are required [adverse-action].", context="  ")

    assert not result.assessed
    assert result.details["skipped"] == "empty_context"
    scorer._call_judge.assert_not_called()


def test_response_without_citations_is_unassessed() -> None:
    scorer = _scorer()

    result = scorer.score(
        "The retrieved guidance does not cover this question.", context=CONTEXT
    )

    assert not result.assessed
    assert result.details["skipped"] == "no_citations"
    scorer._call_judge.assert_not_called()


def test_citations_that_all_fail_to_resolve_are_unassessed() -> None:
    scorer = _scorer()

    # Numeric markers against slug-labelled context: a format mismatch is at
    # least as likely as fabrication, so no judgement is recorded either way.
    result = scorer.score("Notices are required [1] and [2].", context=CONTEXT)

    assert not result.assessed
    assert result.details["skipped"] == "unresolved_citations"
    scorer._call_judge.assert_not_called()


def test_ambiguous_only_citations_are_unassessed() -> None:
    scorer = _scorer()

    result = scorer.score("Per the implementing regulation [reg-b].", context=CONTEXT)

    assert not result.assessed
    assert result.details["skipped"] == "unresolved_citations"
    scorer._call_judge.assert_not_called()


# --- judged path -----------------------------------------------------------


def test_supported_citation_scores_full_and_keeps_marker() -> None:
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "The cited block supports the claim.",
            "supported_citations": [
                {
                    "marker": "adverse-action",
                    "response_span": "Notices are required for denied applicants",
                    "context_span": "notices are required for denied applicants",
                }
            ],
            "misattributed_citations": [],
        }
    )

    result = scorer.score(
        "Notices are required for denied applicants [adverse-action].",
        context=CONTEXT,
    )

    assert result.assessed
    assert result.score == 1.0
    assert result.passed
    assert not result.details["floor_applied"]
    assert result.details["supported_citations"] == [
        {
            "marker": "adverse-action",
            "response_span": "Notices are required for denied applicants",
            "context_span": "notices are required for denied applicants",
        }
    ]


def test_misattributed_citation_fails_and_names_the_real_source() -> None:
    scorer = _scorer(
        {
            "score": 1,
            "explanation": "The claim belongs to the adverse-action block.",
            "supported_citations": [],
            "misattributed_citations": [
                {
                    "marker": "fair-lending",
                    "response_span": "Notices are required for denied applicants",
                    "context_span": "Adverse-action notices are required",
                }
            ],
        }
    )

    result = scorer.score(
        "Notices are required for denied applicants [fair-lending].",
        context=CONTEXT,
    )

    assert result.score == 1 / 3
    assert not result.passed
    assert result.details["misattributed_citations"][0]["marker"] == "fair-lending"


def test_evidence_from_a_block_other_than_the_cited_one_is_discarded() -> None:
    # The span is real context, but it lives in fair-lending, not the cited
    # adverse-action block. Supporting evidence must come from the block named.
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "Claimed support.",
            "supported_citations": [
                {
                    "marker": "adverse-action",
                    "response_span": "Notices are required",
                    "context_span": "creditworthiness criteria",
                }
            ],
            "misattributed_citations": [],
        }
    )

    result = scorer.score("Notices are required [adverse-action].", context=CONTEXT)

    assert result.details["supported_citations"] == []
    assert result.details["discarded_evidence_spans"] == 1


def test_fabricated_citation_floors_the_score_over_the_judge() -> None:
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "The resolved citation checks out.",
            "supported_citations": [],
            "misattributed_citations": [],
        }
    )

    result = scorer.score(
        "Notices are required [adverse-action]. Rates are capped at 8% [reg-z-2024].",
        context=CONTEXT,
    )

    assert result.assessed
    assert result.score == 0.0
    assert not result.passed
    assert result.details["floor_applied"]
    assert result.details["fabricated_citations"] == ["reg-z-2024"]
    # The judge's own verdict survives the override, so the floor is auditable.
    assert result.details["raw_score"] == 3.0


def test_ambiguous_markers_are_recorded_but_do_not_floor() -> None:
    scorer = _scorer(
        {"score": 3, "explanation": "ok", "supported_citations": [], "misattributed_citations": []}
    )

    result = scorer.score(
        "Notices are required [adverse-action]; see also [reg-b].", context=CONTEXT
    )

    assert result.details["ambiguous_citations"] == ["reg-b"]
    assert result.details["fabricated_citations"] == []
    assert not result.details["floor_applied"]
    assert result.score == 1.0


# --- prompt assembly -------------------------------------------------------


def test_fabricated_block_is_appended_only_when_fabrication_exists() -> None:
    scorer = _scorer()

    without = scorer._format_prompt(output="o", input="i", context=CONTEXT)
    with_block = scorer._format_prompt(
        output="o", input="i", context=CONTEXT, fabricated="[reg-z-2024]"
    )

    assert "verified as fabricated" not in without
    assert "verified as fabricated" in with_block
    assert "[reg-z-2024]" in with_block
    # The JSON example's braces survived both passes intact.
    assert '"score": <0-3>' in with_block


def test_overridden_patterns_are_actually_used() -> None:
    # Regression: the patterns were once class attributes that nothing read, so
    # overriding them silently did nothing. With the bracket patterns still in
    # force this row would find no citations and come back un-assessed.
    class AngleBracketCitations(CitationCorrectnessScorer):
        citation_pattern = re.compile(r"<<([A-Za-z0-9][A-Za-z0-9._\-]*)>>")
        source_label_pattern = re.compile(
            r"^<<([A-Za-z0-9][A-Za-z0-9._\-]*)>>\s*", re.MULTILINE
        )

    scorer = AngleBracketCitations(api_key="test")
    scorer._call_judge = Mock(
        return_value={
            "score": 3,
            "explanation": "ok",
            "supported_citations": [],
            "misattributed_citations": [],
        }
    )

    result = scorer.score(
        "Creditworthiness matters <<fair-lending>>.",
        context="<<fair-lending>> Lending decisions must use creditworthiness criteria.",
    )

    assert result.assessed
    assert result.score == 1.0
    scorer._call_judge.assert_called_once()


# --- coverage-gap reporting ------------------------------------------------
# These pass the scorer's own ScorerResult to the assessor rather than a
# hand-built one, so the two files stay in step: renaming a `skipped` value in
# the scorer breaks these instead of silently degrading the report to
# "see explanation in JSON report".


def test_no_citations_row_is_named_in_the_coverage_report() -> None:
    scorer = _scorer()

    result = scorer.score(
        "The retrieved guidance does not cover this question.", context=CONTEXT
    )

    assert _classify_unassessed_reason(result) == "response cited no sources"


def test_unresolved_citations_row_is_named_in_the_coverage_report() -> None:
    scorer = _scorer()

    result = scorer.score("Notices are required [1] and [2].", context=CONTEXT)

    assert _classify_unassessed_reason(result) == "cited sources could not be resolved"


async def test_score_async_takes_the_same_path_as_score() -> None:
    # The evaluation pipeline calls score_async, not score.
    scorer = _scorer()

    result = await scorer.score_async(
        "The retrieved guidance does not cover this question.", context=CONTEXT
    )

    assert not result.assessed
    assert result.details["skipped"] == "no_citations"
    scorer._call_judge.assert_not_called()
