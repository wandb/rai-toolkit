# SPDX-FileCopyrightText: 2026 M4h1m4
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import re
from unittest.mock import Mock

import pytest

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


def _covering_judge(
    output: str,
    context: str,
    score: int = 3,
    supported: list | None = None,
    misattributed: list | None = None,
) -> dict:
    """A judge reply that returns a verifiable verdict for every resolved marker.

    Coverage is required before a row can be assessed, so a fixture returning an
    empty verdict list is now un-assessed rather than scored. Spans are taken
    verbatim from the row so they survive verification: the block body for the
    context span, and the opening of the response for the response span.
    """
    blocks = _parse_source_blocks(context)
    resolved, _, _, _ = _resolve_citations(
        _extract_citations(output), blocks, context
    )
    return {
        "score": score,
        "explanation": "The cited blocks support their claims.",
        "supported_citations": [
            {
                "marker": marker,
                "response_span": output[:10],
                "context_span": blocks[marker.lower()][1],
            }
            for marker in resolved
        ]
        + list(supported or []),
        "misattributed_citations": list(misattributed or []),
    }


def _covering_scorer(
    output: str, context: str, **kwargs
) -> CitationCorrectnessScorer:
    """Scorer whose mocked judge fully covers the row's resolved citations."""
    return _scorer(_covering_judge(output, context, **kwargs))



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


def test_repeated_label_is_dropped_rather_than_resolved_to_the_first() -> None:
    # Keeping the first left the judge reading a context this scorer could not
    # reproduce, so evidence quoted from the later block was rejected while the
    # score still passed. Ambiguous source data is excluded instead.
    blocks = _parse_source_blocks("[dup] first body\n\n[dup] second body")

    assert blocks == {}


def test_empty_block_is_not_a_citable_source() -> None:
    # A label with no body cannot support any claim, so it must not be citable.
    blocks = _parse_source_blocks("[source-a]\n[source-b] Other text here.")

    assert sorted(blocks) == ["source-b"]


# --- citation extraction ---------------------------------------------------


def test_extracts_bare_and_markdown_link_markers() -> None:
    output = "Notices are required [adverse-action]; rates are capped [1](https://x/y)."

    assert [(c.marker, c.from_link) for c in _extract_citations(output)] == [
        ("adverse-action", False),
        ("1", True),
    ]


def test_editorial_and_spaced_brackets_are_not_citations() -> None:
    output = "The ruling [sic] applies [emphasis added] to lenders [ibid]."

    assert _extract_citations(output) == []


def test_markers_are_deduplicated_preserving_first_casing() -> None:
    output = "See [Fair-Lending] and again [fair-lending] and [FAIR-LENDING]."

    assert [c.marker for c in _extract_citations(output)] == ["Fair-Lending"]


# --- three-tier resolution -------------------------------------------------


def test_resolution_sorts_markers_into_three_tiers() -> None:
    blocks = _parse_source_blocks(CONTEXT)

    resolved, ambiguous, fabricated, ignored = _resolve_citations(
        _extract_citations("[adverse-action] [reg-b] [reg-z-2024]"), blocks, CONTEXT
    )

    assert resolved == ["adverse-action"]
    # Bracketed in the context but not a block label: our parsing is the likely
    # culprit, so the response is not accused of inventing it.
    assert ambiguous == ["reg-b"]
    # Absent from the context entirely, and shaped like the real labels.
    assert fabricated == ["reg-z-2024"]
    assert ignored == []


def test_resolution_is_case_insensitive() -> None:
    blocks = _parse_source_blocks(CONTEXT)

    resolved, _, fabricated, _ignored = _resolve_citations(
        _extract_citations("[Fair-Lending]"), blocks, CONTEXT
    )

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


def test_ambiguous_markers_are_recorded_and_block_a_complete_assessment() -> None:
    # An ambiguous marker is a plausible citation this scorer could not grade,
    # so the row has only been partly assessed and must not report as complete.
    # It is still not an accusation: nothing is marked fabricated.
    output = "Notices are required [adverse-action]; see also [reg-b]."
    scorer = _covering_scorer(output, CONTEXT)

    result = scorer.score(output, context=CONTEXT)

    assert not result.assessed
    assert result.details["skipped"] == "partial_citation_coverage"
    assert result.details["ambiguous_citations"] == ["reg-b"]
    assert result.details["fabricated_citations"] == []


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

    output = "Creditworthiness matters <<fair-lending>>."
    context = "<<fair-lending>> Lending decisions must use creditworthiness criteria."
    scorer = AngleBracketCitations(api_key="test")
    scorer._call_judge = Mock(
        return_value={
            "score": 3,
            "explanation": "ok",
            "supported_citations": [
                {
                    "marker": "fair-lending",
                    "response_span": "Creditworthiness matters",
                    "context_span": "Lending decisions must use creditworthiness",
                }
            ],
            "misattributed_citations": [],
        }
    )

    result = scorer.score(output, context=context)

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


# --- review #27: bracketed tokens that are not citations --------------------
# A bracketed token only counts as a citation when it resembles the labels the
# context actually uses. Previously any unresolved bracket became a "fabricated"
# citation and floored the score, so a correctly cited response failed merely
# for containing an array index or a markdown link.

@pytest.mark.parametrize(
    "output,stray",
    [
        ("Notices required [adverse-action]. See arr[0] in the sample.", "0"),
        ("Notices required [adverse-action]. Let f([x]) be the input.", "x"),
        ("Notices required [adverse-action]. [TODO] confirm this.", "TODO"),
        (
            "Notices required [adverse-action]. See [Wikipedia](https://en.wikipedia.org/wiki/X).",
            "Wikipedia",
        ),
    ],
)
def test_non_citation_brackets_do_not_floor_a_valid_response(
    output: str, stray: str
) -> None:
    scorer = _covering_scorer(output, CONTEXT)

    result = scorer.score(output, context=CONTEXT)

    assert result.score == 1.0
    assert result.passed
    assert not result.details["floor_applied"]
    assert result.details["fabricated_citations"] == []
    assert result.details["ignored_markers"] == [stray]


def test_markdown_link_text_still_resolves_when_it_names_a_real_source() -> None:
    # Link text is never a fabrication candidate, but it is still a citation when
    # it happens to name a parsed source.
    output = "Notices are required [adverse-action](https://docs/adverse-action)."
    scorer = _covering_scorer(output, CONTEXT)

    result = scorer.score(output, context=CONTEXT)

    assert result.assessed
    assert result.score == 1.0
    assert result.details["ignored_markers"] == []


def test_a_real_fabrication_alongside_stray_brackets_still_floors() -> None:
    # Guards against over-correcting: the shape test must not swallow genuine
    # fabrications just because other junk is present.
    output = "Notices required [adverse-action]. Rates capped [reg-z-2024]. See arr[0]."
    scorer = _covering_scorer(output, CONTEXT)

    result = scorer.score(output, context=CONTEXT)

    assert result.score == 0.0
    assert result.details["floor_applied"]
    assert result.details["fabricated_citations"] == ["reg-z-2024"]
    assert result.details["ignored_markers"] == ["0"]


# --- review #27: the all-unresolved branch ---------------------------------


def test_all_unresolved_but_label_shaped_fails_instead_of_skipping() -> None:
    # Context parsed into labelled blocks and the markers match how those blocks
    # are named, so these are fabricated sources rather than a format mismatch.
    scorer = _scorer({})

    result = scorer.score(
        "Rates are capped at 8% [reg-z-2024] per [reg-x-1998].", context=CONTEXT
    )

    assert result.assessed
    assert result.score == 0.0
    assert not result.passed
    assert result.details["floor_applied"]
    assert result.details["fabricated_citations"] == ["reg-z-2024", "reg-x-1998"]
    assert result.details["raw_score"] is None
    scorer._call_judge.assert_not_called()


def test_structurally_mismatched_markers_stay_unassessed() -> None:
    # Numeric markers against slug labels: a citation-format mismatch is at
    # least as likely as a fabrication, so no judgement is recorded.
    scorer = _scorer({})

    result = scorer.score("Notices are required [1] and [2].", context=CONTEXT)

    assert not result.assessed
    assert result.details["skipped"] == "unresolved_citations"
    assert result.details["fabricated_citations"] == []
    assert result.details["ignored_markers"] == ["1", "2"]
    scorer._call_judge.assert_not_called()


def test_unassessed_rows_still_name_the_markers_they_found() -> None:
    # Previously these lists were hardcoded empty, so a coverage-gap report could
    # not say which sources were involved.
    scorer = _scorer({})

    result = scorer.score("Per the implementing regulation [reg-b].", context=CONTEXT)

    assert not result.assessed
    assert result.details["ambiguous_citations"] == ["reg-b"]


def test_unlabelled_context_cannot_produce_a_fabrication() -> None:
    # With no parsed labels there are no signatures to match, so nothing is
    # accused and the row stays un-assessed.
    scorer = _scorer({})

    result = scorer.score(
        "Revenue grew 12% [reg-z-2024].",
        context="FY2024 revenue: $12.4B. FY2023 revenue: $10.8B.",
    )

    assert not result.assessed
    assert result.details["skipped"] == "unresolved_citations"
    assert result.details["fabricated_citations"] == []


# --- review #27 item 2: evidence must come from a block the response cited ---


def test_evidence_for_an_uncited_block_cannot_cover_a_cited_one() -> None:
    # The response cites only [adverse-action]. fair-lending is a real parsed
    # block and the span is real text from it, but crediting that verdict would
    # score a citation the response never made, and it leaves the citation that
    # was made without any verdict at all.
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "ok",
            "supported_citations": [
                {
                    "marker": "fair-lending",
                    "response_span": "Notices are required",
                    "context_span": "creditworthiness criteria",
                }
            ],
            "misattributed_citations": [],
        }
    )

    result = scorer.score("Notices are required [adverse-action].", context=CONTEXT)

    assert not result.assessed
    assert result.details["skipped"] == "incomplete_judge_verdicts"
    assert "adverse-action" in result.explanation
    assert result.details["supported_citations"] == []


def test_misattributed_evidence_for_an_uncited_block_cannot_cover_a_cited_one() -> None:
    scorer = _scorer(
        {
            "score": 1,
            "explanation": "ok",
            "supported_citations": [],
            "misattributed_citations": [
                {
                    "marker": "fair-lending",
                    "response_span": "Notices are required",
                    "context_span": "Adverse-action notices are required",
                }
            ],
        }
    )

    result = scorer.score("Notices are required [adverse-action].", context=CONTEXT)

    assert not result.assessed
    assert result.details["skipped"] == "incomplete_judge_verdicts"
    assert result.details["misattributed_citations"] == []


def test_evidence_from_the_wrong_block_leaves_the_citation_unestablished() -> None:
    # The span is real context, but it lives in fair-lending rather than the
    # cited adverse-action block. Discarding it leaves the citation with no
    # surviving verdict, so the row cannot be reported as assessed.
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

    assert not result.assessed
    assert result.details["skipped"] == "incomplete_judge_verdicts"


def test_extra_unverifiable_evidence_is_discarded_but_still_scores() -> None:
    # Coverage is satisfied by the valid verdict, so the row is assessed; the
    # bogus extra is dropped and counted rather than accepted.
    output = "Notices are required for denied applicants [adverse-action]."
    scorer = _covering_scorer(
        output,
        CONTEXT,
        supported=[
            {
                "marker": "adverse-action",
                "response_span": "Notices are required",
                "context_span": "a passage the judge invented",
            }
        ],
    )

    result = scorer.score(output, context=CONTEXT)

    assert result.assessed
    assert result.score == 1.0
    assert len(result.details["supported_citations"]) == 1
    assert result.details["discarded_evidence_spans"] == 1


def test_evidence_for_a_cited_block_is_still_accepted() -> None:
    # Guards against over-narrowing: legitimate evidence must survive.
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "ok",
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
        "Notices are required for denied applicants [adverse-action].", context=CONTEXT
    )

    assert len(result.details["supported_citations"]) == 1
    assert result.details["discarded_evidence_spans"] == 0


# --- review #27 item 3: the judge is told which citations to grade ----------


def test_prompt_scopes_the_judge_to_resolved_citations() -> None:
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "ok",
            "supported_citations": [],
            "misattributed_citations": [],
        }
    )

    scorer.score(
        "Notices required [adverse-action]; see also [reg-b]; and arr[0].",
        context=CONTEXT,
    )

    prompt = scorer._call_judge.call_args.args[1]
    assert "**Grade only these citations:** [adverse-action]" in prompt
    # The ambiguous marker and the array index must not be offered for grading.
    assert "[reg-b]" not in prompt.split("**Grade only these citations:**")[1]
    assert "[0]" not in prompt.split("**Grade only these citations:**")[1]


def test_scope_block_is_absent_when_nothing_is_graded() -> None:
    scorer = _scorer()

    prompt = scorer._format_prompt(output="o", input="i", context=CONTEXT)

    assert "Grade only these citations" not in prompt
    # The JSON example's escaped braces survived both formatting passes.
    assert '"score": <0-3>' in prompt


def test_scope_and_fabricated_blocks_can_both_appear() -> None:
    scorer = _scorer()

    prompt = scorer._format_prompt(
        output="o",
        input="i",
        context=CONTEXT,
        resolved="[adverse-action]",
        fabricated="[reg-z-2024]",
    )

    assert "**Grade only these citations:** [adverse-action]" in prompt
    assert "verified as fabricated" in prompt
    assert '"score": <0-3>' in prompt


# --- review #27 round 2, item 1: label styles that cannot carry an accusation ---
# The shape signature only separates citations from ordinary brackets when the
# source ids have structure. Against bare words or bare numbers, [x], [TODO] and
# arr[0] are indistinguishable from a real label, so accusing on them produced a
# false compliance failure on a correctly cited response.

JUDGE_PASS = {
    "score": 3,
    "explanation": "The cited block supports the claim.",
    "supported_citations": [],
    "misattributed_citations": [],
}


@pytest.mark.parametrize(
    "context,output,stray",
    [
        ("[doc] Retrieved guidance about lending.", "Per [doc], let f([x]) be the input.", "x"),
        ("[doc] Retrieved guidance about lending.", "Per [doc]. [TODO] confirm this.", "TODO"),
        ("[1] Retrieved guidance about lending.", "Per [1], see arr[0] in the sample.", "0"),
    ],
)
def test_indistinguishable_labels_never_produce_a_fabrication(
    context: str, output: str, stray: str
) -> None:
    scorer = _covering_scorer(output, context)

    result = scorer.score(output, context=context)

    assert result.score == 1.0
    assert result.passed
    assert not result.details["floor_applied"]
    assert result.details["fabricated_citations"] == []
    assert result.details["ignored_markers"] == [stray]


def test_indistinguishable_labels_with_nothing_resolved_are_unassessed() -> None:
    scorer = _scorer(JUDGE_PASS)

    result = scorer.score("See [other] for detail.", context="[doc] Retrieved guidance.")

    assert not result.assessed
    assert result.details["skipped"] == "unsupported_label_style"
    scorer._call_judge.assert_not_called()


def test_structured_labels_still_accuse() -> None:
    # Control: the fix must not turn the scorer into one that never accuses.
    output = "Notices required [adverse-action]. Rates capped [reg-z-2024]."
    scorer = _covering_scorer(output, CONTEXT)

    result = scorer.score(output, context=CONTEXT)

    assert result.score == 0.0
    assert result.details["fabricated_citations"] == ["reg-z-2024"]


@pytest.mark.parametrize("separator", ["-", ".", "_"])
def test_any_separator_makes_a_label_grammar_distinguishable(separator: str) -> None:
    label = f"doc{separator}1"
    absent = f"doc{separator}9"
    output = f"Per [{label}] and [{absent}]."
    context = f"[{label}] Retrieved guidance."
    scorer = _covering_scorer(output, context)

    result = scorer.score(output, context=context)

    assert result.details["fabricated_citations"] == [absent]


# --- review #27 round 2, item 2: classification must not depend on order ---


@pytest.mark.parametrize(
    "output",
    [
        "Per [fair-lending]. See [fake-id](https://x/y) and later [fake-id].",
        "Per [fair-lending]. See [fake-id] and later [fake-id](https://x/y).",
    ],
)
def test_mixed_form_markers_classify_the_same_in_either_order(output: str) -> None:
    scorer = _covering_scorer(output, CONTEXT)

    result = scorer.score(output, context=CONTEXT)

    assert result.details["fabricated_citations"] == []
    assert result.details["ignored_markers"] == ["fake-id"]
    assert result.score == 1.0


def test_extraction_collapses_link_form_across_occurrences() -> None:
    for output in (
        "[fake-id](https://x/y) then [fake-id]",
        "[fake-id] then [fake-id](https://x/y)",
    ):
        (citation,) = _extract_citations(output)
        assert citation.marker == "fake-id"
        assert citation.from_link is True


# --- review #27 round 2, item 7/8: empty and duplicate blocks ---


def test_citation_to_an_empty_block_is_not_gradeable() -> None:
    scorer = _scorer(JUDGE_PASS)

    result = scorer.score(
        "Per [source-a].", context="[source-a]\n[source-b] Other text here."
    )

    assert not result.assessed
    assert result.details["ambiguous_citations"] == ["source-a"]
    scorer._call_judge.assert_not_called()


def test_citation_to_a_duplicated_label_is_not_gradeable() -> None:
    scorer = _scorer(JUDGE_PASS)

    result = scorer.score("Per [dup].", context="[dup] first body\n\n[dup] second body")

    assert not result.assessed
    assert result.details["ambiguous_citations"] == ["dup"]
    scorer._call_judge.assert_not_called()


def test_a_surviving_block_is_still_gradeable_alongside_a_dropped_one() -> None:
    # Control: excluding bad source data must not disable the good sources too.
    output = "Per [keep-this]."
    context = "[dup] first\n\n[dup] second\n\n[keep-this] Real body text."
    scorer = _covering_scorer(output, context)

    result = scorer.score(output, context=context)

    assert result.assessed
    assert result.score == 1.0


# --- review #27 round 2, item 2: judge output must be usable and complete ---


@pytest.mark.parametrize(
    "score,reason",
    [
        ("NaN", "not a finite number"),
        (float("inf"), "not finite"),
        (99, "above the rubric maximum"),
        (-5, "below zero"),
        ("abc", "not numeric at all"),
        (None, "missing from the reply"),
    ],
)
def test_unusable_judge_score_is_unassessed(score: object, reason: str) -> None:
    # "NaN" normalised to a perfect 1.0 because min(1.0, nan) is 1.0 in Python;
    # 99 clamped upward to the same result; "abc" and a missing key raised out
    # of float() as unhandled exceptions.
    output = "Notices are required for denied applicants [adverse-action]."
    scorer = _covering_scorer(output, CONTEXT)
    scorer._call_judge.return_value = {
        **scorer._call_judge.return_value,
        "score": score,
    }

    result = scorer.score(output, context=CONTEXT)

    assert not result.assessed, reason
    assert result.details["skipped"] == "invalid_judge_score"


def test_valid_scores_across_the_rubric_are_accepted() -> None:
    output = "Notices are required for denied applicants [adverse-action]."
    for raw, expected in ((3, 1.0), ("2", 2 / 3), (1, 1 / 3), (0, 0.0)):
        scorer = _covering_scorer(output, CONTEXT, score=raw)
        result = scorer.score(output, context=CONTEXT)
        assert result.assessed
        assert result.score == pytest.approx(expected)


def test_verdict_missing_for_a_resolved_citation_is_unassessed() -> None:
    # Two resolved citations, a verdict for one, and a score of 3 previously
    # passed the row while one citation had never been assessed.
    output = "Lending uses creditworthiness [fair-lending]. Notices required [adverse-action]."
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "Looks fine.",
            "supported_citations": [
                {
                    "marker": "fair-lending",
                    "response_span": "Lending uses creditworthiness",
                    "context_span": "Lending decisions must use creditworthiness",
                }
            ],
            "misattributed_citations": [],
        }
    )

    result = scorer.score(output, context=CONTEXT)

    assert not result.assessed
    assert result.details["skipped"] == "incomplete_judge_verdicts"
    assert "adverse-action" in result.explanation


def test_a_misattributed_verdict_counts_as_covering_a_citation() -> None:
    # Coverage asks whether every citation got a checkable answer, not whether
    # the answers were favourable.
    output = "Notices are required for denied applicants [fair-lending]."
    scorer = _scorer(
        {
            "score": 1,
            "explanation": "Attributed to the wrong block.",
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

    result = scorer.score(output, context=CONTEXT)

    assert result.assessed
    assert result.score == pytest.approx(1 / 3)
    assert not result.passed


def test_a_proven_fabrication_still_fails_when_judge_output_is_unusable() -> None:
    # "unless a deterministic local finding already proves failure": an unusable
    # reply must not rescue a response that provably cited a nonexistent source.
    output = "Notices required [adverse-action]. Rates capped [reg-z-2024]."
    scorer = _covering_scorer(output, CONTEXT)
    scorer._call_judge.return_value = {
        **scorer._call_judge.return_value,
        "score": "NaN",
    }

    result = scorer.score(output, context=CONTEXT)

    assert result.assessed
    assert result.score == 0.0
    assert not result.passed
    assert result.details["floor_applied"]
    assert result.details["judge_output_rejected"] == "invalid_judge_score"
    assert result.details["fabricated_citations"] == ["reg-z-2024"]


def test_unusable_judge_output_is_named_in_the_coverage_report() -> None:
    output = "Notices are required for denied applicants [adverse-action]."
    scorer = _covering_scorer(output, CONTEXT)
    scorer._call_judge.return_value = {
        **scorer._call_judge.return_value,
        "score": "NaN",
    }

    result = scorer.score(output, context=CONTEXT)

    assert _classify_unassessed_reason(result) == "the judge returned an unusable score"


def test_incomplete_verdicts_are_named_in_the_coverage_report() -> None:
    output = "Lending uses creditworthiness [fair-lending]. Notices required [adverse-action]."
    scorer = _scorer(
        {"score": 3, "explanation": "ok", "supported_citations": [], "misattributed_citations": []}
    )

    result = scorer.score(output, context=CONTEXT)

    assert (
        _classify_unassessed_reason(result) == "the judge did not assess every citation"
    )


# --- review #27 round 2, item 3: partial resolution must not report complete ---


def test_partly_resolved_row_is_unassessed() -> None:
    # source-b appears only inside source-a's block, so it is ambiguous and never
    # reaches the judge. Grading source-a and reporting the row as complete gave
    # a confident result for a citation nobody assessed.
    context = (
        "[source-a] Guidance about lending. See also [source-b] for detail.\n\n"
        "[other-block] Unrelated passage."
    )
    output = "Per [source-a] and [source-b]."
    scorer = _covering_scorer(output, context)

    result = scorer.score(output, context=context)

    assert not result.assessed
    assert result.details["skipped"] == "partial_citation_coverage"
    assert result.details["ambiguous_citations"] == ["source-b"]
    assert "source-b" in result.explanation


def test_ignored_markers_do_not_block_a_complete_assessment() -> None:
    # Control, and the reason "any unresolved citation" cannot mean every
    # non-resolved bucket: an ignored marker was determined not to be a citation,
    # so treating it as blocking would undo the stray-bracket fix.
    output = "Notices are required for denied applicants [adverse-action]. See arr[0]."
    scorer = _covering_scorer(output, CONTEXT)

    result = scorer.score(output, context=CONTEXT)

    assert result.assessed
    assert result.score == 1.0
    assert result.details["ignored_markers"] == ["0"]


def test_a_fabrication_still_fails_rather_than_going_unassessed() -> None:
    # Ambiguity must not downgrade a proven failure into a coverage gap.
    context = "[fair-lending] Lending rules. See also [reg-b] for detail."
    output = "Per [fair-lending], [reg-b], and [reg-z-2024]."
    scorer = _covering_scorer(output, context)

    result = scorer.score(output, context=context)

    assert result.assessed
    assert result.score == 0.0
    assert result.details["fabricated_citations"] == ["reg-z-2024"]


def test_partial_coverage_is_named_in_the_coverage_report() -> None:
    output = "Notices are required [adverse-action]; see also [reg-b]."
    scorer = _covering_scorer(output, CONTEXT)

    result = scorer.score(output, context=CONTEXT)

    assert (
        _classify_unassessed_reason(result)
        == "only some of the response's citations could be resolved"
    )


# --- review #27 round 2, item 4: the floor must explain itself ---


def test_floor_explanation_names_the_fabricated_sources() -> None:
    # The floor produced score 0 with the judge's text saying the response was
    # fully supported, so the verdict and its stated reason disagreed.
    output = "Lending uses creditworthiness [fair-lending]. Rates capped [reg-z-2024]."
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "The real citation is fully supported.",
            "supported_citations": [
                {
                    "marker": "fair-lending",
                    "response_span": "Lending uses creditworthiness",
                    "context_span": "Lending decisions must use creditworthiness",
                }
            ],
            "misattributed_citations": [],
        }
    )

    result = scorer.score(output, context=CONTEXT)

    assert result.score == 0.0
    assert not result.passed
    assert result.details["floor_applied"]
    assert "reg-z-2024" in result.explanation
    assert "absent from the retrieved context" in result.explanation
    # The judge's own words are kept, after the deterministic finding.
    assert "The real citation is fully supported." in result.explanation
    assert (
        result.details["judge_explanation"] == "The real citation is fully supported."
    )


def test_unfloored_rows_keep_the_judge_explanation_unchanged() -> None:
    output = "Notices are required for denied applicants [adverse-action]."
    scorer = _covering_scorer(output, CONTEXT)

    result = scorer.score(output, context=CONTEXT)

    assert not result.details["floor_applied"]
    assert result.explanation == "The cited blocks support their claims."
    assert "absent from the retrieved context" not in result.explanation
