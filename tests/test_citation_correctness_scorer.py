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
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "The cited block supports the claim.",
            "supported_citations": [],
            "misattributed_citations": [],
        }
    )

    result = scorer.score(output, context=CONTEXT)

    assert result.score == 1.0
    assert result.passed
    assert not result.details["floor_applied"]
    assert result.details["fabricated_citations"] == []
    assert result.details["ignored_markers"] == [stray]


def test_markdown_link_text_still_resolves_when_it_names_a_real_source() -> None:
    # Link text is never a fabrication candidate, but it is still a citation when
    # it happens to name a parsed source.
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "ok",
            "supported_citations": [],
            "misattributed_citations": [],
        }
    )

    result = scorer.score(
        "Notices are required [adverse-action](https://docs/adverse-action).",
        context=CONTEXT,
    )

    assert result.assessed
    assert result.score == 1.0
    assert result.details["ignored_markers"] == []


def test_a_real_fabrication_alongside_stray_brackets_still_floors() -> None:
    # Guards against over-correcting: the shape test must not swallow genuine
    # fabrications just because other junk is present.
    scorer = _scorer(
        {
            "score": 3,
            "explanation": "ok",
            "supported_citations": [],
            "misattributed_citations": [],
        }
    )

    result = scorer.score(
        "Notices required [adverse-action]. Rates capped [reg-z-2024]. See arr[0].",
        context=CONTEXT,
    )

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


def test_evidence_for_an_uncited_block_is_discarded() -> None:
    # The response cites only [adverse-action]. fair-lending is a real parsed
    # block, and the span is real text from it, but crediting it would mean
    # scoring a citation the response never made.
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

    assert result.details["supported_citations"] == []
    assert result.details["discarded_evidence_spans"] == 1


def test_misattributed_evidence_for_an_uncited_block_is_discarded() -> None:
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

    assert result.details["misattributed_citations"] == []
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
    scorer = _scorer(JUDGE_PASS)

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
    scorer = _scorer(JUDGE_PASS)

    result = scorer.score(
        "Notices required [adverse-action]. Rates capped [reg-z-2024].", context=CONTEXT
    )

    assert result.score == 0.0
    assert result.details["fabricated_citations"] == ["reg-z-2024"]


@pytest.mark.parametrize("separator", ["-", ".", "_"])
def test_any_separator_makes_a_label_grammar_distinguishable(separator: str) -> None:
    label = f"doc{separator}1"
    absent = f"doc{separator}9"
    scorer = _scorer(JUDGE_PASS)

    result = scorer.score(
        f"Per [{label}] and [{absent}].", context=f"[{label}] Retrieved guidance."
    )

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
    scorer = _scorer(JUDGE_PASS)

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
    scorer = _scorer(JUDGE_PASS)

    result = scorer.score(
        "Per [keep-this].",
        context="[dup] first\n\n[dup] second\n\n[keep-this] Real body text.",
    )

    assert result.assessed
    assert result.score == 1.0
