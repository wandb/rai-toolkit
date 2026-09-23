# SPDX-FileCopyrightText: 2026 Kunyang Zhang
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Offline tests for `KeywordToxicityScorer` keyword matching.

When explicitly configured, the scorer can flag a response or block it through
`GuardedModel(block_on_scorer_fail=True)`. These tests cover false hits for text that
contains none of the configured keywords as words of its own.
"""

from __future__ import annotations

import pytest

from rai_toolkit.scorers import KeywordToxicityScorer


@pytest.fixture
def scorer() -> KeywordToxicityScorer:
    return KeywordToxicityScorer()


@pytest.mark.parametrize(
    "response",
    [
        "She has strong skills in Python and SQL.",
        "Please describe the skill set required.",
        "The system reached stability after the fix.",
        "The installer is stable and well tested.",
        "An offshoot of the original project.",
        "That sentence is an oxymoron.",
        "The paper describes a bombardier beetle.",
        "Weather forecasts are usually accurate.",
    ],
)
def test_words_that_merely_contain_a_keyword_are_not_flagged(
    scorer: KeywordToxicityScorer, response: str
) -> None:
    """Regression: "skills" contains "kill", "stability" contains "stab",
    "offshoot" contains "shoot" and "oxymoron" contains "moron"."""
    result = scorer.score(response)

    assert result.details["toxic_categories"] == {}
    assert result.details["total_matches"] == 0
    assert result.score == 1.0
    assert result.passed
    assert result.assessed


@pytest.mark.parametrize(
    ("response", "category"),
    [
        ("I will kill the process.", "violence"),
        ("KILL THE PROCESS", "violence"),
        ("That is an attack on the schema.", "violence"),
        ("You are a moron.", "harassment"),
        ("It is worthLESS.", "harassment"),
        ("They called the group subhuman.", "hate_speech"),
        ("Please end your life.", "self_harm"),
        ("Do not cut yourself.", "self_harm"),
        ("A self-harm reference was removed.", "self_harm"),
    ],
)
def test_keywords_are_still_detected_as_words(
    scorer: KeywordToxicityScorer, response: str, category: str
) -> None:
    """The boundary only removes matches inside a longer Latin word."""
    result = scorer.score(response)

    assert category in result.details["toxic_categories"]
    assert not result.passed


@pytest.mark.parametrize(
    ("response", "category"),
    [
        ("You should be murdered.", "violence"),
        ("They are killing the process.", "violence"),
        ("He was stabbing the log file.", "violence"),
        ("They were bombing the cache.", "violence"),
        ("The service was attacked.", "violence"),
        ("It destroys the index.", "violence"),
        ("Bombs were mentioned.", "violence"),
        ("They hated the output.", "hate_speech"),
        ("Stop hating on the design.", "hate_speech"),
    ],
)
def test_inflected_keywords_stay_detected(
    scorer: KeywordToxicityScorer, response: str, category: str
) -> None:
    """Regression: a plain word boundary on the right also stopped "murdered",
    "killing", "stabbing" and "bombing" from matching, so the guardrail below
    allowed them through."""
    result = scorer.score(response)

    assert category in result.details["toxic_categories"]
    assert not result.passed


def test_a_trimmed_stem_on_its_own_is_not_a_hit(scorer: KeywordToxicityScorer) -> None:
    """"hate" is trimmed to "hat" to match "-ing", but "hat" alone is headwear."""
    result = scorer.score("Wear a hat.")

    assert result.details["toxic_categories"] == {}
    assert result.passed


_DEFAULT_KEYWORD_COUNT = 21
"""Keywords configured in the default categories, which is the score denominator."""

_SINGLE_HIT_SCORE = 0.5238095238095238
"""Score of exactly one canonical hit under the default configuration.

Pinned as a literal because this change must not touch the scoring formula: a base
word and every form that maps to it have to keep returning the same score.
"""


@pytest.mark.parametrize(
    ("form", "canonical", "category"),
    [
        ("killer", "kill", "violence"),
        ("killers", "kill", "violence"),
        ("murderer", "murder", "violence"),
        ("murderers", "murder", "violence"),
        ("shooter", "shoot", "violence"),
        ("bomber", "bomb", "violence"),
        ("attacker", "attack", "violence"),
        ("hateful", "hate", "hate_speech"),
        ("stupidity", "stupid", "harassment"),
        ("idiotic", "idiot", "harassment"),
        ("moronic", "moron", "harassment"),
        ("worthlessness", "worthless", "harassment"),
        ("inferiority", "inferior", "hate_speech"),
        ("verminous", "vermin", "hate_speech"),
        ("assaultive", "assault", "violence"),
    ],
)
def test_keyword_derivatives_are_detected_as_their_canonical_keyword(
    scorer: KeywordToxicityScorer, form: str, canonical: str, category: str
) -> None:
    """Regression: each of these forms was reported as `score=1.0`, `passed=True`
    and `assessed=True` with no keyword match."""
    result = scorer.score(form)

    assert result.details["toxic_categories"] == {category: [canonical]}
    assert result.details["total_matches"] == 1
    assert result.score == _SINGLE_HIT_SCORE
    assert not result.passed
    assert result.assessed


def test_a_derivative_does_not_change_the_score_denominator(
    scorer: KeywordToxicityScorer,
) -> None:
    """The derivative is an alternative of the canonical keyword, not a new keyword,
    so the 21 configured keywords - and the score of one hit - are unaffected."""
    assert sum(len(words) for words in scorer._categories.values()) == (
        _DEFAULT_KEYWORD_COUNT
    )
    assert scorer.score("kill").score == _SINGLE_HIT_SCORE
    assert scorer.score("killer").score == scorer.score("kill").score


@pytest.mark.parametrize(
    "response",
    [
        "They killed the process and the killers escaped.",
        "The shooter and the shooters left.",
        "It was an attack by an attacker.",
    ],
)
def test_a_canonical_keyword_counts_once_however_many_forms_occur(
    scorer: KeywordToxicityScorer, response: str
) -> None:
    """One category contributes one canonical keyword at most, so repeating a family
    must not push the score down like several distinct keywords would."""
    result = scorer.score(response)

    assert result.details["total_matches"] == 1
    assert result.score == _SINGLE_HIT_SCORE


def test_derivatives_of_different_keywords_count_separately(
    scorer: KeywordToxicityScorer,
) -> None:
    result = scorer.score("A killer and a murderer.")

    assert result.details["toxic_categories"] == {"violence": ["kill", "murder"]}
    assert result.details["total_matches"] == 2


def test_a_derivative_does_not_weaken_the_benign_controls(
    scorer: KeywordToxicityScorer,
) -> None:
    """The forms added here must not turn the existing benign controls into hits."""
    for response in (
        "She has strong skills in Python.",
        "The system reached stability.",
        "An offshoot of the project.",
        "That sentence is an oxymoron.",
        "The paper describes a bombardier beetle.",
    ):
        assert scorer.score(response).passed, response


@pytest.mark.parametrize("response", ["killbed", "killring", "bombted"])
def test_doubling_must_repeat_the_keywords_own_final_consonant(
    scorer: KeywordToxicityScorer, response: str
) -> None:
    """Regression: any character in `[bdglmnprt]` used to be accepted before "-ed"
    or "-ing", so these synthetic non-inflections counted as hits.

    Kept separate from the derivative tests so a passing derivative case cannot hide
    a suffix regression.
    """
    result = scorer.score(response)

    assert result.details["toxic_categories"] == {}
    assert result.details["total_matches"] == 0
    assert result.score == 1.0
    assert result.passed


@pytest.mark.parametrize(
    ("response", "canonical"),
    [
        ("stabbed", "stab"),
        ("stabbing", "stab"),
        ("bombed", "bomb"),
        ("bombing", "bomb"),
    ],
)
def test_doubled_consonant_inflections_stay_detected(
    scorer: KeywordToxicityScorer, response: str, canonical: str
) -> None:
    """The other half of the suffix correction: tightening the doubled consonant must
    not drop the forms it exists for."""
    result = scorer.score(response)

    assert result.details["toxic_categories"] == {"violence": [canonical]}
    assert result.score == _SINGLE_HIT_SCORE
    assert not result.passed


async def test_guardrail_blocks_a_derivative_threat() -> None:
    """The derivative mapping has to reach an explicitly configured blocking
    guardrail, not only direct scoring."""
    from rai_toolkit.guardrails import GuardedModel
    from rai_toolkit.models import CallableModel

    guarded = GuardedModel(
        model=CallableModel(
            lambda text, context="", **kwargs: "The killer was identified."
        ),
        output_scorers=[KeywordToxicityScorer()],
        block_on_scorer_fail=True,
    )

    response = await guarded.predict("a harmless prompt")

    assert response.metadata["blocked"] is True
    assert response.metadata["blocked_by"] == "scorer:KeywordToxicityScorer"
    assert response.output == guarded.blocked_response


async def test_guardrail_blocks_an_inflected_threat() -> None:
    """An explicitly configured blocking guardrail must reject an inflected threat
    detected by its keyword scorer."""
    from rai_toolkit.guardrails import GuardedModel
    from rai_toolkit.models import CallableModel

    guarded = GuardedModel(
        model=CallableModel(lambda text, context="", **kwargs: "You should be murdered."),
        output_scorers=[KeywordToxicityScorer()],
        block_on_scorer_fail=True,
    )

    response = await guarded.predict("a harmless prompt")

    assert response.metadata["blocked"] is True
    assert response.metadata["blocked_by"] == "scorer:KeywordToxicityScorer"
    assert response.output == guarded.blocked_response


async def test_guardrail_allows_an_ordinary_answer() -> None:
    """The other half of the same guardrail: a word that merely contains a keyword
    must not block a response."""
    from rai_toolkit.guardrails import GuardedModel
    from rai_toolkit.models import CallableModel

    guarded = GuardedModel(
        model=CallableModel(
            lambda text, context="", **kwargs: "She has strong skills in Python."
        ),
        output_scorers=[KeywordToxicityScorer()],
        block_on_scorer_fail=True,
    )

    response = await guarded.predict("a harmless prompt")

    assert response.metadata["blocked"] is False
    assert response.output == "She has strong skills in Python."


def test_matched_keyword_is_reported_in_its_configured_form(
    scorer: KeywordToxicityScorer,
) -> None:
    result = scorer.score("KILL THE PROCESS")

    assert result.details["toxic_categories"] == {"violence": ["kill"]}


def test_extra_keywords_apply_to_the_instance_that_declares_them() -> None:
    scorer = KeywordToxicityScorer(extra_keywords={"violence": ["zap"]})

    result = scorer.score("The tool will zap the record.")

    assert result.details["toxic_categories"] == {"violence": ["zap"]}
    assert not result.passed


def test_extra_keywords_can_introduce_a_new_category() -> None:
    scorer = KeywordToxicityScorer(extra_keywords={"profanity": ["darn"]})

    result = scorer.score("Well, darn.")

    assert result.details["toxic_categories"] == {"profanity": ["darn"]}


def test_extra_keywords_do_not_leak_into_other_instances() -> None:
    """Regression: a shallow copy of the category mapping extended the lists held
    by the class attribute, so a custom keyword reached every later scorer."""
    KeywordToxicityScorer(extra_keywords={"violence": ["zap"]})

    assert "zap" not in KeywordToxicityScorer.TOXIC_CATEGORIES["violence"]

    default_scorer = KeywordToxicityScorer()
    assert "zap" not in default_scorer._categories["violence"]
    assert default_scorer.score("The tool will zap the record.").details[
        "toxic_categories"
    ] == {}


def test_extra_keywords_do_not_leak_between_two_configured_scorers() -> None:
    first = KeywordToxicityScorer(extra_keywords={"violence": ["zap"]})
    second = KeywordToxicityScorer(extra_keywords={"violence": ["squash"]})

    assert first.score("squash").details["toxic_categories"] == {}
    assert second.score("zap").details["toxic_categories"] == {}


def test_keywords_outside_latin_script_still_match() -> None:
    """Word boundaries built from `\\b` would never match these, because
    consecutive characters of such scripts are word characters on both sides."""
    scorer = KeywordToxicityScorer(extra_keywords={"violence": ["暴力"]})

    result = scorer.score("这是一段暴力内容。")

    assert result.details["toxic_categories"] == {"violence": ["暴力"]}


def test_response_without_any_keyword_scores_a_full_pass(
    scorer: KeywordToxicityScorer,
) -> None:
    result = scorer.score("The migration finished without incidents.")

    assert result.score == 1.0
    assert result.passed
    assert result.explanation == "No toxic keywords detected"
    assert result.category == "MIT-1.2"
