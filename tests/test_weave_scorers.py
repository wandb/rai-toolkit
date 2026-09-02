# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Tests for Weave scorer adapter name handling and collision prevention (issue #32)."""

from __future__ import annotations

from typing import Any

from integrations.weave_integration.scorers import (
    _wrapped_scorer_display_name,
    make_weave_rai_scorer,
)
from rai_toolkit.scorers.base import BaseScorer, ScorerResult


class ConfigurableTestScorer(BaseScorer):
    """Test scorer that returns a score and explanation based on its configuration."""

    def __init__(
        self, name: str | None = None, score_val: float = 0.8, category: str = "MIT-1.1"
    ) -> None:
        super().__init__(name=name, category=category)
        self.score_val = score_val

    def score(
        self,
        output: str,
        input: str = "",
        context: str = "",
        **kwargs: Any,
    ) -> ScorerResult:
        return ScorerResult(
            score=self.score_val,
            passed=self.score_val >= self.threshold,
            category=self.category,
            explanation=f"Scored by {self.name}",
        )


def test_two_instances_of_same_scorer_class_have_distinct_names() -> None:
    """Two instances of the same scorer class must have distinct names on WeaveRAIScorer."""
    s1 = ConfigurableTestScorer(name="alpha", score_val=0.9)
    s2 = ConfigurableTestScorer(name="beta", score_val=0.3)

    w1 = make_weave_rai_scorer(s1)
    w2 = make_weave_rai_scorer(s2)

    assert w1.name == "alpha"
    assert w2.name == "beta"
    assert w1.name != w2.name


def test_empty_scorer_name_falls_back_to_class_name() -> None:
    """When an RAI scorer has an empty name, WeaveRAIScorer falls back to the class name."""
    s = ConfigurableTestScorer(name="")
    # Manually clear name attribute in case BaseScorer filled it
    s.name = ""

    w = make_weave_rai_scorer(s)
    assert w.name == "ConfigurableTestScorer"


def test_custom_name_override_in_make_weave_rai_scorer() -> None:
    """Passing an explicit name to make_weave_rai_scorer overrides the scorer's name."""
    s = ConfigurableTestScorer(name="default_name")
    w = make_weave_rai_scorer(s, name="custom_override")
    assert w.name == "custom_override"


def test_wrapped_scorer_display_name_uses_configured_name() -> None:
    """_wrapped_scorer_display_name formats with the configured scorer name."""
    s = ConfigurableTestScorer(name="custom_judge", category="MIT-2.1")
    wrapper = make_weave_rai_scorer(s)

    call_mock = type("Call", (), {"inputs": {"self": wrapper}})()
    display_name = _wrapped_scorer_display_name(call_mock)

    assert "custom_judge" in display_name
    assert "MIT-2.1" in display_name


def test_two_instances_in_weave_evaluation_do_not_collide() -> None:
    """Two differently named instances of one scorer class both appear in evaluation results."""
    s_alpha = ConfigurableTestScorer(name="alpha", score_val=0.95)
    s_beta = ConfigurableTestScorer(name="beta", score_val=0.25)

    w_alpha = make_weave_rai_scorer(s_alpha)
    w_beta = make_weave_rai_scorer(s_beta)

    # Evaluate directly with the two wrapped scorers
    row_output = "Model response text"
    res_alpha = w_alpha.score(output=row_output)
    res_beta = w_beta.score(output=row_output)

    assert res_alpha["score"] == 0.95
    assert res_alpha["explanation"] == "Scored by alpha"

    assert res_beta["score"] == 0.25
    assert res_beta["explanation"] == "Scored by beta"

    # In Weave evaluations, evaluation scorers are keyed by their .name
    scorers = [w_alpha, w_beta]
    scorer_names = [s.name for s in scorers]
    assert scorer_names == ["alpha", "beta"]
    assert len(set(scorer_names)) == 2

    # Simulate evaluation aggregation dictionary
    summary = {s.name: s.score(output=row_output) for s in scorers}
    assert "alpha" in summary
    assert "beta" in summary
    assert summary["alpha"]["score"] == 0.95
    assert summary["beta"]["score"] == 0.25
