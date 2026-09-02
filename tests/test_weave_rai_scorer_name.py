# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for WeaveRAIScorer name handling.

See https://github.com/wandb/rai-toolkit/issues/32.
"""

from __future__ import annotations

from rai_toolkit.scorers.base import BaseScorer, ScorerResult
from integrations.weave_integration.scorers import WeaveRAIScorer


class DummyScorer(BaseScorer):
    """Minimal scorer for testing wrapper name resolution."""

    category = "MIT-1.1"

    def score(self, output: str, input: str = "", context: str = "", **kwargs) -> ScorerResult:
        return ScorerResult(score=0.5, passed=True, category=self.category, explanation="ok")


def test_weave_rai_scorer_uses_configured_name_not_class_name() -> None:
    """Two instances of the same class with different names must not collide."""
    alpha = DummyScorer(name="alpha")
    beta = DummyScorer(name="beta")

    w_alpha = WeaveRAIScorer(rai_scorer=alpha)
    w_beta = WeaveRAIScorer(rai_scorer=beta)

    assert w_alpha.name == "alpha"
    assert w_beta.name == "beta"
    assert w_alpha.name != w_beta.name


def test_weave_rai_scorer_falls_back_to_class_name_when_empty() -> None:
    """When no name is configured, the class name is used (BaseScorer fallback)."""
    unnamed = DummyScorer()  # name defaults to class name via BaseScorer
    wrapper = WeaveRAIScorer(rai_scorer=unnamed)
    assert wrapper.name == "DummyScorer"
