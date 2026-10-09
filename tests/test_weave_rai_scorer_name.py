# SPDX-FileCopyrightText: 2026 EffNine
# SPDX-FileCopyrightText: 2026 Karan Nisar
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Regression tests for WeaveRAIScorer name handling.

See https://github.com/wandb/rai-toolkit/issues/32.
"""

from __future__ import annotations

import pytest

from rai_toolkit.scorers.base import BaseScorer, ScorerResult

weave = pytest.importorskip("weave")


class DummyScorer(BaseScorer):
    """Minimal scorer for testing wrapper name resolution."""

    category = "MIT-1.1"

    def score(
        self,
        output: str,
        input: str = "",
        context: str = "",
        **kwargs: object,
    ) -> ScorerResult:
        score = 1.0 if self.name == "alpha" else 0.0
        return ScorerResult(
            score=score,
            passed=bool(score),
            category=self.category,
            explanation=self.name,
        )


def test_weave_rai_scorer_uses_configured_name_not_class_name() -> None:
    """Two instances of the same class with different names must not collide."""
    from integrations.weave_integration.scorers import WeaveRAIScorer

    alpha = DummyScorer(name="alpha")
    beta = DummyScorer(name="beta")

    w_alpha = WeaveRAIScorer(rai_scorer=alpha)
    w_beta = WeaveRAIScorer(rai_scorer=beta)

    assert w_alpha.name == "alpha"
    assert w_beta.name == "beta"
    assert w_alpha.name != w_beta.name


def test_weave_rai_scorer_falls_back_to_class_name_when_empty() -> None:
    """When no name is configured, the class name is used (BaseScorer fallback)."""
    from integrations.weave_integration.scorers import WeaveRAIScorer

    unnamed = DummyScorer()  # name defaults to class name via BaseScorer
    wrapper = WeaveRAIScorer(rai_scorer=unnamed)
    assert wrapper.name == "DummyScorer"


@pytest.mark.asyncio
async def test_real_weave_evaluation_preserves_both_configured_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weave must keep both results from differently named instances."""
    from integrations.weave_integration.scorers import make_weave_rai_scorer

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")

    class StaticWeaveModel(weave.Model):
        @weave.op()
        async def predict(self, input_text: str) -> dict[str, str]:
            return {"output": "response"}

    evaluation = weave.Evaluation(
        dataset=[{"input_text": "prompt"}],
        scorers=[
            make_weave_rai_scorer(DummyScorer(name="alpha")),
            make_weave_rai_scorer(DummyScorer(name="beta")),
        ],
    )

    results = await evaluation.get_eval_results(StaticWeaveModel())
    row = next(iter(results.rows))

    assert set(row["scores"]) == {"alpha", "beta"}
    assert row["scores"]["alpha"]["score"] == 1.0
    assert row["scores"]["beta"]["score"] == 0.0

    summary = await evaluation.summarize(results)

    assert summary["alpha"]["score"]["mean"] == 1.0
    assert summary["beta"]["score"]["mean"] == 0.0
