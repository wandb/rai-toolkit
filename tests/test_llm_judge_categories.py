# SPDX-FileCopyrightText: 2026 Yusef Syed
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

from rai_toolkit.scorers import FactualityJudge


def test_llm_judge_preserves_class_category_by_default() -> None:
    scorer = FactualityJudge(api_key="test")

    assert scorer.category == "MIT-3.1"


def test_llm_judge_accepts_explicit_category_override() -> None:
    scorer = FactualityJudge(api_key="test", category="custom")

    assert scorer.category == "custom"


def test_llm_judge_accepts_explicit_empty_category_override() -> None:
    scorer = FactualityJudge(api_key="test", category="")

    assert scorer.category == ""
