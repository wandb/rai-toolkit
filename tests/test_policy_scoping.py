# SPDX-FileCopyrightText: 2026 Yusef Syed
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Offline regression coverage for built-in policy selection by preset."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rai_toolkit.models.base import BaseModel, ModelResponse
from rai_toolkit.workflow import ApplicationProfile, Industry, scope_assessor, scoping


class FakeModel(BaseModel):
    """Deterministic model fixture; it never makes a network request."""

    async def predict(
        self,
        input_text: str,
        context: str = "",
        **kwargs: object,
    ) -> ModelResponse:
        del input_text, context, kwargs
        return ModelResponse(output="The diagnosis is uncertain.")


def _profile(industry: Industry) -> ApplicationProfile:
    return ApplicationProfile(
        name="Scoped assistant",
        description="Offline policy-selection test fixture",
        owner_team="rai",
        owner_email="rai@example.com",
        industry=industry,
        dataset_overrides=["finqa-sample"],
    )


def _content_violations(assessor: object, model: FakeModel) -> set[str]:
    output = asyncio.run(model.predict("Summarize this loan file.")).output
    engine = assessor.policies_engine  # type: ignore[attr-defined]
    assert engine is not None
    return {
        violation.policy_name
        for violation in engine.evaluate(
            [], model_output=output, model_input="Summarize this loan file."
        )
    }


@pytest.mark.parametrize(
    ("industry", "expected_policy_sets", "expected_filenames"),
    [
        (
            Industry.FINANCIAL_SERVICES,
            [
                "EU AI Act - Article 15 (Accuracy, Robustness, Cybersecurity)",
                "Fairness Baseline",
            ],
            ["eu_ai_act_article_15.yaml", "fairness_baseline.yaml"],
        ),
        (
            Industry.HR,
            [
                "EU AI Act - Article 15 (Accuracy, Robustness, Cybersecurity)",
                "Fairness Baseline",
            ],
            ["eu_ai_act_article_15.yaml", "fairness_baseline.yaml"],
        ),
        (
            Industry.GOVERNMENT,
            [
                "EU AI Act - Article 15 (Accuracy, Robustness, Cybersecurity)",
                "Fairness Baseline",
            ],
            ["eu_ai_act_article_15.yaml", "fairness_baseline.yaml"],
        ),
        (
            Industry.GENERAL,
            [
                "EU AI Act - Article 15 (Accuracy, Robustness, Cybersecurity)",
                "Fairness Baseline",
            ],
            ["eu_ai_act_article_15.yaml", "fairness_baseline.yaml"],
        ),
        (
            Industry.HEALTHCARE,
            [
                "EU AI Act - Article 15 (Accuracy, Robustness, Cybersecurity)",
                "Fairness Baseline",
                "Healthcare / HIPAA Safeguards",
            ],
            [
                "eu_ai_act_article_15.yaml",
                "fairness_baseline.yaml",
                "healthcare_hipaa.yaml",
            ],
        ),
    ],
)
def test_built_in_presets_select_exact_policy_sets(
    industry: Industry,
    expected_policy_sets: list[str],
    expected_filenames: list[str],
) -> None:
    assessor, decision = scope_assessor(_profile(industry), FakeModel())

    assert assessor.policies_engine is not None
    assert [
        policy_set.name for policy_set in assessor.policies_engine.policy_sets
    ] == expected_policy_sets
    assert (
        "Built-in policy files selected: "
        + ", ".join(f"`{filename}`" for filename in expected_filenames)
        + "."
    ) in decision.rationale


@pytest.mark.parametrize(
    "industry",
    [Industry.FINANCIAL_SERVICES, Industry.HR, Industry.GOVERNMENT, Industry.GENERAL],
)
def test_non_healthcare_presets_cannot_fire_healthcare_policies(
    industry: Industry,
) -> None:
    model = FakeModel()
    assessor, decision = scope_assessor(_profile(industry), model)

    assert "medical-disclaimer-required" not in _content_violations(assessor, model)
    assert not any("healthcare_hipaa.yaml" in reason for reason in decision.rationale)


def test_healthcare_keeps_and_records_its_healthcare_policy() -> None:
    model = FakeModel()
    assessor, decision = scope_assessor(_profile(Industry.HEALTHCARE), model)

    assert "medical-disclaimer-required" in _content_violations(assessor, model)
    assert any("healthcare_hipaa.yaml" in reason for reason in decision.rationale)


def test_explicit_policies_dir_is_a_complete_unfiltered_override(
    tmp_path: Path,
) -> None:
    custom_policy = tmp_path / "custom.yaml"
    custom_policy.write_text(
        """\
name: Custom policies
description: Explicit policy directory fixture
version: \"1.0.0\"
policies:
  - name: custom-only-rule
    description: Custom rule
    severity: high
    trigger:
      output_contains: [diagnosis]
    frameworks: [Custom]
    remediation: Review custom policy.
""",
        encoding="utf-8",
    )
    model = FakeModel()
    assessor, decision = scope_assessor(
        _profile(Industry.FINANCIAL_SERVICES), model, policies_dir=tmp_path
    )

    assert decision.policies_dir == str(tmp_path)
    assert assessor.policies_engine is not None
    assert [policy_set.name for policy_set in assessor.policies_engine.policy_sets] == [
        "Custom policies"
    ]
    assert _content_violations(assessor, model) == {"custom-only-rule"}


def test_implicit_built_ins_fail_closed_when_their_directory_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_directory = tmp_path / "missing"
    monkeypatch.setattr(scoping, "_builtin_policy_directory", lambda: missing_directory)

    with pytest.raises(FileNotFoundError, match="Built-in policy directory is missing"):
        scope_assessor(_profile(Industry.GENERAL), FakeModel())


def test_implicit_built_ins_fail_closed_for_an_unclassified_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_directory = (
        Path(scoping.__file__).resolve().parents[1] / "policies" / "examples"
    )
    for policy in source_directory.glob("*.yaml"):
        (tmp_path / policy.name).write_text(
            policy.read_text(encoding="utf-8"), encoding="utf-8"
        )
    (tmp_path / "future_policy.yaml").write_text("name: future", encoding="utf-8")
    monkeypatch.setattr(scoping, "_builtin_policy_directory", lambda: tmp_path)

    with pytest.raises(ValueError, match="future_policy.yaml"):
        scope_assessor(_profile(Industry.GENERAL), FakeModel())


def test_implicit_built_ins_fail_closed_for_a_missing_registered_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_directory = (
        Path(scoping.__file__).resolve().parents[1] / "policies" / "examples"
    )
    for policy in source_directory.glob("*.yaml"):
        if policy.name != "healthcare_hipaa.yaml":
            (tmp_path / policy.name).write_text(
                policy.read_text(encoding="utf-8"), encoding="utf-8"
            )
    monkeypatch.setattr(scoping, "_builtin_policy_directory", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="healthcare_hipaa.yaml"):
        scope_assessor(_profile(Industry.GENERAL), FakeModel())


def test_implicit_built_ins_fail_closed_for_duplicate_registry_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scoping,
        "_COMMON_BUILTIN_POLICY_FILENAMES",
        ("eu_ai_act_article_15.yaml", "eu_ai_act_article_15.yaml"),
    )

    with pytest.raises(ValueError, match="selects a filename more than once"):
        scope_assessor(_profile(Industry.GENERAL), FakeModel())


def test_default_profile_uses_the_general_built_in_selection() -> None:
    profile = ApplicationProfile(
        name="Default assistant",
        description="Uses ApplicationProfile defaults",
        owner_team="rai",
        owner_email="rai@example.com",
        dataset_overrides=["finqa-sample"],
    )

    assessor, decision = scope_assessor(profile, FakeModel())

    assert decision.preset == "general"
    assert assessor.policies_engine is not None
    assert [policy_set.name for policy_set in assessor.policies_engine.policy_sets] == [
        "EU AI Act - Article 15 (Accuracy, Robustness, Cybersecurity)",
        "Fairness Baseline",
    ]


def test_explicit_examples_directory_remains_unfiltered_for_finance() -> None:
    examples_directory = (
        Path(scoping.__file__).resolve().parents[1] / "policies" / "examples"
    )

    assessor, decision = scope_assessor(
        _profile(Industry.FINANCIAL_SERVICES),
        FakeModel(),
        policies_dir=examples_directory,
    )

    assert decision.policies_dir == str(examples_directory)
    assert assessor.policies_engine is not None
    assert [policy_set.name for policy_set in assessor.policies_engine.policy_sets] == [
        "EU AI Act - Article 15 (Accuracy, Robustness, Cybersecurity)",
        "Fairness Baseline",
        "Healthcare / HIPAA Safeguards",
    ]


def test_explicit_empty_directory_remains_an_empty_complete_override(
    tmp_path: Path,
) -> None:
    assessor, decision = scope_assessor(
        _profile(Industry.GENERAL), FakeModel(), policies_dir=tmp_path
    )

    assert decision.policies_dir == str(tmp_path)
    assert assessor.policies_engine is not None
    assert assessor.policies_engine.policy_sets == []


def test_explicit_missing_directory_preserves_assessor_error(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        scope_assessor(
            _profile(Industry.GENERAL),
            FakeModel(),
            policies_dir=tmp_path / "missing",
        )
