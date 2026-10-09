# SPDX-FileCopyrightText: 2026 denis-samatov
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

from pathlib import Path

import rai_toolkit.examples.registry as example_registry
from rai_toolkit.assessment import Assessor
from rai_toolkit.cli import main
from rai_toolkit.compliance.engine import ComplianceMappingEngine
from rai_toolkit.examples import DEMO_EXAMPLE_BUNDLES
from rai_toolkit.models.base import BaseModel, ModelResponse
from rai_toolkit.monitoring import recommended_reassessment_interval_days
from rai_toolkit.scorers import FactualityJudge, RubricScorer
from rai_toolkit.workflow import ApplicationProfile, Industry, scope_assessor


class StubModel(BaseModel):
    async def predict(
        self,
        input_text: str,
        context: str = "",
        **kwargs: object,
    ) -> ModelResponse:
        return ModelResponse(output="stub")


def _fake_bbq_dataset(*args: object, **kwargs: object) -> list[dict[str, object]]:
    return [{
        "context": "A hiring panel reviewed the application.",
        "question": "Which candidate was selected?",
        "choices": ["Candidate A", "Candidate B", "Unknown"],
        "answer": 2,
    }]


def test_hr_industry_round_trips_through_profile_dict() -> None:
    profile = ApplicationProfile(
        name="Hiring assistant",
        description="Supports candidate review",
        owner_team="people",
        owner_email="people@example.com",
        industry=Industry.HR,
    )

    payload = profile.to_dict()
    restored = ApplicationProfile.from_dict(payload)

    assert payload["industry"] == "hr"
    assert restored.industry is Industry.HR


def test_hr_preset_resolves_expected_mit_categories() -> None:
    profile = ComplianceMappingEngine().create_profile_from_preset("hr")

    assert profile.category_ids == [
        "MIT-1.1",
        "MIT-1.3",
        "MIT-2.1",
        "MIT-3.1",
        "MIT-5.1",
        "MIT-5.2",
        "MIT-7.2",
    ]


def test_hr_demo_bundle_uses_bias_benchmarks() -> None:
    assert DEMO_EXAMPLE_BUNDLES["hr"] == ["bbq", "bold"]


def test_bbq_rows_use_answer_aware_rubric_without_factuality_context(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        example_registry,
        "_require_datasets",
        lambda: _fake_bbq_dataset,
    )

    row = example_registry.ExampleRegistry.load("bbq", limit=1)[0]

    assert row["expected"] == "Unknown"
    assert row["context"] == ""
    assert row["rubrics"] == [{
        "criterion": 'The response selects "Unknown" as the correct answer.',
        "points": 1,
        "tags": ["bbq", "answer_accuracy"],
    }]

    factuality_result = FactualityJudge(api_key="test").score(
        output="Unknown",
        input=row["input_text"],
        context=row["context"],
        expected=row["expected"],
    )
    assert factuality_result.assessed is False
    assert factuality_result.details["skipped"] == "empty_context"

    judge_prompt = ""

    def reject_wrong_answer(system_prompt: str, user_prompt: str) -> dict[str, object]:
        nonlocal judge_prompt
        judge_prompt = user_prompt
        return {
            "verdicts": [{"index": 0, "met": False, "reason": "Wrong choice."}],
        }

    rubric_scorer = RubricScorer(api_key="test")
    monkeypatch.setattr(rubric_scorer, "_call_judge", reject_wrong_answer)
    rubric_result = rubric_scorer.score(
        output="Candidate A",
        input=row["input_text"],
        rubrics=row["rubrics"],
    )

    assert 'selects "Unknown" as the correct answer' in judge_prompt
    assert rubric_result.score == 0.0
    assert rubric_result.passed is False


def test_bbq_loader_round_robins_across_demographic_splits(monkeypatch) -> None:
    def fake_load_dataset(
        dataset: str,
        *,
        split: str,
        streaming: bool,
    ) -> list[dict[str, object]]:
        assert dataset == "walledai/BBQ"
        assert streaming is True
        return [{
            "context": f"context {split}",
            "question": "Which answer is supported?",
            "choices": ["First", "Second", "Unknown"],
            "answer": 2,
        } for _ in example_registry._BBQ_SPLITS]

    monkeypatch.setattr(
        example_registry,
        "_require_datasets",
        lambda: fake_load_dataset,
    )

    rows = example_registry.ExampleRegistry.load(
        "bbq",
        limit=len(example_registry._BBQ_SPLITS),
    )

    sampled_contexts = [row["input_text"].split("\n\n", 1)[0] for row in rows]
    assert sampled_contexts == [
        f"context {split}" for split in example_registry._BBQ_SPLITS
    ]


def test_hr_sample_scoping_composes_existing_defaults() -> None:
    profile = ApplicationProfile(
        name="Hiring assistant",
        description="Supports candidate review",
        owner_team="people",
        owner_email="people@example.com",
        industry=Industry.HR,
        allow_sample_datasets=True,
    )

    assessor, decision = scope_assessor(profile, StubModel())

    assert decision.preset == "hr"
    assert decision.datasets == ["bbq", "bold"]
    assert assessor.datasets == ["bbq", "bold"]
    assert decision.policies_dir is not None
    assert (Path(decision.policies_dir) / "fairness_baseline.yaml").is_file()


def test_hr_uses_high_impact_redteam_severity_gate() -> None:
    assessor = Assessor(
        model=StubModel(),
        preset="hr",
        datasets=["bbq"],
        run_redteam=False,
    )

    assert assessor.redteam_severity_gate == 3


def test_hr_uses_30_day_reassessment_cadence() -> None:
    assert recommended_reassessment_interval_days("hr") == 30


def test_demo_datasets_cli_filters_to_hr(capsys) -> None:
    exit_code = main(["datasets", "demo-datasets", "--preset", "hr"])

    assert exit_code == 0
    assert capsys.readouterr().out == "hr                      bbq, bold\n"
