# SPDX-FileCopyrightText: 2026 Yusef Syed
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Coverage-gap reasons shown in assessment reports, checked on real runs."""

import asyncio
import json
import warnings

import pytest

from rai_toolkit.assessment.assessor import (
    Assessor,
    _classify_unassessed_reason,
    _coverage_gap_breakdown,
    _coverage_gap_rationale,
)
from rai_toolkit.compliance.engine import ComplianceMappingEngine
from rai_toolkit.compliance.frameworks import Framework
from rai_toolkit.evaluation.pipeline import RAIEvaluationPipeline
from rai_toolkit.examples import ExampleRegistry
from rai_toolkit.models.base import BaseModel, ModelResponse
from rai_toolkit.scorers.base import BaseScorer, ScorerResult
from rai_toolkit.scorers.llm_judges import FactualityJudge
from rai_toolkit.scorers.programmatic import KeywordToxicityScorer

PREDICTION_ERROR = "model prediction error"
NO_CONTEXT = "no grounding context"
DATASET = "clinical-triage-policy-examples"


class ScriptedModel(BaseModel):
    """Answers every row except the inputs it is told to fail on."""

    name = "offline-scripted"

    def __init__(self, failing_inputs=()):
        super().__init__()
        self.failing_inputs = set(failing_inputs)

    async def predict(self, input_text, context="", **kwargs):
        if input_text in self.failing_inputs:
            raise TimeoutError("offline timeout")
        return ModelResponse(output="Please call 911 or see a clinician today.")


class RaisingScorer(BaseScorer):
    name = "raising"
    category = "MIT-3.1"

    def score(self, output, input="", context="", **kwargs):
        raise RuntimeError("model prediction failed inside the scorer")


def offline_factuality_judge():
    """The shipped judge; it returns before any judge call on rows without context."""
    judge = FactualityJudge(api_key="offline-never-called")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("no judge call is expected for rows without context")

    judge._call_judge = fail_if_called
    return judge


def run_pipeline(model, scorers, dataset):
    engine = ComplianceMappingEngine()
    profile = engine.create_profile(
        Framework.MIT_AI_RISK, category_ids=[], name="offline"
    )
    pipeline = RAIEvaluationPipeline(engine, additional_scorers=scorers)
    return asyncio.run(pipeline.run_evaluation(model, profile, dataset))


def coverage_line(lines):
    matches = [line for line in lines if line.startswith("Coverage gap: ")]
    assert len(matches) == 1
    return matches[0]


def test_failed_prediction_is_labelled_from_the_emitted_result():
    scorer = KeywordToxicityScorer()
    results = run_pipeline(
        ScriptedModel(failing_inputs={"failed"}),
        [scorer],
        [{"input": "failed"}, {"input": "answered"}],
    )

    emitted = results.items[0].scores[scorer.name]
    assert emitted.details["skipped"] == "model_prediction_error"
    assert _classify_unassessed_reason(emitted) == PREDICTION_ERROR
    assert _coverage_gap_breakdown(results) == [
        {
            "scorer": scorer.name,
            "reason": PREDICTION_ERROR,
            "count": 1,
            "category": scorer.category,
        }
    ]
    assert coverage_line([_coverage_gap_rationale(results)]).startswith(
        "Coverage gap: 1 scorer run(s) produced no signal and were excluded "
        f"from the gate: {scorer.name} ×1 ({PREDICTION_ERROR}). "
    )


def test_rationale_counts_each_reason_of_one_scorer_separately():
    judge = offline_factuality_judge()
    results = run_pipeline(
        ScriptedModel(failing_inputs={"failed"}),
        [judge],
        [{"input": "failed"}, {"input": "no context"}, {"input": "none either"}],
    )

    assert _coverage_gap_breakdown(results) == [
        {"scorer": judge.name, "reason": NO_CONTEXT, "count": 2, "category": "MIT-3.1"},
        {
            "scorer": judge.name,
            "reason": PREDICTION_ERROR,
            "count": 1,
            "category": "MIT-3.1",
        },
    ]
    assert (
        f"from the gate: {judge.name} ×3 ({NO_CONTEXT} ×2, {PREDICTION_ERROR} ×1). "
    ) in _coverage_gap_rationale(results)


def test_single_reason_rationale_is_unchanged():
    judge = offline_factuality_judge()
    scorer = KeywordToxicityScorer()
    results = run_pipeline(
        ScriptedModel(),
        [judge, scorer],
        [{"input": "first"}, {"input": "second"}, {"input": "third"}],
    )

    assert _coverage_gap_rationale(results) == (
        "Coverage gap: 3 scorer run(s) produced no signal and were excluded "
        f"from the gate: {judge.name} ×3 ({NO_CONTEXT}). Un-assessed runs do not "
        "pass or fail; they are reported separately so coverage gaps stay visible. "
        "To close the gap: map a behavioral scorer (privacy/security/safety) for "
        "rows without retrieval context, or fix the scorer integration when the "
        "reason is a parser failure."
    )


def test_label_follows_the_skip_marker_not_the_explanation_text():
    scorer = RaisingScorer()
    results = run_pipeline(ScriptedModel(), [scorer], [{"input": "answered"}])

    emitted = results.items[0].scores[scorer.name]
    assert not emitted.assessed
    assert "model prediction failed" in emitted.explanation
    assert _classify_unassessed_reason(emitted) == "scorer runtime error"


def test_assessor_labels_failed_predictions_on_every_report_surface(
    monkeypatch, tmp_path
):
    for variable in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(variable, raising=False)
    failing_input = ExampleRegistry.load(DATASET)[0]["input_text"]
    judge = offline_factuality_judge()
    toxicity = KeywordToxicityScorer()
    assessor = Assessor(
        model=ScriptedModel(failing_inputs={failing_input}),
        preset="general",
        datasets=[DATASET],
        run_redteam=False,
        additional_scorers=[judge, toxicity],
        use_weave_evaluation=False,
    )

    with warnings.catch_warnings():
        # The general preset's own judges are not built without an API key.
        warnings.simplefilter("ignore", RuntimeWarning)
        result = asyncio.run(assessor.run())

    assert result.coverage_gaps == [
        {"scorer": judge.name, "reason": NO_CONTEXT, "count": 2, "category": "MIT-3.1"},
        {
            "scorer": judge.name,
            "reason": PREDICTION_ERROR,
            "count": 1,
            "category": "MIT-3.1",
        },
        {
            "scorer": toxicity.name,
            "reason": PREDICTION_ERROR,
            "count": 1,
            "category": "MIT-1.2",
        },
        {
            "scorer": "RegexPIIScorer",
            "reason": PREDICTION_ERROR,
            "count": 1,
            "category": "MIT-2.1",
        },
    ]
    expected_tokens = (
        f"{judge.name} ×3 ({NO_CONTEXT} ×2, {PREDICTION_ERROR} ×1); "
        f"{toxicity.name} ×1 ({PREDICTION_ERROR}); "
        f"RegexPIIScorer ×1 ({PREDICTION_ERROR}). "
    )
    assert (
        "Coverage gap: 5 scorer run(s) produced no signal and were excluded "
        f"from the gate: {expected_tokens}"
    ) in coverage_line(result.verdict_rationale)
    assert expected_tokens in result.format_summary()

    report_path = tmp_path / "report.json"
    result.to_json(report_path)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["coverage_gaps"] == result.coverage_gaps
    assert "see explanation in JSON report" not in json.dumps(saved["coverage_gaps"])

    html = result.to_html()
    assert html.count(f"<td>{PREDICTION_ERROR}</td>") == 3
    assert f"<td>{NO_CONTEXT}</td>" in html


@pytest.mark.parametrize(
    ("skipped", "expected"),
    [
        ("behavioral_refusal_expected", "behavioral/refusal row"),
        ("empty_context", NO_CONTEXT),
        ("non_finite_judge_score", "scorer integration / parser failure"),
        ("model_prediction_error", PREDICTION_ERROR),
        ("some_future_reason", "see explanation in JSON report"),
    ],
)
def test_registered_skip_markers_keep_their_labels(skipped, expected):
    result = ScorerResult(
        score=0.0,
        passed=False,
        category="MIT-3.1",
        explanation="Unassessed.",
        details={"skipped": skipped},
        assessed=False,
    )
    assert _classify_unassessed_reason(result) == expected
