# SPDX-FileCopyrightText: 2026 Yusef Syed
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Offline coverage of unavailable predictions versus usable model responses."""

import asyncio
import json
from dataclasses import asdict

import pytest

from rai_toolkit.compliance.engine import ComplianceMappingEngine
from rai_toolkit.compliance.frameworks import Framework
from rai_toolkit.evaluation import pipeline as pipeline_module
from rai_toolkit.evaluation.pipeline import RAIEvaluationPipeline
from rai_toolkit.models.base import BaseModel, ModelResponse
from rai_toolkit.scorers.base import BaseScorer, ScorerResult
from rai_toolkit.scorers.programmatic import KeywordToxicityScorer


class FakeModel(BaseModel):
    name = "offline-fake"

    def __init__(self, responses):
        super().__init__()
        self.responses = responses

    async def predict(self, input_text, context="", **kwargs):
        response = self.responses[input_text]
        if isinstance(response, BaseException):
            raise response
        return response


class SpyScorer(BaseScorer):
    category = "MIT-3.1"

    def __init__(self, name, error=None):
        super().__init__(name=name)
        self.calls = []
        self.error = error

    def score(self, output, input="", context="", expected="", rubrics=None):
        if self.error is not None:
            raise self.error
        return ScorerResult(0.8, True, self.category, "Offline measurement")

    async def score_async(self, **kwargs):
        self.calls.append(kwargs)
        return self.score(**kwargs)


def evaluation(scorers):
    engine = ComplianceMappingEngine()
    profile = engine.create_profile(
        Framework.MIT_AI_RISK, category_ids=[], name="offline"
    )
    return RAIEvaluationPipeline(engine, additional_scorers=scorers), profile


@pytest.mark.asyncio
async def test_builtin_does_not_turn_failed_predictions_into_perfect_pass():
    scorer = KeywordToxicityScorer()
    pipeline, profile = evaluation([scorer])
    results = await pipeline.run_evaluation(
        FakeModel({"failed": RuntimeError("offline failure")}),
        profile,
        [{"input": "failed"}],
    )
    assert len(results.items) == 1
    result = results.items[0].scores[scorer.name]
    assert not result.assessed and not result.passed and result.score == 0.0
    assert results.overall_score == 0.0 and not results.overall_passed
    assert results.summary[scorer.category] == {
        "mean_score": None,
        "min_score": None,
        "max_score": None,
        "pass_rate": None,
        "total_items": 0,
        "passed_items": 0,
        "failed_items": 0,
        "unassessed_items": 1,
    }


@pytest.mark.parametrize("message", ["offline failure", "", "kill hate"])
@pytest.mark.asyncio
async def test_failed_prediction_skips_every_scorer_and_retains_json_safe_row(
    message, monkeypatch
):
    scorers = [SpyScorer("first"), SpyScorer("second")]
    pipeline, profile = evaluation(scorers)
    monkeypatch.setattr(
        pipeline_module._tracing, "current_call_url", lambda: "offline://trace"
    )
    row = {
        "input": "failed",
        "context": "reference",
        "expected": "answer",
        "policy_expectations": {"MIT-3.1": "required"},
    }
    results = await pipeline.run_evaluation(
        FakeModel({"failed": RuntimeError(message)}), profile, [row]
    )
    item = results.items[0]
    assert (item.input, item.context, item.expected) == (
        "failed",
        "reference",
        "answer",
    )
    assert item.model_output == f"[ERROR: {message}]"
    assert item.metadata == {
        "weave_call_url": "offline://trace",
        "policy_expectations": row["policy_expectations"],
    }
    assert set(item.scores) == {s.name for s in scorers}
    for scorer in scorers:
        assert scorer.calls == []
        result = item.scores[scorer.name]
        assert not result.assessed and not result.passed and result.score == 0.0
        assert result.category == scorer.category
        assert result.details == {
            "scorer_name": scorer.name,
            "skipped": "model_prediction_error",
        }
    json.dumps(asdict(item), allow_nan=False)


class BrokenOutput:
    @property
    def output(self):
        raise RuntimeError("output unavailable")


class BrokenContext:
    def __str__(self):
        raise RuntimeError("context unavailable")


@pytest.mark.parametrize(
    "response",
    [BrokenOutput(), ModelResponse("valid", {"retrieved_context": BrokenContext()})],
)
@pytest.mark.asyncio
async def test_response_extraction_error_is_unassessed_without_scoring(response):
    scorer = SpyScorer("spy")
    pipeline, profile = evaluation([scorer])
    results = await pipeline.run_evaluation(
        FakeModel({"input": response}), profile, [{"input": "input"}]
    )
    assert results.items[0].model_output.startswith("[ERROR:")
    assert not results.items[0].scores[scorer.name].assessed
    assert scorer.calls == []


@pytest.mark.asyncio
async def test_mixed_run_aggregates_only_successful_predictions():
    scorer = KeywordToxicityScorer()
    pipeline, profile = evaluation([scorer])
    responses = {
        "safe": ModelResponse("Friendly greeting"),
        "toxic": ModelResponse("You should be murdered"),
        "failed": RuntimeError("offline failure"),
    }
    assessed = await pipeline.run_evaluation(
        FakeModel(responses), profile, [{"input": "safe"}, {"input": "toxic"}]
    )
    mixed = await pipeline.run_evaluation(
        FakeModel(responses), profile, [{"input": key} for key in responses]
    )
    assert len(mixed.items) == 3
    summary = mixed.summary[scorer.category]
    expected = {**assessed.summary[scorer.category], "unassessed_items": 1}
    assert summary == expected
    assert (
        summary["total_items"] == 2
        and summary["passed_items"] == 1
        and summary["failed_items"] == 1
    )
    assert (mixed.overall_score, mixed.overall_passed) == (
        assessed.overall_score,
        assessed.overall_passed,
    )


@pytest.mark.asyncio
async def test_real_error_prefix_response_is_scored_with_success_metadata(monkeypatch):
    scorer = SpyScorer("spy")
    pipeline, profile = evaluation([scorer])
    monkeypatch.setattr(
        pipeline_module._tracing, "current_call_url", lambda: "offline://trace"
    )
    row = {
        "input": "input",
        "context": "dataset context",
        "expected": "answer",
        "rubrics": [{"criterion": "use evidence"}],
        "policy_expectations": {"MIT-3.1": True},
    }
    response = ModelResponse(
        "[ERROR: legitimate text]", {"retrieved_context": "actual evidence"}
    )
    results = await pipeline.run_evaluation(
        FakeModel({"input": response}), profile, [row]
    )
    assert scorer.calls == [
        {
            "output": response.output,
            "input": "input",
            "context": "actual evidence",
            "expected": "answer",
            "rubrics": row["rubrics"],
        }
    ]
    item = results.items[0]
    assert item.scores[scorer.name].assessed
    assert item.context == row["context"] and item.expected == row["expected"]
    assert item.metadata == {
        "weave_call_url": "offline://trace",
        "policy_expectations": row["policy_expectations"],
    }


@pytest.mark.asyncio
async def test_scorer_failure_remains_unassessed_and_other_scorers_run():
    failing, healthy = (
        SpyScorer("failing", RuntimeError("scorer failed")),
        SpyScorer("healthy"),
    )
    pipeline, profile = evaluation([failing, healthy])
    results = await pipeline.run_evaluation(
        FakeModel({"input": ModelResponse("valid")}), profile, [{"input": "input"}]
    )
    scores = results.items[0].scores
    assert len(failing.calls) == len(healthy.calls) == 1
    assert (
        not scores["failing"].assessed
        and scores["failing"].explanation == "Scorer error: scorer failed"
    )
    assert scores["healthy"].assessed


@pytest.mark.parametrize("origin", ["model", "scorer"])
@pytest.mark.asyncio
async def test_cancellation_propagates(origin):
    scorer = SpyScorer("spy", asyncio.CancelledError() if origin == "scorer" else None)
    pipeline, profile = evaluation([scorer])
    response = asyncio.CancelledError() if origin == "model" else ModelResponse("valid")
    with pytest.raises(asyncio.CancelledError):
        await pipeline.run_evaluation(
            FakeModel({"input": response}), profile, [{"input": "input"}]
        )
    assert len(scorer.calls) == (1 if origin == "scorer" else 0)
