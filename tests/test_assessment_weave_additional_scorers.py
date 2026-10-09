# SPDX-FileCopyrightText: 2026 Yusef Syed
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest

from rai_toolkit.assessment import Assessor
from rai_toolkit.models.base import BaseModel, ModelResponse
from rai_toolkit.scorers.base import BaseScorer, ScorerResult
from rai_toolkit.scorers.composite import CompositeScorer
from rai_toolkit.scorers.llm_judges import LLMJudgeScorer


class StubModel(BaseModel):
    async def predict(
        self,
        input_text: str,
        context: str = "",
        **kwargs: Any,
    ) -> ModelResponse:
        return ModelResponse(output="stub")


class StubScorer(BaseScorer):
    def score(
        self,
        output: str,
        input: str = "",
        context: str = "",
        **kwargs: Any,
    ) -> ScorerResult:
        return ScorerResult(
            score=1.0,
            passed=True,
            category="custom",
            explanation="stub",
        )


class AsyncContractScorer(BaseScorer):
    def __init__(self) -> None:
        super().__init__(name="async_contract")
        self.sync_calls = 0
        self.async_calls = 0

    def score(
        self,
        output: str,
        input: str = "",
        context: str = "",
        **kwargs: Any,
    ) -> ScorerResult:
        self.sync_calls += 1
        return ScorerResult(
            score=0.0,
            passed=False,
            category="custom",
            explanation="sync",
        )

    async def score_async(
        self,
        output: str,
        input: str = "",
        context: str = "",
        **kwargs: Any,
    ) -> ScorerResult:
        self.async_calls += 1
        return ScorerResult(
            score=1.0,
            passed=True,
            category="custom",
            explanation="async",
        )


class BlockingSyncScorer(BaseScorer):
    def __init__(self) -> None:
        super().__init__(name="blocking_sync")
        self._lock = threading.Lock()
        self.active_calls = 0
        self.max_active_calls = 0
        self.total_calls = 0

    def score(
        self,
        output: str,
        input: str = "",
        context: str = "",
        **kwargs: Any,
    ) -> ScorerResult:
        with self._lock:
            self.active_calls += 1
            self.total_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(0.05)
        finally:
            with self._lock:
                self.active_calls -= 1
        return ScorerResult(
            score=1.0,
            passed=True,
            category="custom",
            explanation="blocking sync",
        )


class BlockingLLMJudgeScorer(LLMJudgeScorer):
    name = "blocking_llm_judge"
    _judge_name = "FactualityJudge"

    def __init__(self) -> None:
        super().__init__(api_key="test")
        self._lock = threading.Lock()
        self.active_calls = 0
        self.max_active_calls = 0
        self.total_calls = 0

    def _call_judge(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        with self._lock:
            self.active_calls += 1
            self.total_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(0.05)
        finally:
            with self._lock:
                self.active_calls -= 1
        return {"score": 3, "explanation": "blocking llm judge"}


@dataclass
class FakeWeaveRAIScorer:
    rai_scorer: BaseScorer


def _install_fake_weave_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    adapter: Callable[[BaseScorer], FakeWeaveRAIScorer],
    captured: dict[str, Any],
    expected_result: object,
) -> None:
    weave_package = ModuleType("integrations.weave_integration")
    weave_package.__path__ = []  # type: ignore[attr-defined]

    evaluation_module = ModuleType("integrations.weave_integration.evaluation")

    class FakeWeaveEvaluationRunner:
        def __init__(self, engine: object) -> None:
            self.engine = engine

        async def get_detailed_evaluation(
            self,
            model: object,
            profile: object,
            dataset: object,
            *,
            name: str,
            include_weave_builtins: bool,
            additional_scorers: list[FakeWeaveRAIScorer],
        ) -> tuple[object, dict[str, Any]]:
            assert all(
                isinstance(scorer, FakeWeaveRAIScorer) for scorer in additional_scorers
            )
            captured.update(
                name=name,
                include_weave_builtins=include_weave_builtins,
                additional_scorers=additional_scorers,
            )
            return object(), {}

    evaluation_module.WeaveEvaluationRunner = FakeWeaveEvaluationRunner  # type: ignore[attr-defined]

    scorers_module = ModuleType("integrations.weave_integration.scorers")
    scorers_module.make_weave_rai_scorer = adapter  # type: ignore[attr-defined]

    models_module = ModuleType("integrations.weave_integration.models")
    models_module.WeaveModel = lambda **kwargs: object()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "integrations.weave_integration", weave_package)
    monkeypatch.setitem(
        sys.modules,
        "integrations.weave_integration.evaluation",
        evaluation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "integrations.weave_integration.scorers",
        scorers_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "integrations.weave_integration.models",
        models_module,
    )

    from rai_toolkit.evaluation import weave_adapter

    monkeypatch.setattr(
        weave_adapter,
        "weave_eval_results_to_evaluation_results",
        lambda *args, **kwargs: expected_result,
    )


@pytest.mark.asyncio
async def test_weave_evaluation_receives_adapted_additional_scorers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = StubScorer()
    adapted_scorer = FakeWeaveRAIScorer(scorer)
    captured: dict[str, Any] = {}
    expected_result = object()
    _install_fake_weave_modules(
        monkeypatch,
        adapter=lambda value: (
            adapted_scorer if value is scorer else FakeWeaveRAIScorer(value)
        ),
        captured=captured,
        expected_result=expected_result,
    )

    assessor = Assessor(
        model=StubModel(),
        preset="general",
        datasets=["unused"],
        additional_scorers=[scorer],
    )
    monkeypatch.setattr(assessor, "_should_use_weave_evaluation", lambda: True)
    profile = assessor.engine.create_profile_from_preset("general")

    result = await assessor._run_evaluation(profile, [{"input": "hello"}])

    assert result is expected_result
    assert captured["additional_scorers"] == [adapted_scorer]


@pytest.mark.asyncio
async def test_weave_evaluation_falls_back_to_core_pipeline_when_adaptation_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adaptable = StubScorer(name="adaptable")
    unsupported = StubScorer(name="unsupported")
    captured: dict[str, Any] = {}
    weave_result = object()
    fallback_result = object()

    def adapt(scorer: BaseScorer) -> FakeWeaveRAIScorer:
        if scorer is unsupported:
            raise TypeError("unsupported scorer")
        return FakeWeaveRAIScorer(scorer)

    _install_fake_weave_modules(
        monkeypatch,
        adapter=adapt,
        captured=captured,
        expected_result=weave_result,
    )
    assessor = Assessor(
        model=StubModel(),
        preset="general",
        datasets=["unused"],
        additional_scorers=[adaptable, unsupported],
    )
    monkeypatch.setattr(assessor, "_should_use_weave_evaluation", lambda: True)
    profile = assessor.engine.create_profile_from_preset("general")

    async def run_core_evaluation(**kwargs: Any) -> object:
        captured["core_kwargs"] = kwargs
        return fallback_result

    monkeypatch.setattr(assessor.pipeline, "run_evaluation", run_core_evaluation)

    with caplog.at_level(logging.WARNING):
        result = await assessor._run_evaluation(profile, [{"input": "hello"}])

    assert result is fallback_result
    assert captured["core_kwargs"]["model"] is assessor.model
    assert captured["core_kwargs"]["profile"] is profile
    assert captured["core_kwargs"]["dataset"] == [{"input": "hello"}]
    assert assessor.pipeline.additional_scorers == [adaptable, unsupported]
    assert "additional_scorers" not in captured
    assert (
        "Weave-native evaluation unavailable (unsupported scorer); using core pipeline."
        in caplog.text
    )


@pytest.mark.asyncio
async def test_real_weave_evaluation_awaits_async_additional_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("weave")
    from rai_toolkit.compliance.frameworks import ComplianceProfile, Framework

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")

    profile = ComplianceProfile(
        name="empty",
        framework=Framework.MIT_AI_RISK,
        categories=[],
    )
    dataset = [{"input": "hello"}]

    weave_scorer = AsyncContractScorer()
    weave_assessor = Assessor(
        model=StubModel(),
        preset="general",
        datasets=["unused"],
        additional_scorers=[weave_scorer],
        use_weave_evaluation=True,
        include_weave_builtin_scorers=False,
    )
    weave_result = await weave_assessor._run_evaluation(profile, dataset)

    core_scorer = AsyncContractScorer()
    core_assessor = Assessor(
        model=StubModel(),
        preset="general",
        datasets=["unused"],
        additional_scorers=[core_scorer],
        use_weave_evaluation=False,
    )
    core_result = await core_assessor._run_evaluation(profile, dataset)

    scorer_name = "async_contract"
    assert weave_result.metadata["evaluation_backend"] == "weave"
    assert weave_result.metadata["scorers_used"] == [scorer_name]
    assert set(weave_result.items[0].scores) == {scorer_name}
    assert set(core_result.items[0].scores) == {scorer_name}
    weave_cell = weave_result.items[0].scores[scorer_name]
    core_cell = core_result.items[0].scores[scorer_name]

    assert weave_scorer.sync_calls == 0
    assert weave_scorer.async_calls == 1
    assert core_scorer.sync_calls == 0
    assert core_scorer.async_calls == 1
    assert weave_cell == core_cell
    assert weave_cell.score == 1.0
    assert weave_cell.passed is True
    assert weave_cell.explanation == "async"


@pytest.mark.asyncio
@pytest.mark.parametrize("composite", [False, True])
async def test_real_weave_evaluation_keeps_sync_scorers_concurrent(
    monkeypatch: pytest.MonkeyPatch,
    composite: bool,
) -> None:
    pytest.importorskip("weave")
    from rai_toolkit.compliance.frameworks import ComplianceProfile, Framework

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")

    profile = ComplianceProfile(
        name="empty",
        framework=Framework.MIT_AI_RISK,
        categories=[],
    )
    dataset = [{"input": f"hello {index}"} for index in range(12)]
    sync_scorer = BlockingSyncScorer()
    llm_scorer = BlockingLLMJudgeScorer()
    assessor = Assessor(
        model=StubModel(),
        preset="general",
        datasets=["unused"],
        additional_scorers=(
            [CompositeScorer([sync_scorer, llm_scorer], name="composite")]
            if composite
            else [sync_scorer, llm_scorer]
        ),
        use_weave_evaluation=True,
        include_weave_builtin_scorers=False,
    )

    result = await assessor._run_evaluation(profile, dataset)

    scorer_names = (
        {"composite"} if composite else {"blocking_sync", "blocking_llm_judge"}
    )
    assert result.metadata["evaluation_backend"] == "weave"
    assert set(result.metadata["scorers_used"]) == scorer_names
    assert len(result.items) == len(dataset)
    for item in result.items:
        assert set(item.scores) == scorer_names
        assert all(cell.score == 1.0 for cell in item.scores.values())
        assert all(cell.passed for cell in item.scores.values())

    for scorer in (sync_scorer, llm_scorer):
        assert scorer.total_calls == len(dataset)
        assert scorer.active_calls == 0
        assert scorer.max_active_calls > 1
