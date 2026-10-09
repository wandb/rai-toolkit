# SPDX-FileCopyrightText: 2026 Karan Nisar
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Offline execution against the real optional Garak package and its probe data."""

import asyncio
import builtins
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
import io
import json
import random
import socket
from threading import Barrier, Event

import pytest

pytest.importorskip("garak")
from garak import _config
from garak.attempt import Conversation, Message, Turn
from garak.generators.base import Generator

from integrations.garak_integration import adapter
from rai_toolkit.models.base import BaseModel, ModelResponse
from rai_toolkit.models.callable import CallableModel

REFUSAL = "I cannot help with that request."
PROBES = adapter.default_garak_probes()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """A real probe must use bundled data and must not call a remote service."""

    def denied(*args, **kwargs):
        raise AssertionError("Network access is forbidden in Garak integration tests")

    for target, name in (
        (socket, "create_connection"),
        (socket, "getaddrinfo"),
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
    ):
        monkeypatch.setattr(target, name, denied)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def local_model(output=REFUSAL):
    async def predict(text, **kwargs):
        return output

    return CallableModel(predict, name="offline-garak-test")


@pytest.mark.parametrize("count", [0, 1, 3])
async def test_native_generator_contract_and_trace_context(count, monkeypatch):
    calls = []
    trace = ContextVar("garak-test-trace", default=None)
    trace.set("parent-attack")

    async def predict(text, **kwargs):
        calls.append((text, trace.get()))
        return "local response"

    monkeypatch.setattr(_config.system, "parallel_requests", 4, raising=False)
    generator = adapter.RAIGenerator(CallableModel(predict))
    prompt = Conversation([Turn("user", Message(text="  original\ntext  "))])
    assert adapter.RAIGenerator.generate is Generator.generate
    outputs = await asyncio.to_thread(generator.generate, prompt, count)
    assert len(outputs) == count
    assert all(
        isinstance(output, Message) and output.text == "local response"
        for output in outputs
    )
    assert calls == [("  original\ntext  ", "parent-attack")] * count
    assert _config.system.parallel_requests == 4


@pytest.mark.parametrize(
    "prompt",
    [
        Conversation(),
        Conversation([Turn("system", Message(text="system instructions"))]),
        Conversation(
            [Turn("user", Message(text="first")), Turn("user", Message(text="second"))]
        ),
        Conversation([Turn("user", Message(text="caption", data_path="missing.png"))]),
        Conversation([Turn("user", Message(text=None))]),
    ],
)
async def test_unsupported_inputs_are_not_silently_flattened(prompt):
    def unexpected(*args, **kwargs):
        pytest.fail("An unsupported prompt reached the model")

    generator = adapter.RAIGenerator(CallableModel(unexpected))
    with pytest.raises(ValueError, match="single user turn|text only"):
        await asyncio.to_thread(generator.generate, prompt)


@pytest.mark.parametrize("count", [-1, 1.5])
def test_invalid_generation_count(count):
    generator = adapter.RAIGenerator(local_model())
    with pytest.raises(AssertionError, match="generations_per_call"):
        generator.generate(Conversation([Turn("user", Message(text="hello"))]), count)


@pytest.mark.parametrize("spec", PROBES, ids=lambda spec: spec.attack_id)
@pytest.mark.parametrize(
    "output,succeeded", [(REFUSAL, False), ("local non-refusal response", True)]
)
async def test_real_curated_probe(spec, output, succeeded):
    calls = []

    async def predict(text, **kwargs):
        calls.append(text)
        return output

    report = await adapter.run_garak_probes(
        CallableModel(predict), [spec], prompt_cap=1
    )
    assert len(calls) == 1
    assert calls[0].strip()
    (result,) = report.results
    assert result.error is None
    assert result.succeeded is succeeded
    assert result.model_output == output
    assert result.prompt == calls[0]
    assert report.total_assessed == 1
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize("spec", PROBES, ids=lambda spec: spec.attack_id)
def test_prompt_caps_preserve_metadata(spec):
    rng_state = random.getstate()
    probe = adapter._instantiate_probe(spec.probe_path, prompt_cap=2)
    assert random.getstate() == rng_state
    assert 0 < len(probe.prompts) <= 2
    for name in ("triggers", "pi_prompts", "_prompt_intents", "prompt_intents"):
        values = getattr(probe, name, None)
        if values is not None:
            assert len(values) == len(probe.prompts)
    if hasattr(probe, "pi_prompts"):
        assert probe.prompts == [row["prompt"] for row in probe.pi_prompts]
    if hasattr(probe, "encoding_name"):
        import base64

        for prompt, trigger in zip(probe.prompts, probe.triggers):
            assert base64.b64encode(trigger.encode()).decode() in prompt


@pytest.mark.parametrize("output", ["", "   ", None])
async def test_blank_and_missing_outputs_are_unassessed(output):
    class EmptyModel(BaseModel):
        async def predict(self, input_text, **kwargs):
            return ModelResponse(output=output)

    report = await adapter.run_garak_probes(EmptyModel(), [PROBES[0]], prompt_cap=1)
    (result,) = report.results
    assert result.assessed is False
    assert "no model output" in result.error
    assert report.overall_resistance_rate is None
    assert (
        json.loads(json.dumps(report.to_dict(), allow_nan=False))["total_assessed"] == 0
    )


@pytest.mark.parametrize(
    "failure", ["model", "generator", "resource", "missing_class", "probe"]
)
async def test_execution_failures_are_unassessed(failure, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("intentional integration failure")

    model = local_model()
    spec = PROBES[0]
    if failure == "model":
        model = CallableModel(fail)
    elif failure == "generator":
        monkeypatch.setattr(adapter, "RAIGenerator", fail)
    elif failure == "missing_class":
        spec = replace(spec, probe_path="garak.probes.dan.MissingProbe")
    elif failure == "resource":
        original_open = builtins.open

        def missing_resource(path, *args, **kwargs):
            if str(path).endswith("Dan_11_0.json"):
                raise FileNotFoundError("packaged probe resource unavailable")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", missing_resource)
    else:
        from garak.probes.dan import Dan_11_0

        monkeypatch.setattr(Dan_11_0, "probe", fail)
    previous = _config.transient.reportfile
    report = await adapter.run_garak_probes(model, [spec], prompt_cap=1)
    assert report.total_errors == 1
    assert report.total_assessed == 0
    assert report.overall_resistance_rate is None
    assert report.results[0].error
    assert _config.transient.reportfile is previous


async def test_non_refusal_evidence_survives_a_later_refusal():
    outputs = iter(["local non-refusal response", REFUSAL])
    calls = []

    async def predict(text, **kwargs):
        calls.append(text)
        return next(outputs)

    report = await adapter.run_garak_probes(
        CallableModel(predict), [PROBES[1]], prompt_cap=2
    )
    (result,) = report.results
    assert len(calls) == 2
    assert result.succeeded is True
    assert result.model_output == "local non-refusal response"
    assert result.prompt == calls[0]


async def test_overlapping_runs_restore_state_and_do_not_change_catalogs(monkeypatch):
    from garak.probes.encoding import EncodingMixin

    # Force overlap so a single run cannot close another run's report sink.
    rendezvous = Barrier(2, timeout=10)

    async def predict(text, **kwargs):
        rendezvous.wait()
        return REFUSAL

    monkeypatch.setattr(_config.transient, "reportfile", None)
    tags = list(EncodingMixin.tags)
    config = deepcopy(vars(_config.run))
    rng_state = random.getstate()
    for _ in range(2):
        reports = await asyncio.gather(
            *(
                adapter.run_garak_probes(
                    CallableModel(predict), [PROBES[3]], prompt_cap=1
                )
                for _ in range(2)
            )
        )
        assert all(report.total_assessed == 1 for report in reports)
        assert _config.transient.reportfile is None
        assert adapter._report_sink is None
    assert EncodingMixin.tags == tags
    assert vars(_config.run) == config
    assert random.getstate() == rng_state


async def test_native_report_stream_is_preserved(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(_config.transient, "reportfile", stream)
    report = await adapter.run_garak_probes(local_model(), [PROBES[0]], prompt_cap=1)
    assert report.total_assessed == 1
    assert _config.transient.reportfile is stream
    assert not stream.closed
    (record,) = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert record["probe_classname"] == "dan.Dan_11_0"


async def test_cancellation_keeps_sink_alive_until_worker_exits(monkeypatch):
    entered, release, finished = Event(), Event(), Event()
    sink_seen = []

    async def predict(text, **kwargs):
        entered.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test did not release the worker")
        sink_seen.append(_config.transient.reportfile)
        return REFUSAL

    from garak.probes.dan import Dan_11_0

    original = Dan_11_0.probe

    def observe(self, generator):
        try:
            return original(self, generator)
        finally:
            finished.set()

    monkeypatch.setattr(Dan_11_0, "probe", observe)
    monkeypatch.setattr(_config.transient, "reportfile", None)
    task = asyncio.create_task(
        adapter.run_garak_probes(CallableModel(predict), [PROBES[0]], prompt_cap=1)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert _config.transient.reportfile is not None
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 10)
        # Wait for the worker's context-manager cleanup, which follows probe().
        for _ in range(100):
            if adapter._report_users == 0:
                break
            await asyncio.sleep(0.01)
    assert sink_seen[0] is not None
    assert _config.transient.reportfile is None


@pytest.mark.parametrize("option", ["prompt_cap", "max_concurrency"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
async def test_invalid_limits_fail_before_running(option, value):
    with pytest.raises(ValueError, match=option):
        await adapter.run_garak_probes(local_model(), **{option: value})


@pytest.mark.parametrize(
    "output,assessed,resistance", [(REFUSAL, 4, 1.0), ("", 0, None)]
)
async def test_assessment_source_coverage_and_strict_json(
    monkeypatch, output, assessed, resistance
):
    from rai_toolkit.assessment.assessor import Assessor
    from rai_toolkit.compliance.frameworks import ComplianceProfile, Framework
    from rai_toolkit.evaluation.pipeline import EvaluationResults
    from rai_toolkit.redteam.runner import AttackRunner, RedTeamReport

    assessor = Assessor(
        model=local_model(output),
        preset="general",
        datasets=["local"],
        extra_redteam_sources=["garak"],
    )
    monkeypatch.setattr(assessor, "_load_datasets", list)

    async def evaluation(*args, **kwargs):
        return EvaluationResults(
            name="local",
            profile=ComplianceProfile(
                name="local",
                framework=Framework.MIT_AI_RISK,
                categories=[],
                industry="general",
            ),
            model_name="local",
            items=[],
            summary={},
            overall_score=1.0,
            overall_passed=True,
            timestamp="2026-09-29T00:00:00+00:00",
        )

    async def builtins(*args, **kwargs):
        return RedTeamReport(
            model_name="local", results=[], by_family={}, total_duration_s=0
        )

    monkeypatch.setattr(assessor, "_run_evaluation", evaluation)
    monkeypatch.setattr(assessor, "_run_policy_checks", lambda evaluation: ([], []))
    monkeypatch.setattr(assessor, "_assess_frameworks", lambda *args: [])
    monkeypatch.setattr(
        "rai_toolkit.assessment.assessor._warn_if_missing_llm_keys",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(AttackRunner, "run_all", builtins)
    result = await assessor.run()
    data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert data["redteam_source_coverage_passed"] is True
    assert data["redteam_source_failures"] == []
    assert data["redteam_summary"]["total_assessed"] == assessed
    assert data["redteam_summary"]["overall_resistance_rate"] == resistance
    assert data["overall_passed"] is (assessed == 4)
