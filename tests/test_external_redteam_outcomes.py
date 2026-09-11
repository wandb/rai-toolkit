# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Outcome translation coverage for optional red-team integrations."""

from __future__ import annotations

import asyncio
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest

from integrations.garak_integration import adapter as garak_adapter
from integrations.pyrit_integration import adapter as pyrit_adapter
from rai_toolkit.redteam.attacks import AttackCategory


class _PyRITOutcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNDETERMINED = "undetermined"
    FUTURE_VALUE = "future_value"


@pytest.mark.parametrize(
    ("outcome", "expected_succeeded", "expected_error"),
    [
        (_PyRITOutcome.SUCCESS, True, None),
        (_PyRITOutcome.FAILURE, False, None),
        (_PyRITOutcome.UNDETERMINED, False, "undetermined"),
        (_PyRITOutcome.FUTURE_VALUE, False, "unsupported"),
        (None, False, "no attack outcome"),
    ],
)
def test_pyrit_outcomes_preserve_unassessed_states(
    outcome: Any,
    expected_succeeded: bool,
    expected_error: str | None,
) -> None:
    succeeded, error = pyrit_adapter._translate_outcome(outcome)

    assert succeeded is expected_succeeded
    if expected_error is None:
        assert error is None
    else:
        assert error is not None
        assert expected_error in error.lower()


def test_pyrit_runner_marks_undetermined_result_unassessed() -> None:
    class _FakeAttack:
        async def execute_async(self, *, objective: str) -> Any:
            return SimpleNamespace(
                outcome=_PyRITOutcome.UNDETERMINED,
                last_response=SimpleNamespace(
                    converted_value="partial output",
                    original_value="",
                ),
            )

    factory = pyrit_adapter.PyRITAttackFactory(
        attack_id="pyrit-test",
        category=AttackCategory.JAILBREAK,
        severity=4,
        objective="test objective",
        factory=lambda target: _FakeAttack(),
    )

    result = asyncio.run(pyrit_adapter._run_one_factory(object(), factory))

    assert result.succeeded is False
    assert result.assessed is False
    assert result.error is not None
    assert "undetermined" in result.error.lower()
    assert result.model_output == "partial output"


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        ([], None),
        ([None, "", "   "], None),
        (["I can't help with that request."], False),
        (["Step 1: bypass the safeguard."], True),
        (
            [
                "I can't help with that request.",
                "Step 1: bypass the safeguard.",
            ],
            True,
        ),
        (
            [
                "Step 1: bypass the safeguard.",
                "I can't help with that request.",
            ],
            True,
        ),
    ],
)
def test_garak_uses_worst_meaningful_output(
    outputs: list[str | None],
    expected: bool | None,
) -> None:
    assert garak_adapter._attempt_succeeded(outputs) is expected


@pytest.mark.parametrize(
    "attempts",
    [
        [],
        [SimpleNamespace(outputs=[])],
        [SimpleNamespace(outputs=[SimpleNamespace(text=None)])],
        [SimpleNamespace(outputs=[SimpleNamespace(text="   ")])],
    ],
)
def test_garak_runner_marks_missing_outputs_unassessed(
    attempts: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProbe:
        def probe(self, generator: Any) -> list[Any]:
            return attempts

    monkeypatch.setattr(garak_adapter, "RAIGenerator", lambda model: object())
    monkeypatch.setattr(
        garak_adapter,
        "_instantiate_probe",
        lambda probe_path, prompt_cap: _FakeProbe(),
    )
    spec = garak_adapter.GarakProbeSpec(
        attack_id="garak-test",
        probe_path="fake.Probe",
        category=AttackCategory.JAILBREAK,
        severity=4,
        objective="test objective",
    )

    result = asyncio.run(
        garak_adapter._run_one_probe(object(), spec, prompt_cap=1)
    )

    assert result.succeeded is False
    assert result.assessed is False
    assert result.error == "Garak probe produced no model output."


def test_garak_runner_preserves_the_output_that_makes_the_attack_succeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harmful = "Step 1: bypass the safeguard."

    class _FakeProbe:
        def probe(self, generator: Any) -> list[Any]:
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text=harmful),
                        SimpleNamespace(text="I can't help with that request."),
                    ]
                )
            ]

    monkeypatch.setattr(garak_adapter, "RAIGenerator", lambda model: object())
    monkeypatch.setattr(
        garak_adapter,
        "_instantiate_probe",
        lambda probe_path, prompt_cap: _FakeProbe(),
    )
    spec = garak_adapter.GarakProbeSpec(
        attack_id="garak-test",
        probe_path="fake.Probe",
        category=AttackCategory.JAILBREAK,
        severity=4,
        objective="test objective",
    )

    result = asyncio.run(
        garak_adapter._run_one_probe(object(), spec, prompt_cap=1)
    )

    assert result.succeeded is True
    assert result.assessed is True
    assert result.error is None
    assert result.model_output == harmful
