# SPDX-FileCopyrightText: 2026 Jhye
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from rai_toolkit.models import openai_compatible


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="offline answer"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=7,
                completion_tokens=3,
                total_tokens=10,
            ),
        )


class FakeAsyncOpenAI:
    instances: ClassVar[list[FakeAsyncOpenAI]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.instances.append(self)


@pytest.fixture(autouse=True)
def fake_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAsyncOpenAI.instances.clear()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(openai_compatible, "AsyncOpenAI", FakeAsyncOpenAI)


def latest_client() -> FakeAsyncOpenAI:
    assert len(FakeAsyncOpenAI.instances) == 1
    return FakeAsyncOpenAI.instances[0]


def test_client_receives_explicit_endpoint_credentials() -> None:
    model = openai_compatible.OpenAICompatibleModel(
        model="local-model",
        base_url="http://localhost:8000/v1",
        api_key="secret-test-key",
    )

    assert model.name == "local-model"
    assert latest_client().kwargs == {
        "api_key": "secret-test-key",
        "base_url": "http://localhost:8000/v1",
    }


def test_client_uses_non_empty_fallback_key_for_local_endpoint() -> None:
    openai_compatible.OpenAICompatibleModel(
        model="local-model",
        base_url="http://localhost:8000/v1",
    )

    assert latest_client().kwargs == {
        "api_key": "not-used",
        "base_url": "http://localhost:8000/v1",
    }


def test_client_leaves_environment_key_to_openai_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")

    openai_compatible.OpenAICompatibleModel(model="hosted-model")

    assert latest_client().kwargs == {}


async def test_predict_preserves_message_order_and_response_metadata() -> None:
    model = openai_compatible.OpenAICompatibleModel(
        model="local-model",
        base_url="http://localhost:8000/v1",
        system_prompt="Follow the policy.",
        temperature=0.25,
        name="Local reviewer",
    )

    response = await model.predict("Review this answer.", context="Policy text")

    assert model.name == "Local reviewer"
    assert latest_client().completions.calls == [
        {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": "Follow the policy."},
                {
                    "role": "system",
                    "content": "Retrieved context:\nPolicy text",
                },
                {"role": "user", "content": "Review this answer."},
            ],
            "temperature": 0.25,
        }
    ]
    assert response.output == "offline answer"
    assert response.metadata == {
        "model": "local-model",
        "base_url": "http://localhost:8000/v1",
        "finish_reason": "stop",
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


async def test_predict_accepts_per_call_temperature() -> None:
    model = openai_compatible.OpenAICompatibleModel(
        model="local-model",
        temperature=0.25,
    )

    await model.predict("Review this answer.", temperature=0.8)

    assert latest_client().completions.calls[0]["temperature"] == 0.8


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            {"model": "local-model"},
            {
                "base_url": None,
                "system_prompt": None,
                "temperature": 0.0,
                "name": "local-model",
                "client_kwargs": {"api_key": "not-used"},
            },
        ),
        (
            {
                "model": "local-model",
                "base_url": "",
                "api_key": "",
                "system_prompt": "",
                "temperature": "0.35",
                "name": "",
            },
            {
                "base_url": None,
                "system_prompt": None,
                "temperature": 0.35,
                "name": "local-model",
                "client_kwargs": {"api_key": "not-used"},
            },
        ),
        (
            {
                "model": "local-model",
                "base_url": "https://models.example/v1",
                "api_key": "configured-key",
                "system_prompt": "Be concise.",
                "temperature": 1,
                "name": "Review model",
            },
            {
                "base_url": "https://models.example/v1",
                "system_prompt": "Be concise.",
                "temperature": 1.0,
                "name": "Review model",
                "client_kwargs": {
                    "api_key": "configured-key",
                    "base_url": "https://models.example/v1",
                },
            },
        ),
    ],
)
def test_from_args_normalizes_optional_values_and_temperature(
    args: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    model = openai_compatible.from_args(args)

    assert model.base_url == expected["base_url"]
    assert model.system_prompt == expected["system_prompt"]
    assert model.temperature == expected["temperature"]
    assert model.name == expected["name"]
    assert latest_client().kwargs == expected["client_kwargs"]
