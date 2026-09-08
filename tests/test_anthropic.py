# SPDX-FileCopyrightText: 2026 schallten
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

import rai_toolkit.models as models_pkg
from rai_toolkit.models import AnthropicModel
from rai_toolkit.models import anthropic as anthropic_module

anthropic = pytest.importorskip("anthropic")


class FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="offline answer")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=7, output_tokens=3),
        )


class FakeAsyncAnthropic:
    instances: ClassVar[list[FakeAsyncAnthropic]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.messages = FakeMessages()
        self.instances.append(self)


@pytest.fixture(autouse=True)
def fake_anthropic_client(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAsyncAnthropic.instances.clear()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAsyncAnthropic)


def latest_client() -> FakeAsyncAnthropic:
    assert len(FakeAsyncAnthropic.instances) == 1
    return FakeAsyncAnthropic.instances[0]


def test_client_receives_explicit_credentials() -> None:
    model = AnthropicModel(
        model="claude-sonnet-4-5",
        base_url="https://proxy.example/v1",
        api_key="secret-test-key",
    )

    assert model.name == "claude-sonnet-4-5"
    assert model.client.kwargs == {
        "api_key": "secret-test-key",
        "base_url": "https://proxy.example/v1",
    }


def test_client_leaves_key_resolution_to_anthropic_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-key")

    model = AnthropicModel(model="claude-sonnet-4-5")

    assert model.client.kwargs == {}


def test_default_max_tokens_is_documented_and_positive() -> None:
    model = AnthropicModel(model="claude-sonnet-4-5")

    assert model.max_tokens == anthropic_module.DEFAULT_MAX_TOKENS
    assert model.max_tokens > 0


@pytest.mark.parametrize("bad", [0, -3, True, 1.5, 2.0, "abc", "1.5"])
def test_constructor_rejects_invalid_max_tokens(bad: Any) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicModel(model="claude-sonnet-4-5", max_tokens=bad)


@pytest.mark.parametrize("value", [None, ""])
def test_constructor_uses_documented_default_for_none_or_blank(value: Any) -> None:
    model = AnthropicModel(model="claude-sonnet-4-5", max_tokens=value)

    assert model.max_tokens == anthropic_module.DEFAULT_MAX_TOKENS


async def test_predict_builds_system_and_context_through_top_level_system() -> None:
    model = AnthropicModel(
        model="claude-sonnet-4-5",
        base_url="https://proxy.example/v1",
        system_prompt="Follow the policy.",
        max_tokens=512,
        name="Claude reviewer",
    )

    response = await model.predict("Review this answer.", context="Policy text")

    assert model.name == "Claude reviewer"
    (call,) = latest_client().messages.calls
    assert call["model"] == "claude-sonnet-4-5"
    assert call["max_tokens"] == 512
    assert call["system"] == "Follow the policy.\n\nRetrieved context:\nPolicy text"
    assert call["messages"] == [{"role": "user", "content": "Review this answer."}]
    assert response.output == "offline answer"


async def test_predict_omits_system_param_when_unset() -> None:
    model = AnthropicModel(model="claude-sonnet-4-5")

    await model.predict("Review this answer.")

    (call,) = latest_client().messages.calls
    assert "system" not in call
    assert call["max_tokens"] == anthropic_module.DEFAULT_MAX_TOKENS


async def test_predict_overrides_max_tokens_per_call() -> None:
    model = AnthropicModel(model="claude-sonnet-4-5", max_tokens=128)

    await model.predict("Review this answer.", max_tokens=2048)

    (call,) = latest_client().messages.calls
    assert call["max_tokens"] == 2048


@pytest.mark.parametrize("bad", [0, -3, True, 1.5, "abc", "1.5"])
async def test_predict_rejects_invalid_per_call_max_tokens(bad: Any) -> None:
    model = AnthropicModel(model="claude-sonnet-4-5")

    with pytest.raises(ValueError, match="max_tokens"):
        await model.predict("Review this answer.", max_tokens=bad)


async def test_predict_uses_documented_default_for_none_override() -> None:
    model = AnthropicModel(model="claude-sonnet-4-5", max_tokens=512)

    await model.predict("Review this answer.", max_tokens=None)

    (call,) = latest_client().messages.calls
    assert call["max_tokens"] == anthropic_module.DEFAULT_MAX_TOKENS


async def test_predict_concatenates_text_blocks_without_invented_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def multi_block_create(self: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="first"),
                SimpleNamespace(type="tool_use", text="ignored"),
                SimpleNamespace(type="text", text="second"),
            ],
            stop_reason="max_tokens",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )

    monkeypatch.setattr(FakeMessages, "create", multi_block_create)

    model = AnthropicModel(model="claude-sonnet-4-5")
    response = await model.predict("Review this answer.")

    assert response.output == "firstsecond"


async def test_predict_normalizes_metadata() -> None:
    model = AnthropicModel(
        model="claude-sonnet-4-5",
        base_url="https://proxy.example/v1",
    )

    response = await model.predict("Review this answer.")

    assert response.metadata == {
        "model": "claude-sonnet-4-5",
        "base_url": "https://proxy.example/v1",
        "finish_reason": "end_turn",
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


async def test_predict_handles_missing_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_usage_create(self: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="offline answer")],
            stop_reason="end_turn",
            usage=None,
        )

    monkeypatch.setattr(FakeMessages, "create", no_usage_create)

    model = AnthropicModel(model="claude-sonnet-4-5")
    response = await model.predict("Review this answer.")

    assert response.output == "offline answer"
    assert response.metadata["prompt_tokens"] is None
    assert response.metadata["completion_tokens"] is None
    assert response.metadata["total_tokens"] is None


def test_from_args_requires_model() -> None:
    with pytest.raises(ValueError, match="model"):
        anthropic_module.from_args({})


def test_from_args_converts_and_validates() -> None:
    model = anthropic_module.from_args(
        {
            "model": "claude-sonnet-4-5",
            "base_url": "https://proxy.example/v1",
            "api_key": "configured-key",
            "system_prompt": "Be concise.",
            "max_tokens": "4000",
            "name": "Claude reviewer",
        }
    )

    assert model.name == "Claude reviewer"
    assert model.max_tokens == 4000
    assert model.client.kwargs == {
        "api_key": "configured-key",
        "base_url": "https://proxy.example/v1",
    }


def test_from_args_treats_empty_optionals_as_unset() -> None:
    model = anthropic_module.from_args(
        {
            "model": "claude-sonnet-4-5",
            "base_url": "",
            "api_key": "",
            "system_prompt": "",
            "max_tokens": "1024",
            "name": "",
        }
    )

    assert model.name == "claude-sonnet-4-5"
    assert model.base_url is None
    assert model.system_prompt is None
    assert model.client.kwargs == {}


def test_from_args_uses_documented_default_for_none_or_blank() -> None:
    for value in [None, "", "   "]:
        model = anthropic_module.from_args(
            {"model": "claude-sonnet-4-5", "max_tokens": value}
        )

        assert model.max_tokens == anthropic_module.DEFAULT_MAX_TOKENS


@pytest.mark.parametrize("bad", [True, 1.5, "1.5", "abc", 0, "0"])
def test_from_args_rejects_non_integer_max_tokens(bad: Any) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        anthropic_module.from_args({"model": "claude-sonnet-4-5", "max_tokens": bad})


def test_anthropic_model_exported_from_models_package() -> None:
    assert issubclass(AnthropicModel, anthropic_module.BaseModel)
    assert "AnthropicModel" in models_pkg.__all__


def test_predict_is_traced_once_as_rai_model_predict() -> None:
    assert AnthropicModel.predict.__rai_traced_op_name__ == "rai.model.predict"  # type: ignore[attr-defined]


def test_package_imports_work_without_anthropic_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)

    importlib.invalidate_caches()
    reloaded = importlib.reload(anthropic_module)
    assert issubclass(reloaded.AnthropicModel, reloaded.BaseModel)


async def test_predict_raises_helpful_error_without_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)

    model = AnthropicModel(model="claude-sonnet-4-5")

    with pytest.raises(ImportError, match="anthropic"):
        await model.predict("Review this answer.")