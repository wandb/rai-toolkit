# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Reusable conformance suite for :class:`~rai_toolkit.models.base.BaseModel`.

Every model adapter in the toolkit owes callers the same behaviour,
regardless of which provider sits behind it: user input that survives
intact, context forwarded exactly once and never through a privileged
message channel, provider text preserved, the standard metadata keys, one
tracing op, propagated transport failures, and credentials that never
surface in ``repr`` or metadata. The contract is written up in
``docs/model_adapters.md``; this module is its executable form.

Adapter test modules subclass :class:`ModelAdapterContractTests`, name the
subclass ``Test...`` so pytest collects it, and supply a factory that
builds a configured adapter whose transport is mocked. The suite stays
provider-neutral: it never imports a provider SDK and never inspects
provider request types. It only reads the plain dicts the adapter handed
to its transport.

Example::

    class TestMyModelContract(ModelAdapterContractTests):
        adapter_module = my_module
        secret_values = ("secret-test-key",)

        def make_adapter(self):
            return my_module.MyModel(model="m", api_key="secret-test-key")

        def provider_calls(self, adapter):
            return adapter._client.calls

        def set_transport_error(self, adapter, error):
            ...
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

from rai_toolkit.models.base import BaseModel, ModelResponse

TRACING_OP_NAME = "rai.model.predict"

STANDARD_METADATA_KEYS = (
    "model",
    "base_url",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)

INPUT_MARKER = "contract-input-6f1c2a"
CONTEXT_MARKER = "contract-context-9d4b7e"

#: Roles whose content the model is entitled to treat as trusted
#: instructions. Retrieved context must never be routed through one of
#: them, whatever the provider calls it.
PRIVILEGED_ROLES = frozenset({"system", "developer", "tool"})

#: Keys of the JSON object adapters send as the user turn when a caller
#: supplies retrieved context.
CONTEXT_PAYLOAD_KEYS = ("retrieved_context", "input_text")


class ContractTransportError(RuntimeError):
    """Stand-in for a provider transport or authentication failure."""


def iter_strings(value: Any) -> Iterator[str]:
    """Yield every string reachable inside a nested payload.

    Adapters hand their transport plain dicts, lists, and strings. Walking
    them keeps the assertions provider-neutral: the suite never has to know
    whether a provider carries context in a message list, a top-level
    field, or anywhere else.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_strings(key)
            yield from iter_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from iter_strings(item)


def count_occurrences(payload: Any, needle: str) -> int:
    return sum(text.count(needle) for text in iter_strings(payload))


def content_text(content: Any) -> str:
    """Flatten a message body to the text a model would read.

    Providers spell a message body either as a plain string or as a list
    of typed blocks. Joining the strings out of either shape keeps the
    privileged-channel assertions from needing to know which.
    """
    return "".join(iter_strings(content))


class ModelAdapterContractTests:
    """Shared behavioural assertions every adapter must satisfy.

    Subclasses override :meth:`make_adapter`, :meth:`provider_calls`, and
    :meth:`set_transport_error`, and set :attr:`secret_values`. Adapters
    whose SDK ships in a named extra also set :attr:`adapter_module`,
    :attr:`optional_sdk_module`, and :attr:`optional_sdk_extra`; the
    optional-import tests skip themselves for adapters built on a core
    dependency.
    """

    adapter_module: Any = None
    optional_sdk_module: str | None = None
    optional_sdk_extra: str | None = None
    secret_values: tuple[str, ...] = ()
    from_args_invalid_args: ClassVar[dict[str, Any]] = {}

    def make_adapter(self) -> BaseModel:
        """Return a configured adapter instance with a mocked transport."""
        raise NotImplementedError

    def provider_calls(self, adapter: BaseModel) -> list[dict[str, Any]]:
        """Return the request payloads the adapter sent to its transport."""
        raise NotImplementedError

    def set_transport_error(self, adapter: BaseModel, error: Exception) -> None:
        """Make the adapter's next transport call raise ``error``."""
        raise NotImplementedError

    def privileged_texts(self, call: dict[str, Any]) -> list[str]:
        """Return the trusted instruction text in one recorded request.

        The default covers the two shapes the built-in adapters use: a
        top-level ``system`` parameter and messages carrying a privileged
        role. An adapter whose provider spells trusted instructions some
        other way overrides this so the channel still gets checked.
        """
        texts: list[str] = []
        system = call.get("system")
        if system is not None:
            texts.append(content_text(system))
        for message in call.get("messages") or []:
            if not isinstance(message, dict):
                continue
            if message.get("role") in PRIVILEGED_ROLES:
                texts.append(content_text(message.get("content")))
        return texts

    def user_texts(self, call: dict[str, Any]) -> list[str]:
        """Return the user-turn bodies in one recorded request."""
        return [
            content_text(message.get("content"))
            for message in call.get("messages") or []
            if isinstance(message, dict) and message.get("role") == "user"
        ]

    @pytest.fixture
    def adapter(self) -> BaseModel:
        return self.make_adapter()

    def test_adapter_subclasses_base_model(self, adapter: BaseModel) -> None:
        assert isinstance(adapter, BaseModel)
        assert issubclass(type(adapter), BaseModel)

    def test_display_name_is_a_non_empty_string(self, adapter: BaseModel) -> None:
        assert isinstance(adapter.name, str)
        assert adapter.name.strip() != ""

    async def test_display_name_is_stable_across_calls(
        self, adapter: BaseModel
    ) -> None:
        before = adapter.name
        await adapter.predict(INPUT_MARKER)

        assert adapter.name == before

    def test_predict_is_asynchronous(self, adapter: BaseModel) -> None:
        assert inspect.iscoroutinefunction(type(adapter).predict)

    async def test_predict_returns_a_model_response(
        self, adapter: BaseModel
    ) -> None:
        response = await adapter.predict(INPUT_MARKER)

        assert isinstance(response, ModelResponse)
        assert isinstance(response.output, str)
        assert isinstance(response.metadata, dict)

    async def test_predict_sends_user_input_verbatim_exactly_once(
        self, adapter: BaseModel
    ) -> None:
        """With no context, the input is the user turn, untouched."""
        await adapter.predict(INPUT_MARKER)

        (call,) = self.provider_calls(adapter)
        assert count_occurrences(call, INPUT_MARKER) == 1

    async def test_non_empty_context_reaches_the_provider_exactly_once(
        self, adapter: BaseModel
    ) -> None:
        await adapter.predict(INPUT_MARKER, context=CONTEXT_MARKER)

        (call,) = self.provider_calls(adapter)
        assert count_occurrences(call, CONTEXT_MARKER) == 1
        assert count_occurrences(call, INPUT_MARKER) == 1

    async def test_context_never_reaches_a_privileged_channel(
        self, adapter: BaseModel
    ) -> None:
        await adapter.predict(INPUT_MARKER, context=CONTEXT_MARKER)

        (call,) = self.provider_calls(adapter)
        for text in self.privileged_texts(call):
            assert CONTEXT_MARKER not in text

    async def test_context_is_carried_as_json_in_the_user_turn(
        self, adapter: BaseModel
    ) -> None:
        await adapter.predict(INPUT_MARKER, context=CONTEXT_MARKER)

        (call,) = self.provider_calls(adapter)
        (user_text,) = self.user_texts(call)
        payload = json.loads(user_text)

        assert isinstance(payload, dict)
        assert sorted(payload) == sorted(CONTEXT_PAYLOAD_KEYS)
        assert payload["retrieved_context"] == CONTEXT_MARKER
        assert payload["input_text"] == INPUT_MARKER

    async def test_empty_context_sends_input_as_the_bare_user_content(
        self, adapter: BaseModel
    ) -> None:
        await adapter.predict(INPUT_MARKER)

        (call,) = self.provider_calls(adapter)
        assert self.user_texts(call) == [INPUT_MARKER]

    async def test_empty_context_adds_nothing_to_the_request(self) -> None:
        omitted = self.make_adapter()
        await omitted.predict(INPUT_MARKER)
        (without_context,) = self.provider_calls(omitted)

        explicit = self.make_adapter()
        await explicit.predict(INPUT_MARKER, context="")
        (with_empty_context,) = self.provider_calls(explicit)

        assert with_empty_context == without_context
        assert sorted(iter_strings(with_empty_context)) == sorted(
            iter_strings(without_context)
        )

    async def test_output_preserves_provider_text(
        self, adapter: BaseModel
    ) -> None:
        response = await adapter.predict(INPUT_MARKER)

        assert response.output == self.expected_output(adapter)

    def expected_output(self, adapter: BaseModel) -> str:
        """The exact text the mocked transport is expected to return."""
        raise NotImplementedError

    async def test_metadata_exposes_every_standard_key(
        self, adapter: BaseModel
    ) -> None:
        response = await adapter.predict(INPUT_MARKER)

        missing = [
            key for key in STANDARD_METADATA_KEYS if key not in response.metadata
        ]
        assert missing == []

    async def test_metadata_reports_the_configured_model(
        self, adapter: BaseModel
    ) -> None:
        response = await adapter.predict(INPUT_MARKER)

        assert isinstance(response.metadata["model"], str)
        assert response.metadata["model"] != ""

    def test_predict_is_traced_once_under_the_canonical_op_name(
        self, adapter: BaseModel
    ) -> None:
        predict = type(adapter).__dict__["predict"]

        assert getattr(predict, "__rai_traced_op_name__", None) == TRACING_OP_NAME

        inner = getattr(predict, "__wrapped__", None)
        assert inner is not None
        assert getattr(inner, "__rai_traced_op_name__", None) is None

    async def test_transport_failures_propagate(self, adapter: BaseModel) -> None:
        self.set_transport_error(adapter, ContractTransportError("upstream refused"))

        with pytest.raises(ContractTransportError, match="upstream refused"):
            await adapter.predict(INPUT_MARKER)

    def test_repr_never_exposes_credentials(self, adapter: BaseModel) -> None:
        rendered = repr(adapter)

        assert type(adapter).__name__ in rendered
        assert adapter.name in rendered
        for secret in self.secret_values:
            assert secret not in rendered

    async def test_response_metadata_never_exposes_credentials(
        self, adapter: BaseModel
    ) -> None:
        response = await adapter.predict(INPUT_MARKER)

        rendered = list(iter_strings(response.metadata))
        for secret in self.secret_values:
            assert all(secret not in text for text in rendered)

    async def test_transport_error_messages_never_gain_credentials(
        self, adapter: BaseModel
    ) -> None:
        self.set_transport_error(adapter, ContractTransportError("upstream refused"))

        with pytest.raises(ContractTransportError) as exc_info:
            await adapter.predict(INPUT_MARKER)

        for secret in self.secret_values:
            assert secret not in str(exc_info.value)

    def test_from_args_rejects_invalid_configuration_early(self) -> None:
        if self.adapter_module is None:
            pytest.skip("adapter module not supplied")

        with pytest.raises((ValueError, KeyError, TypeError)):
            self.adapter_module.from_args(dict(self.from_args_invalid_args))

    def test_adapter_module_imports_without_the_optional_sdk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if self.optional_sdk_module is None:
            pytest.skip("adapter does not depend on an optional SDK")

        monkeypatch.setitem(sys.modules, self.optional_sdk_module, None)
        importlib.invalidate_caches()
        reloaded = importlib.reload(self.adapter_module)

        assert issubclass(reloaded.BaseModel, BaseModel)

    async def test_missing_optional_sdk_names_the_install_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if self.optional_sdk_module is None:
            pytest.skip("adapter does not depend on an optional SDK")

        adapter = self.make_adapter()
        monkeypatch.setitem(sys.modules, self.optional_sdk_module, None)

        with pytest.raises(ImportError) as exc_info:
            await adapter.predict(INPUT_MARKER)

        message = str(exc_info.value)
        assert self.optional_sdk_module in message
        assert self.optional_sdk_extra is not None
        assert self.optional_sdk_extra in message
