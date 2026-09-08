# SPDX-FileCopyrightText: 2026 schallten
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Anthropic Messages API model adapter.

Wraps Anthropic's asynchronous Python client behind the toolkit's vendor-
neutral :class:`~rai_toolkit.models.base.BaseModel`. The ``anthropic``
package is an optional dependency (the ``[anthropic]`` extra) and is
imported lazily so that ``rai_toolkit`` and ``rai_toolkit.models`` keep
working in environments that don't have it installed.

Build one directly or via :func:`from_args`, a configuration helper for
callers that work from flat dicts or form payloads.
"""

from __future__ import annotations

import logging
from typing import Any

from rai_toolkit.models.base import BaseModel, ModelResponse

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 1024

_ANTHROPIC_IMPORT_ERROR = (
    "The 'anthropic' package is required to use AnthropicModel. "
    "Install it with: pip install -e '.[anthropic]'"
)


def _coerce_max_tokens(value: Any) -> int:
    """Normalize ``max_tokens`` across every entry point.

    ``None`` and blank strings fall back to :data:`DEFAULT_MAX_TOKENS`.
    Anything else must be a strict positive integer: booleans, floats,
    and non-integer strings are rejected.
    """
    if value is None:
        return DEFAULT_MAX_TOKENS
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return DEFAULT_MAX_TOKENS
        if not stripped.lstrip("+-").isdigit():
            raise ValueError(f"max_tokens must be a positive integer, got {value!r}")
        value = int(stripped)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"max_tokens must be a positive integer, got {value!r}")
    return value


class AnthropicModel(BaseModel):
    """A ``BaseModel`` backed by Anthropic's Messages API.

    Args:
        model: Anthropic model identifier (``claude-sonnet-4-5``, etc.).
        api_key: API key. When omitted, resolution is left to the Anthropic
            SDK (``ANTHROPIC_API_KEY`` env var). No fallback key is
            invented.
        base_url: Optional override for the Anthropic client (proxies /
            compatible gateways). Leave ``None`` for the Anthropic API.
        system_prompt: Optional system message prepended to every call.
            This is how a triage-assistant or RAG-style app wires its
            system instructions while still being a generic adapter.
        max_tokens: Default maximum number of tokens to generate. Must be
            a strict positive integer; ``None`` resets it to the
            documented default. Override per-call via
            ``predict(..., max_tokens=...)``.
        name: Display name shown in reports / logs. Defaults to the model
            id.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = DEFAULT_MAX_TOKENS,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or model)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.max_tokens = _coerce_max_tokens(max_tokens)
        self._client: Any = None

    @property
    def client(self) -> Any:
        """The lazily-created ``AsyncAnthropic`` client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - absent only without the extra
                raise ImportError(_ANTHROPIC_IMPORT_ERROR) from exc

            client_kwargs: dict[str, Any] = {}
            if self.api_key is not None:
                client_kwargs["api_key"] = self.api_key
            if self.base_url is not None:
                client_kwargs["base_url"] = self.base_url
            self._client = AsyncAnthropic(**client_kwargs)
        return self._client

    async def predict(
        self,
        input_text: str,
        context: str = "",
        **kwargs: Any,
    ) -> ModelResponse:
        """Run inference against Anthropic's Messages API.

        Args:
            input_text: The user input or query. Sent as the user message.
            context: Optional retrieved context (for RAG systems). Sent
                through the top-level ``system`` parameter alongside the
                system prompt.
            **kwargs: ``max_tokens`` overrides the adapter default for this
                call. Must be positive when provided.

        Returns:
            ModelResponse with all text blocks concatenated into ``output``
            and normalized metadata.
        """
        max_tokens = _coerce_max_tokens(kwargs.get("max_tokens", self.max_tokens))

        system_parts: list[str] = []
        if self.system_prompt:
            system_parts.append(self.system_prompt)
        if context:
            system_parts.append(f"Retrieved context:\n{context}")

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": input_text}],
        }
        if system_parts:
            params["system"] = "\n\n".join(system_parts)

        response = await self.client.messages.create(**params)

        text_blocks = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None)
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        else:
            total_tokens = None

        return ModelResponse(
            output="".join(text_blocks),
            metadata={
                "model": self.model,
                "base_url": self.base_url,
                "finish_reason": getattr(response, "stop_reason", None),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )


def from_args(args: dict[str, Any]) -> AnthropicModel:
    """Build an adapter from a flat configuration dict.

    Requires a ``model`` key. Optional keys (``api_key``, ``base_url``,
    ``system_prompt``, ``max_tokens``, ``name``) are treated as unset when
    missing or empty. ``max_tokens`` accepts an int or an integer-literal
    string; ``None`` and blank values fall back to
    :data:`DEFAULT_MAX_TOKENS`.
    """
    if not args.get("model"):
        raise ValueError("AnthropicModel requires a 'model' argument")
    return AnthropicModel(
        model=args["model"],
        api_key=args.get("api_key") or None,
        base_url=args.get("base_url") or None,
        system_prompt=args.get("system_prompt") or None,
        max_tokens=_coerce_max_tokens(args.get("max_tokens")),
        name=args.get("name") or None,
    )