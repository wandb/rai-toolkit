"""OpenAI-compatible model adapter.

Lets the RAI team point the toolkit at any service that speaks the OpenAI
chat-completions protocol — the public OpenAI API, Azure, vLLM, Ollama,
LiteLLM proxies, internal corporate proxies, anything. The reviewer
doesn't have to clone the app team's repo or load a Python class — they
paste a URL, a model name, and an API key.

Use ``rai_toolkit.models.openai_compatible:from_args`` to build one
from a single dict (the Streamlit intake form does this).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import AsyncOpenAI

from rai_toolkit.models.base import BaseModel, ModelResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleModel(BaseModel):
    """A ``BaseModel`` backed by any OpenAI-compatible chat endpoint.

    Args:
        model: Model identifier on the target service (``gpt-4o-mini``,
            ``llama3:8b``, ``meta-llama/Llama-3-8B-Instruct``, etc.).
        base_url: Optional override for the OpenAI client. Set this to
            point at Azure / vLLM / Ollama / LiteLLM. Leave ``None`` for
            the public OpenAI API.
        api_key: API key. Falls back to ``OPENAI_API_KEY`` env var if
            unset. Many local stacks accept any non-empty string.
        system_prompt: Optional system message prepended to every call —
            this is how a triage-assistant or RAG-style app wires its
            system instructions while still being a generic adapter.
        temperature: Default 0 for reproducibility. Override per-call via
            ``predict(..., temperature=...)``.
        name: Display name shown in reports / logs. Defaults to the model
            id.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or model)
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.temperature = temperature

        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        elif not os.environ.get("OPENAI_API_KEY"):
            # Local stacks like Ollama/vLLM don't care; satisfy the SDK.
            client_kwargs["api_key"] = "not-used"
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)

    async def predict(
        self,
        input_text: str,
        context: str = "",
        **kwargs: Any,
    ) -> ModelResponse:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Retrieved context:\n{context}",
                }
            )
        messages.append({"role": "user", "content": input_text})

        completion = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
        )
        choice = completion.choices[0]
        usage = getattr(completion, "usage", None)
        return ModelResponse(
            output=choice.message.content or "",
            metadata={
                "model": self.model,
                "base_url": self.base_url,
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
        )


def from_args(args: dict[str, Any]) -> OpenAICompatibleModel:
    """Build an adapter from a flat dict (used by the Streamlit intake)."""
    return OpenAICompatibleModel(
        model=args["model"],
        base_url=args.get("base_url") or None,
        api_key=args.get("api_key") or None,
        system_prompt=args.get("system_prompt") or None,
        temperature=float(args.get("temperature", 0.0)),
        name=args.get("name") or None,
    )
