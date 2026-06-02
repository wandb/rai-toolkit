# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Weave model adapter — wraps rai_toolkit BaseModel as weave.Model."""

from __future__ import annotations

from typing import Any

import weave

from rai_toolkit import _tracing
from rai_toolkit.models.base import BaseModel, ModelResponse


class WeaveModel(weave.Model):
    """Adapts any rai_toolkit BaseModel to a Weave-tracked model.

    Automatically versions model configuration and traces all predictions.

    Example::

        from integrations.weave_integration.models import WeaveModel

        weave_model = WeaveModel(
            rai_model=my_openai_model,
            model_name="gpt-4-turbo",
            system_prompt="You are a helpful assistant",
            temperature=0.7,
        )
        result = await weave_model.predict("What is AI?")
    """

    model_name: str = ""
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000

    _rai_model: BaseModel | None = None

    def __init__(self, rai_model: BaseModel | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._rai_model = rai_model

    @weave.op()
    async def predict(
        self,
        input_text: str,
        context: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run prediction through the wrapped model with Weave tracing.

        Returns a dict (Weave-friendly) rather than ModelResponse.
        """
        if self._rai_model is None:
            raise ValueError("No rai_model provided to WeaveModel")

        response = await self._rai_model.predict(
            input_text=input_text, context=context, **kwargs
        )

        payload = {
            "output": response.output,
            "model": self.model_name or self._rai_model.name,
            **response.metadata,
        }
        call_url = _tracing.current_call_url()
        if call_url:
            payload["weave_call_url"] = call_url
        return payload


class WeaveOpenAIModel(weave.Model):
    """Direct OpenAI model with Weave tracking. No rai_toolkit dependency.

    Use this when you don't need the BaseModel abstraction — just a
    quick Weave-traced OpenAI wrapper for demos.
    """

    model_name: str = "gpt-4o"
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7
    max_tokens: int = 2000

    @weave.op()
    async def predict(
        self,
        input_text: str,
        context: str = "",
    ) -> dict[str, Any]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.append({
                "role": "system",
                "content": f"Context:\n{context}",
            })

        messages.append({"role": "user", "content": input_text})

        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        usage = response.usage
        return {
            "output": response.choices[0].message.content or "",
            "model": self.model_name,
            "tokens": {
                "prompt": usage.prompt_tokens if usage else 0,
                "completion": usage.completion_tokens if usage else 0,
                "total": usage.total_tokens if usage else 0,
            },
        }
