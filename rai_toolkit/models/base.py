# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Base model interface: platform-agnostic model abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    """Standardized model response.

    Attributes:
        output: The model's text response.
        metadata: Additional info (token counts, latency, cost, model name, etc.).
    """

    output: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract model interface. Subclass this to wrap any LLM provider.

    Example::

        class MyOpenAIModel(BaseModel):
            name = "gpt-4-turbo"

            def __init__(self, api_key: str):
                super().__init__()
                self.client = openai.OpenAI(api_key=api_key)

            async def predict(self, input_text, context="", **kwargs):
                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=[{"role": "user", "content": input_text}],
                )
                return ModelResponse(
                    output=response.choices[0].message.content,
                    metadata={"model": self.name, "tokens": response.usage.total_tokens},
                )
    """

    name: str = ""

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        if not self.name:
            self.name = self.__class__.__name__

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-wrap each subclass's ``predict`` with the toolkit tracing op.

        Without this, only ``BaseModel`` subclasses that opted into
        ``@_tracing.traced`` (e.g. ``GuardedModel``) emitted a Weave span,
        and their span output was a flat dict. Plain subclasses like the
        demo RAG apps emitted no span of their own. The ``ModelResponse``
        dataclass surfaced only as the output of whatever parent op called
        them (eval pipeline, chat probe, red-team), where Weave's
        dataclass serializer nests fields under a ``result`` wrapper.

        Wrapping here means every subclass's ``predict`` shows up under a
        single op name (``rai.model.predict``) and a single registered
        ``postprocess_output`` flattens its return value to the same shape
        ``WeaveModel.predict`` already emits. Downstream automation can
        rely on one trace shape regardless of how the call was made.

        Skips:
        - subclasses that already wrapped ``predict`` with
          ``_tracing.traced`` (detected via the ``__rai_traced_op_name__``
          sentinel), keeps their custom op name (e.g.
          ``rai.guardrails.predict``) intact.
        - subclasses that don't override ``predict`` (still abstract).
        """
        super().__init_subclass__(**kwargs)
        own_predict = cls.__dict__.get("predict")
        if own_predict is None:
            return
        if getattr(own_predict, "__isabstractmethod__", False):
            return
        if getattr(own_predict, "__rai_traced_op_name__", None):
            return

        from rai_toolkit import _tracing

        cls.predict = _tracing.traced(  # type: ignore[method-assign]
            name="rai.model.predict",
            kind="llm",
        )(own_predict)

    @abstractmethod
    async def predict(
        self,
        input_text: str,
        context: str = "",
        **kwargs: Any,
    ) -> ModelResponse:
        """Run inference. Override this method.

        Args:
            input_text: The user input or query.
            context: Optional retrieved context (for RAG systems).
            **kwargs: Additional model-specific arguments.

        Returns:
            ModelResponse with output text and metadata.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
