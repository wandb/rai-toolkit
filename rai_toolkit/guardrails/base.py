# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Base guardrail interface — platform-agnostic guardrail abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailResult:
    """Result from a guardrail check.

    Attributes:
        allowed: Whether the content passed the guardrail.
        modified_content: Optionally modified content (if guardrail rewrites).
        triggered_rules: List of rule names that fired.
        explanation: Human-readable explanation of why content was blocked/modified.
        details: Additional metadata.
    """

    allowed: bool
    modified_content: str | None = None
    triggered_rules: list[str] = field(default_factory=list)
    explanation: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class BaseGuardrail(ABC):
    """Abstract guardrail interface. Subclass for custom guardrail implementations.

    Guardrails check inputs and outputs for safety, compliance, or policy violations.
    They can block, modify, or flag content.

    Example::

        class ProfanityGuardrail(BaseGuardrail):
            name = "profanity_filter"

            async def check_input(self, user_input, **kwargs):
                has_profanity = contains_profanity(user_input)
                return GuardrailResult(
                    allowed=not has_profanity,
                    explanation="Profanity detected" if has_profanity else "",
                )

            async def check_output(self, output, user_input="", **kwargs):
                has_profanity = contains_profanity(output)
                return GuardrailResult(
                    allowed=not has_profanity,
                    explanation="Profanity in response" if has_profanity else "",
                )
    """

    name: str = ""

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        if not self.name:
            self.name = self.__class__.__name__

    @abstractmethod
    async def check_input(
        self,
        user_input: str,
        **kwargs: Any,
    ) -> GuardrailResult:
        """Check user input before it reaches the model.

        Returns:
            GuardrailResult indicating if the input is allowed.
        """
        ...

    @abstractmethod
    async def check_output(
        self,
        output: str,
        user_input: str = "",
        **kwargs: Any,
    ) -> GuardrailResult:
        """Check model output before it reaches the user.

        Returns:
            GuardrailResult indicating if the output is allowed.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
