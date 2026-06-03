# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""NeMo Guardrails implementation of the BaseGuardrail interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rai_toolkit import _tracing
from rai_toolkit.guardrails.base import BaseGuardrail, GuardrailResult

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = str(
    Path(__file__).parent / "colang_configs" / "rai_config"
)


class NeMoGuardrail(BaseGuardrail):
    """NVIDIA NeMo Guardrails as a rai_toolkit guardrail.

    Wraps NeMo's programmable guardrails with the BaseGuardrail interface
    so it can be used with GuardedModel.

    Example::

        from integrations.nemo_integration import NeMoGuardrail

        guardrail = NeMoGuardrail()  # Uses default RAI config
        result = await guardrail.check_input("How do I hack a system?")
        print(result.allowed)  # False

        # With custom config
        guardrail = NeMoGuardrail(config_path="/path/to/my/config")
    """

    name = "NeMoGuardrail"

    def __init__(
        self,
        config_path: str | None = None,
        model: str = "openai/gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.model = model
        self._rails = None

    def _get_rails(self) -> Any:
        """Lazy-initialize NeMo Guardrails."""
        if self._rails is None:
            try:
                from nemoguardrails import RailsConfig, LLMRails

                config = RailsConfig.from_path(self.config_path)
                self._rails = LLMRails(config)
                logger.info("NeMo Guardrails initialized from %s", self.config_path)
            except ImportError:
                raise ImportError(
                    "NeMo Guardrails not installed. "
                    "Install with: pip install rai-toolkit[nemo]"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize NeMo Guardrails: {e}")
        return self._rails

    @_tracing.traced(name="rai.guardrail.nemo.input", kind="guardrail")
    async def check_input(
        self,
        user_input: str,
        **kwargs: Any,
    ) -> GuardrailResult:
        """Check user input through NeMo input rails.

        Args:
            user_input: The user's message.

        Returns:
            GuardrailResult indicating if the input is allowed.
        """
        try:
            rails = self._get_rails()
            response = await rails.generate_async(
                messages=[{"role": "user", "content": user_input}]
            )

            # NeMo returns the response; if it's a refusal, the input was blocked
            response_text = self._response_text(response)
            blocked = self._is_refusal(response_text)

            return GuardrailResult(
                allowed=not blocked,
                modified_content=None if blocked else user_input,
                triggered_rules=["input_rail"] if blocked else [],
                explanation=response_text if blocked else "",
                details={"nemo_response": response_text},
            )
        except Exception as e:
            logger.error("NeMo input check failed: %s", e)
            # Fail open — allow the input but log the error
            return GuardrailResult(
                allowed=True,
                explanation=f"NeMo check error: {e}",
                details={"error": str(e)},
            )

    @_tracing.traced(name="rai.guardrail.nemo.output", kind="guardrail")
    async def check_output(
        self,
        output: str,
        user_input: str = "",
        **kwargs: Any,
    ) -> GuardrailResult:
        """Check model output through NeMo output rails.

        Args:
            output: The model's response.
            user_input: The original user input.

        Returns:
            GuardrailResult indicating if the output is allowed.
        """
        try:
            rails = self._get_rails()

            messages = []
            if user_input:
                messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": output})

            # NeMo processes the full conversation and may modify the response
            response = await rails.generate_async(messages=messages)
            response_text = self._response_text(response)

            # NeMo returns an empty content payload for some pass-through
            # output-rail checks. Treat that as "no rewrite" so the guardrail
            # never erases the model output in WeaveModel.predict.
            has_replacement = bool(response_text.strip())
            modified = has_replacement and response_text != output
            blocked = self._is_refusal(response_text)

            return GuardrailResult(
                allowed=not blocked,
                modified_content=response_text if modified and not blocked else None,
                triggered_rules=["output_rail"] if modified or blocked else [],
                explanation=f"Output modified by NeMo" if modified else "",
                details={
                    "original_output": output,
                    "nemo_output": response_text,
                    "was_modified": modified,
                    "empty_nemo_output": not has_replacement,
                },
            )
        except Exception as e:
            logger.error("NeMo output check failed: %s", e)
            return GuardrailResult(
                allowed=True,
                explanation=f"NeMo check error: {e}",
                details={"error": str(e)},
            )

    @staticmethod
    def _is_refusal(response: str) -> bool:
        """Heuristic to detect if NeMo generated a refusal response."""
        refusal_indicators = [
            "i cannot",
            "i can't",
            "i'm not able",
            "i am not able",
            "i apologize",
            "not allowed",
            "against my guidelines",
            "i'm unable",
            "i must decline",
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in refusal_indicators)

    @staticmethod
    def _response_text(response: Any) -> str:
        """Extract text from the shapes returned by NeMo rails."""
        if response is None:
            return ""
        if isinstance(response, dict):
            content = response.get("content")
            return "" if content is None else str(content)
        return str(response)
