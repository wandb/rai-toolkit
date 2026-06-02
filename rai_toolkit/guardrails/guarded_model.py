# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""GuardedModel — wraps any model with guardrails and scorer checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rai_toolkit import _tracing
from rai_toolkit.guardrails.base import BaseGuardrail, GuardrailResult
from rai_toolkit.models.base import BaseModel, ModelResponse
from rai_toolkit.scorers.base import BaseScorer, ScorerResult

logger = logging.getLogger(__name__)


def _guarded_predict_display_name(call: Any) -> str:
    """Label guarded model spans with the wrapped model name."""
    try:
        guarded = (call.inputs or {}).get("self")
        model_name = getattr(getattr(guarded, "model", None), "name", None)
        return f"guarded.predict[{model_name}]" if model_name else "guarded.predict"
    except Exception:  # pragma: no cover - display names must never break tracing
        return "guarded.predict"


@dataclass
class GuardedResponse:
    """Response from a guarded model prediction.

    Includes the model output plus all guardrail and scorer results.
    """

    output: str
    blocked: bool = False
    blocked_by: str = ""
    input_guardrail_results: list[GuardrailResult] = field(default_factory=list)
    output_guardrail_results: list[GuardrailResult] = field(default_factory=list)
    scorer_results: dict[str, ScorerResult] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)


class GuardedModel(BaseModel):
    """Wraps any BaseModel with guardrails and scorer checks.

    Pipeline: Input Guardrails -> Model.predict() -> Output Guardrails -> Scorer Checks

    If any guardrail blocks, the pipeline short-circuits with a safe response.
    Scorer checks run on the output but don't block by default (they flag).

    Example::

        from rai_toolkit.guardrails import GuardedModel
        from integrations.nemo_integration import NeMoGuardrail

        guarded = GuardedModel(
            model=my_model,
            input_guardrails=[NeMoGuardrail()],
            output_guardrails=[NeMoGuardrail()],
            output_scorers=[RegexPIIScorer(), KeywordToxicityScorer()],
            block_on_scorer_fail=True,  # Also block if scorers fail
        )

        response = await guarded.predict("Tell me about the drug")
    """

    def __init__(
        self,
        model: BaseModel,
        input_guardrails: list[BaseGuardrail] | None = None,
        output_guardrails: list[BaseGuardrail] | None = None,
        output_scorers: list[BaseScorer] | None = None,
        block_on_scorer_fail: bool = False,
        blocked_response: str = "I'm unable to process this request due to safety constraints.",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or f"Guarded({model.name})")
        self.model = model
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.output_scorers = output_scorers or []
        self.block_on_scorer_fail = block_on_scorer_fail
        self.blocked_response = blocked_response

    @_tracing.traced(
        name="rai.guardrails.predict",
        kind="guardrail",
        call_display_name=lambda call: _guarded_predict_display_name(call),
    )
    async def predict(
        self,
        input_text: str,
        context: str = "",
        **kwargs: Any,
    ) -> ModelResponse:
        """Run the guarded prediction pipeline.

        Returns a standard ModelResponse. The metadata field contains
        the full GuardedResponse with guardrail and scorer details.
        """
        guarded = await self.predict_guarded(input_text, context, **kwargs)
        return ModelResponse(
            output=guarded.output,
            metadata={
                "blocked": guarded.blocked,
                "blocked_by": guarded.blocked_by,
                "input_guardrails": [
                    {
                        "allowed": r.allowed,
                        "triggered_rules": r.triggered_rules,
                        "explanation": r.explanation,
                    }
                    for r in guarded.input_guardrail_results
                ],
                "output_guardrails": [
                    {
                        "allowed": r.allowed,
                        "triggered_rules": r.triggered_rules,
                        "explanation": r.explanation,
                    }
                    for r in guarded.output_guardrail_results
                ],
                "scorer_results": {
                    name: {"score": r.score, "passed": r.passed}
                    for name, r in guarded.scorer_results.items()
                },
            },
        )

    async def predict_guarded(
        self,
        input_text: str,
        context: str = "",
        **kwargs: Any,
    ) -> GuardedResponse:
        """Run the full guarded pipeline with detailed results."""
        input_results: list[GuardrailResult] = []
        output_results: list[GuardrailResult] = []

        # Step 1: Input guardrails
        effective_input = input_text
        for guardrail in self.input_guardrails:
            try:
                result = await guardrail.check_input(effective_input, **kwargs)
                input_results.append(result)
                if not result.allowed:
                    logger.warning(
                        "Input blocked by %s: %s",
                        guardrail.name, result.explanation,
                    )
                    return GuardedResponse(
                        output=self.blocked_response,
                        blocked=True,
                        blocked_by=f"input_guardrail:{guardrail.name}",
                        input_guardrail_results=input_results,
                    )
                # Use modified content if guardrail rewrote it
                if result.modified_content is not None:
                    effective_input = result.modified_content
            except Exception as e:
                logger.error("Input guardrail '%s' error: %s", guardrail.name, e)

        # Step 2: Model prediction
        try:
            response = await self.model.predict(
                input_text=effective_input, context=context, **kwargs
            )
            model_output = response.output
            model_metadata = response.metadata
        except Exception as e:
            logger.error("Model prediction failed: %s", e)
            return GuardedResponse(
                output=f"Model error: {e}",
                blocked=True,
                blocked_by="model_error",
                input_guardrail_results=input_results,
            )

        # Step 3: Output guardrails
        effective_output = model_output
        for guardrail in self.output_guardrails:
            try:
                result = await guardrail.check_output(
                    effective_output, user_input=input_text, **kwargs
                )
                output_results.append(result)
                if not result.allowed:
                    logger.warning(
                        "Output blocked by %s: %s",
                        guardrail.name, result.explanation,
                    )
                    return GuardedResponse(
                        output=self.blocked_response,
                        blocked=True,
                        blocked_by=f"output_guardrail:{guardrail.name}",
                        input_guardrail_results=input_results,
                        output_guardrail_results=output_results,
                        model_metadata=model_metadata,
                    )
                if result.modified_content is not None:
                    effective_output = result.modified_content
            except Exception as e:
                logger.error("Output guardrail '%s' error: %s", guardrail.name, e)

        # Step 4: Scorer checks
        scorer_results: dict[str, ScorerResult] = {}
        for scorer in self.output_scorers:
            try:
                result = await scorer.score_async(
                    output=effective_output,
                    input=input_text,
                    context=context,
                )
                scorer_results[scorer.name] = result

                if self.block_on_scorer_fail and not result.passed:
                    logger.warning(
                        "Output blocked by scorer %s: %s",
                        scorer.name, result.explanation,
                    )
                    return GuardedResponse(
                        output=self.blocked_response,
                        blocked=True,
                        blocked_by=f"scorer:{scorer.name}",
                        input_guardrail_results=input_results,
                        output_guardrail_results=output_results,
                        scorer_results=scorer_results,
                        model_metadata=model_metadata,
                    )
            except Exception as e:
                logger.error("Scorer '%s' error: %s", scorer.name, e)

        return GuardedResponse(
            output=effective_output,
            blocked=False,
            input_guardrail_results=input_results,
            output_guardrail_results=output_results,
            scorer_results=scorer_results,
            model_metadata=model_metadata,
        )
