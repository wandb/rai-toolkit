# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Weave scorer adapter — bridges rai_toolkit scorers to Weave scorers."""

from __future__ import annotations

import logging
from typing import Any

import weave

from rai_toolkit.scorers.base import BaseScorer, ScorerResult

logger = logging.getLogger(__name__)


def _weave_op_compat(**kwargs: Any) -> Any:
    """Return ``weave.op`` decorator across Weave versions.

    Older Weave releases do not accept newer metadata kwargs such as ``kind``.
    Dropping that hint is better than making the integration unimportable.
    """
    try:
        return weave.op(**kwargs)
    except TypeError as e:
        if "kind" not in kwargs or "kind" not in str(e):
            raise
        fallback = dict(kwargs)
        fallback.pop("kind", None)
        return weave.op(**fallback)


def _to_plain(obj: Any) -> Any:
    """Recursively coerce Weave-flavored containers (``WeaveList``,
    ``WeaveDict``) and other list/dict subclasses into plain ``list`` /
    ``dict``. Leaves scalar types alone.

    Required because ``dataclasses.asdict`` clones list/tuple values via
    ``type(obj)(generator)`` — a ``WeaveList`` constructor requires a
    ``server`` kwarg and raises ``TypeError`` otherwise. Stripping the
    subclass before values enter ``ScorerResult.details`` keeps the
    serializer happy without coupling rai_toolkit core to Weave types.
    """
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _filter_inner_scorer_kwargs(scorer: BaseScorer, extras: dict[str, Any]) -> dict[str, Any]:
    """Only forward optional row metadata a scorer's concrete ``score`` accepts."""
    if not extras:
        return {}
    import inspect

    try:
        sig = inspect.signature(scorer.score)
    except (TypeError, ValueError):
        return extras
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(extras)
    return {k: v for k, v in extras.items() if k in sig.parameters}


class WeaveRAIScorer(weave.Scorer):
    """Wraps any rai_toolkit BaseScorer as a Weave-compatible scorer.

    This adapter allows all rai_toolkit scorers (custom, LLM judges,
    programmatic) to work seamlessly with weave.Evaluation.

    Example::

        from rai_toolkit.scorers import FactualityJudge
        from integrations.weave_integration.scorers import WeaveRAIScorer

        rai_scorer = FactualityJudge(model="gpt-4o")
        weave_scorer = WeaveRAIScorer(rai_scorer=rai_scorer)

        # Use in weave.Evaluation
        evaluation = weave.Evaluation(
            dataset=my_dataset,
            scorers=[weave_scorer],
        )
    """

    rai_scorer_name: str = ""
    rai_scorer_category: str = ""

    _rai_scorer: BaseScorer | None = None

    def __init__(self, rai_scorer: BaseScorer, **kwargs: Any) -> None:
        # Set ``name`` (a weave.Scorer base field) to the rai scorer's class
        # name so the Evals UI shows e.g. ``FairnessJudge`` in the scorer
        # column instead of an unhelpful object-ref label. ``name`` also
        # drives ``Scorer.display_name``.
        rai_class = type(rai_scorer).__name__
        kwargs.setdefault("name", rai_class)
        super().__init__(
            rai_scorer_name=rai_scorer.name,
            rai_scorer_category=rai_scorer.category,
            **kwargs,
        )
        self._rai_scorer = rai_scorer

    @_weave_op_compat(
        call_display_name=lambda call: _wrapped_scorer_display_name(call),
        kind="scorer",
    )
    def score(
        self,
        output: dict[str, Any] | str,
        input_text: str = "",
        context: str = "",
        expected: str = "",
        rubrics: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Score using the wrapped rai_toolkit scorer.

        Accepts both dict output (from WeaveModel) and plain string. If the
        model exposed ``retrieved_context`` in its response metadata (i.e. the
        text it actually grounded on), prefer that over the dataset's
        ``context`` snippet — otherwise grounding scorers penalize models for
        responses grounded in retrievals the scorer never saw.

        ``rubrics`` is declared explicitly so Weave's column mapping forwards
        the dataset's per-row rubric list (e.g. HealthBench) to the underlying
        scorer. Scorers that don't consume rubrics ignore the kwarg.
        """
        if self._rai_scorer is None:
            return {"score": 0.0, "passed": False, "error": "No scorer configured"}

        if isinstance(output, dict):
            output_text = output.get("output", str(output))
            model_retrieved = output.get("retrieved_context")
        else:
            output_text = str(output)
            model_retrieved = None

        effective_context = (
            model_retrieved if model_retrieved else context
        )

        # Weave passes dataset rows as ``WeaveList`` / ``WeaveDict`` (list/
        # dict subclasses with extra runtime state). The rai_toolkit scorers
        # may stash these inputs in ``ScorerResult.details``; ``asdict``
        # then tries to clone the list with ``type(obj)(generator)`` and
        # crashes because ``WeaveList.__init__`` requires a ``server``
        # kwarg. Coerce to plain types at this boundary so nothing past
        # this point sees Weave-specific subclasses.
        plain_rubrics = _to_plain(rubrics) if rubrics is not None else None

        optional_inputs: dict[str, Any] = {}
        if expected:
            optional_inputs["expected"] = expected
        if plain_rubrics is not None:
            optional_inputs["rubrics"] = plain_rubrics

        result = self._rai_scorer.score(
            output=output_text,
            input=input_text,
            context=effective_context,
            **_filter_inner_scorer_kwargs(self._rai_scorer, optional_inputs),
            **kwargs,
        )

        # Serialize manually instead of ``asdict`` so we can coerce
        # WeaveList/WeaveDict anywhere they sneak into ``details``
        # (e.g. judge responses that echo dataset values). ``asdict``
        # tries to clone list values via ``type(obj)(generator)`` and
        # crashes on Weave's list subclass.
        return {
            "score": result.score,
            "passed": result.passed,
            "category": result.category,
            "explanation": result.explanation,
            "details": _to_plain(result.details),
            "assessed": result.assessed,
        }


def _wrapped_scorer_display_name(call: Any) -> str:
    """Per-call label for ``WeaveRAIScorer.score``.

    Renders e.g. ``FairnessJudge · MIT-1.1`` so each judge's trace row is
    distinguishable in the Weave UI. Without this, every wrapped rai
    scorer (LLM judge or programmatic) shows up as ``WeaveRAIScorer.score``
    and they are impossible to tell apart at a glance.

    Note: dynamically subclassing ``WeaveRAIScorer`` does *not* change the
    op_name — Weave captures op_name at decoration time on the parent
    class. Per-call display name is the supported override.
    """
    try:
        wrapper = (call.inputs or {}).get("self")
        if wrapper is None:
            return "scorer"
        # When Weave evaluates a published Scorer, ``call.inputs["self"]``
        # is an ObjectRef rather than the live ``WeaveRAIScorer`` instance.
        # ``ObjectRef`` doesn't carry our ``_rai_scorer`` private attr, so
        # we'd previously fall back to ``type(wrapper).__name__`` which
        # rendered as ``ObjectRef`` in the trace tree. The Scorer's
        # ``name`` field IS resolved through the ref (set in
        # ``WeaveRAIScorer.__init__`` to the rai-scorer class name), so
        # use it first.
        name = getattr(wrapper, "name", None)
        rai = getattr(wrapper, "_rai_scorer", None)
        if rai is not None:
            rai_name = type(rai).__name__
            category = getattr(rai, "category", None)
            return f"{rai_name} · {category}" if category else rai_name
        if name:
            return str(name)
        return type(wrapper).__name__
    except Exception:  # pragma: no cover — display name must never raise
        return "scorer"


def make_weave_rai_scorer(rai_scorer: BaseScorer) -> WeaveRAIScorer:
    """Construct a ``WeaveRAIScorer`` for an rai_toolkit scorer.

    Per-call display names come from ``call_display_name`` on the op (see
    :func:`_wrapped_scorer_display_name`); no dynamic subclassing needed.
    """
    return WeaveRAIScorer(rai_scorer=rai_scorer)


DEFAULT_COLUMN_MAP: dict[str, str] = {
    "query": "input_text",
    "input": "input_text",
    "prompt": "input_text",
    "question": "input_text",
    "expected": "expected",
    "ground_truth": "expected",
    "expected_output": "expected",
    "reference": "expected",
}


def get_weave_builtin_scorers(
    category_ids: list[str] | None = None,
    column_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Get Weave built-in scorers mapped to MIT risk categories.

    Only returns scorers that genuinely match risk categories.
    Pass category_ids to filter to specific categories.

    Args:
        category_ids: Filter to specific MIT category IDs.
        column_map: Optional mapping of scorer argument names to dataset column
            names. Merged over ``DEFAULT_COLUMN_MAP``. Used to reconcile Weave
            scorer signatures (e.g. ``query``, ``ground_truth``) with this
            project's dataset schema (``input_text``, ``context``, ``expected``).

    Returns:
        Dict mapping category_id to list of instantiated Weave scorers.
    """
    mappings: dict[str, list[dict[str, Any]]] = {
        "MIT-1.1": [
            {"class": "WeaveBiasScorerV1", "module": "weave.scorers", "kwargs": {"threshold": 0.5}},
        ],
        "MIT-1.2": [
            {"class": "WeaveToxicityScorerV1", "module": "weave.scorers", "kwargs": {"category_threshold": 2, "total_threshold": 5}},
            {"class": "OpenAIModerationScorer", "module": "weave.scorers", "kwargs": {}},
        ],
        "MIT-2.1": [
            {"class": "PresidioScorer", "module": "weave.scorers", "kwargs": {"language": "en"}},
        ],
        "MIT-3.1": [
            {"class": "WeaveHallucinationScorerV1", "module": "weave.scorers", "kwargs": {"threshold": 0.5}},
            {"class": "HallucinationFreeScorer", "module": "weave.scorers", "kwargs": {"model_id": "openai/gpt-4o"}},
            {"class": "ContextRelevancyScorer", "module": "weave.scorers", "kwargs": {"model_id": "openai/gpt-4o"}},
        ],
        "MIT-7.2": [
            {"class": "WeaveCoherenceScorerV1", "module": "weave.scorers", "kwargs": {}},
            {"class": "WeaveFluencyScorerV1", "module": "weave.scorers", "kwargs": {}},
        ],
        "COMPOSITE": [
            {"class": "WeaveTrustScorerV1", "module": "weave.scorers", "kwargs": {}},
        ],
    }

    effective_column_map = {**DEFAULT_COLUMN_MAP, **(column_map or {})}

    result: dict[str, list[Any]] = {}
    target_categories = set(category_ids) if category_ids else set(mappings.keys())

    for cat_id in target_categories:
        scorer_defs = mappings.get(cat_id, [])
        instantiated: list[Any] = []

        for scorer_def in scorer_defs:
            try:
                module = __import__(scorer_def["module"], fromlist=[scorer_def["class"]])
                cls = _make_dict_aware(getattr(module, scorer_def["class"]))
                scorer = cls(**scorer_def["kwargs"])
                _apply_column_map(scorer, effective_column_map)
                instantiated.append(scorer)
                logger.info("Loaded Weave scorer: %s for %s", scorer_def["class"], cat_id)
            except Exception as e:
                logger.debug(
                    "Weave scorer %s not available: %s", scorer_def["class"], e
                )

        if instantiated:
            result[cat_id] = instantiated

    return result


def _coerce_output_to_str(output: Any) -> str:
    """Normalize a model output payload to a plain string.

    ``WeaveModel.predict`` returns ``{"output": "...", "model": ..., ...}``,
    which Weave forwards verbatim as the ``output=`` arg to scorers. Most
    built-in Weave scorers (toxicity, fluency, OpenAI moderation) expect a
    plain ``str`` and pass it directly to a tokenizer or API call — handing
    them a dict raises a type/validation error.
    """
    if output is None:
        return ""
    if isinstance(output, dict):
        text = output.get("output") or output.get("text")
        return str(text) if text is not None else str(output)
    return str(output)


def _make_dict_aware(scorer_cls: type) -> type:
    """Subclass ``scorer_cls`` so its ``score`` coerces dict ``output`` to ``str``.

    Returns the input class unchanged if it has no ``score`` method.
    """
    import inspect

    original_score = getattr(scorer_cls, "score", None)
    if original_score is None:
        return scorer_cls

    try:
        original_sig = inspect.signature(original_score)
    except (TypeError, ValueError):
        return scorer_cls

    base_name = scorer_cls.__name__

    def _make_display_name(label: str) -> Any:
        def _display(call: Any) -> str:  # exactly one positional arg
            return label
        return _display

    @_weave_op_compat(
        name=f"{base_name}.score",
        call_display_name=_make_display_name(base_name),
        kind="scorer",
    )
    def score(self, *args: Any, **kwargs: Any) -> Any:
        # Lift the model's actual retrieved context out of the dict output
        # before we coerce output → str. Built-in Weave grounding scorers
        # (HallucinationFreeScorer, ContextRelevancyScorer, etc.) score
        # against the ``context`` arg they receive from the dataset row;
        # without this, they only see the fixture's reference snippet and
        # flag any model response grounded in additional retrievals.
        raw_output = kwargs.get("output") if "output" in kwargs else (args[0] if args else None)
        if isinstance(raw_output, dict):
            model_retrieved = raw_output.get("retrieved_context")
            if model_retrieved:
                kwargs["context"] = model_retrieved
        if "output" in kwargs:
            kwargs["output"] = _coerce_output_to_str(kwargs["output"])
        elif args:
            args = (_coerce_output_to_str(args[0]),) + tuple(args[1:])
        return original_score(self, *args, **kwargs)

    score.__signature__ = original_sig  # type: ignore[attr-defined]
    score.__doc__ = original_score.__doc__

    return type(
        f"{scorer_cls.__name__}DictAware",
        (scorer_cls,),
        {"score": score},
    )


def _apply_column_map(scorer: Any, column_map: dict[str, str]) -> None:
    """Set ``column_map`` on a Weave scorer, filtered to its actual score args.

    Weave raises if ``column_map`` contains keys that aren't parameters of the
    scorer's ``score`` method, so we introspect the signature and only pass
    relevant entries.
    """
    import inspect

    score_fn = getattr(scorer, "score", None)
    if score_fn is None:
        return

    try:
        sig = inspect.signature(score_fn)
        param_names = set(sig.parameters.keys())
    except (TypeError, ValueError):
        return

    filtered = {k: v for k, v in column_map.items() if k in param_names}
    if not filtered:
        return

    existing = getattr(scorer, "column_map", None) or {}
    try:
        scorer.column_map = {**filtered, **existing}
    except Exception as e:
        logger.debug("Could not set column_map on %s: %s", type(scorer).__name__, e)
