"""W&B Weave integration — connects the rai_toolkit toolkit to Weave.

Provides adapters for: tracing, models, scorers, and evaluations. LLM
cost tracking is intentionally not in here — Weave records per-op token
spend automatically when ``weave.init`` runs.
"""

from integrations.weave_integration.tracing import weave_init, traced
from integrations.weave_integration.models import WeaveModel
from integrations.weave_integration.scorers import WeaveRAIScorer, get_weave_builtin_scorers
from integrations.weave_integration.evaluation import WeaveEvaluationRunner
from integrations.weave_integration.feedback import (
    attach_manual_finding,
    attach_reviewer_decision,
    probe_thread,
)

# Importing ``views`` registers the assessment HTML renderer, the
# ``weave.set_view`` adapter, and the per-op extensions (call_display_name,
# postprocess_output) with ``rai_toolkit._tracing``. Side-effect import
# is intentional — keeping it here means the toolkit never has to know
# the integration's internals.
from integrations.weave_integration import views as _views  # noqa: F401

__all__ = [
    "weave_init",
    "traced",
    "WeaveModel",
    "WeaveRAIScorer",
    "get_weave_builtin_scorers",
    "WeaveEvaluationRunner",
    "attach_manual_finding",
    "attach_reviewer_decision",
    "probe_thread",
]
