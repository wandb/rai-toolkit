"""Scorer framework — base classes and built-in scorers for RAI evaluation."""

from rai_toolkit.scorers.base import BaseScorer, ScorerResult
from rai_toolkit.scorers.llm_judges import (
    LLMJudgeScorer,
    FactualityJudge,
    FairnessJudge,
    ContentSafetyJudge,
    PrivacyJudge,
    SecurityJudge,
    TransparencyJudge,
    ExplainabilityJudge,
    RubricScorer,
)
from rai_toolkit.scorers.programmatic import (
    RegexPIIScorer,
    KeywordToxicityScorer,
    OutputFormatScorer,
    ResponseLengthScorer,
)
from rai_toolkit.scorers.composite import CompositeScorer
from rai_toolkit.scorers.normalizer import ScoreNormalizer

__all__ = [
    "BaseScorer",
    "ScorerResult",
    "LLMJudgeScorer",
    "FactualityJudge",
    "FairnessJudge",
    "ContentSafetyJudge",
    "PrivacyJudge",
    "SecurityJudge",
    "TransparencyJudge",
    "ExplainabilityJudge",
    "RubricScorer",
    "RegexPIIScorer",
    "KeywordToxicityScorer",
    "OutputFormatScorer",
    "ResponseLengthScorer",
    "CompositeScorer",
    "ScoreNormalizer",
]
