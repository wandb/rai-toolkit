# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Scorer framework: base classes and built-in scorers for RAI evaluation."""

from rai_toolkit.scorers.base import BaseScorer, ScorerResult
from rai_toolkit.scorers.composite import CompositeScorer
from rai_toolkit.scorers.llm_judges import (
    ContentSafetyJudge,
    ExplainabilityJudge,
    FactualityJudge,
    FairnessJudge,
    GroundednessScorer,
    LLMJudgeScorer,
    PrivacyJudge,
    RetrievalRelevanceScorer,
    RubricScorer,
    SecurityJudge,
    TransparencyJudge,
)
from rai_toolkit.scorers.normalizer import ScoreNormalizer
from rai_toolkit.scorers.programmatic import (
    KeywordToxicityScorer,
    OutputFormatScorer,
    RegexPIIScorer,
    ResponseLengthScorer,
)

__all__ = [
    "BaseScorer",
    "CompositeScorer",
    "ContentSafetyJudge",
    "ExplainabilityJudge",
    "FactualityJudge",
    "FairnessJudge",
    "GroundednessScorer",
    "KeywordToxicityScorer",
    "LLMJudgeScorer",
    "OutputFormatScorer",
    "PrivacyJudge",
    "RegexPIIScorer",
    "RetrievalRelevanceScorer",
    "ResponseLengthScorer",
    "RubricScorer",
    "ScoreNormalizer",
    "ScorerResult",
    "SecurityJudge",
    "TransparencyJudge",
]
