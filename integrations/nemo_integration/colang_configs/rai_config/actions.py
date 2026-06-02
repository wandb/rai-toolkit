# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom NeMo Guardrails actions for RAI checks."""

import re


async def check_toxicity(text: str) -> bool:
    """Check if text contains toxic content using keyword matching.

    In production, replace with a model-based classifier like
    WeaveToxicityScorerV1 or Detoxify.
    """
    toxic_patterns = [
        r"\b(hate|kill|murder|attack|destroy)\b",
        r"\b(stupid|idiot|moron)\b",
        r"\b(racist|sexist)\s+(content|material)\b",
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in toxic_patterns)


async def check_pii(text: str) -> bool:
    """Check if text contains PII using regex patterns.

    In production, use PresidioScorer or a dedicated PII detection model.
    """
    pii_patterns = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # Credit card
    ]
    return any(re.search(p, text) for p in pii_patterns)


async def check_factuality(text: str) -> bool:
    """Basic factuality check.

    In production, use FactualityJudge or WeaveHallucinationScorerV1.
    This is a placeholder that always returns True.
    """
    # Placeholder — in production, call an LLM or use a fact-checking service
    return True
