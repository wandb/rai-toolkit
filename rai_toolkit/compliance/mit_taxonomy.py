# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""MIT AI Risk Repository taxonomy — 7 domains, 24 subdomains.

Based on: "The AI Risk Repository: A Comprehensive Meta-Review, Database,
and Taxonomy of Risks From Artificial Intelligence" (MIT, 2024-2025).
Source: https://airisk.mit.edu/
"""

from __future__ import annotations

from rai_toolkit.compliance.frameworks import (
    Framework,
    RiskCategory,
    RiskSeverity,
)

# Domain 1: Discrimination & Toxicity
MIT_1_1 = RiskCategory(
    id="MIT-1.1",
    domain="Discrimination & Toxicity",
    subdomain="Unfair discrimination and bias",
    description=(
        "AI systems producing outputs that unfairly discriminate against individuals "
        "or groups based on protected characteristics such as race, gender, age, "
        "disability, or socioeconomic status."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.HIGH,
    scorer_ids=["FairnessJudge"],
    tags=["bias", "fairness", "discrimination"],
)

MIT_1_2 = RiskCategory(
    id="MIT-1.2",
    domain="Discrimination & Toxicity",
    subdomain="Exposure to toxic content",
    description=(
        "AI systems generating or surfacing toxic, hateful, abusive, or otherwise "
        "harmful content including hate speech, violent language, and harassment."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.HIGH,
    scorer_ids=["ContentSafetyJudge", "KeywordToxicityScorer"],
    tags=["toxicity", "hate-speech", "content-safety"],
)

MIT_1_3 = RiskCategory(
    id="MIT-1.3",
    domain="Discrimination & Toxicity",
    subdomain="Unequal performance across groups",
    description=(
        "AI systems performing significantly differently across demographic groups, "
        "leading to disparate quality of service or outcomes."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.HIGH,
    scorer_ids=["FairnessJudge"],
    tags=["equity", "performance-disparity", "fairness"],
)

# Domain 2: Privacy & Security
MIT_2_1 = RiskCategory(
    id="MIT-2.1",
    domain="Privacy & Security",
    subdomain="Compromise of privacy",
    description=(
        "AI systems obtaining, leaking, or inferring sensitive personal information "
        "including PII, PHI, financial data, or other confidential information."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.CRITICAL,
    scorer_ids=["PrivacyJudge", "RegexPIIScorer"],
    tags=["privacy", "pii", "data-leakage", "hipaa", "gdpr"],
)

MIT_2_2 = RiskCategory(
    id="MIT-2.2",
    domain="Privacy & Security",
    subdomain="AI system security vulnerabilities",
    description=(
        "Vulnerabilities in AI systems that can be exploited through prompt injection, "
        "jailbreaking, data poisoning, model extraction, or other adversarial attacks."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.CRITICAL,
    scorer_ids=["SecurityJudge"],
    tags=["security", "prompt-injection", "jailbreak", "adversarial"],
)

# Domain 3: Misinformation
MIT_3_1 = RiskCategory(
    id="MIT-3.1",
    domain="Misinformation",
    subdomain="False or misleading information",
    description=(
        "AI systems generating inaccurate, fabricated, or misleading information "
        "including hallucinations, false citations, invented facts, and unsupported claims."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.HIGH,
    scorer_ids=["FactualityJudge"],
    tags=["hallucination", "accuracy", "factuality", "misinformation"],
)

MIT_3_2 = RiskCategory(
    id="MIT-3.2",
    domain="Misinformation",
    subdomain="Pollution of information ecosystem",
    description=(
        "AI-generated content at scale degrading the quality and trustworthiness "
        "of the broader information ecosystem, making it harder to distinguish "
        "reliable from unreliable information."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.MEDIUM,
    scorer_ids=["FactualityJudge"],
    tags=["information-quality", "trust", "content-integrity"],
)

# Domain 4: Malicious Actors & Misuse
MIT_4_1 = RiskCategory(
    id="MIT-4.1",
    domain="Malicious Actors & Misuse",
    subdomain="Cyberattacks, weapons, and mass harm",
    description=(
        "AI being used to develop or execute cyberattacks, create weapons, "
        "or facilitate mass harm through automated capabilities."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.CRITICAL,
    scorer_ids=["SecurityJudge", "ContentSafetyJudge"],
    tags=["cybersecurity", "weapons", "dual-use"],
)

MIT_4_2 = RiskCategory(
    id="MIT-4.2",
    domain="Malicious Actors & Misuse",
    subdomain="Fraud, scams, and targeted manipulation",
    description=(
        "AI being used to generate convincing fraudulent content, social engineering "
        "attacks, deepfakes, or personalized manipulation campaigns."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.HIGH,
    scorer_ids=["ContentSafetyJudge"],
    tags=["fraud", "manipulation", "deepfakes", "social-engineering"],
)

MIT_4_3 = RiskCategory(
    id="MIT-4.3",
    domain="Malicious Actors & Misuse",
    subdomain="Disinformation and influence at scale",
    description=(
        "AI used to generate and distribute disinformation, propaganda, or "
        "influence operations at scale to manipulate public opinion."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.HIGH,
    scorer_ids=["ContentSafetyJudge"],
    tags=["disinformation", "propaganda", "influence-operations"],
)

# Domain 5: Human-Computer Interaction
MIT_5_1 = RiskCategory(
    id="MIT-5.1",
    domain="Human-Computer Interaction",
    subdomain="Overreliance and unsafe use",
    description=(
        "Users placing excessive trust in AI outputs without appropriate verification, "
        "leading to errors in critical decision-making contexts."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.HIGH,
    scorer_ids=["TransparencyJudge"],
    tags=["overreliance", "trust-calibration", "human-oversight"],
)

MIT_5_2 = RiskCategory(
    id="MIT-5.2",
    domain="Human-Computer Interaction",
    subdomain="Loss of human agency and autonomy",
    description=(
        "AI systems that undermine human decision-making autonomy, create "
        "dependency, or manipulate users through persuasive design."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.MEDIUM,
    scorer_ids=["TransparencyJudge"],
    tags=["autonomy", "agency", "manipulation"],
)

# Domain 6: Socioeconomic & Environmental
MIT_6_1 = RiskCategory(
    id="MIT-6.1",
    domain="Socioeconomic & Environmental",
    subdomain="Power centralization and unfair distribution of benefits",
    description=(
        "AI amplifying existing power imbalances by concentrating capabilities "
        "and benefits among a small number of actors."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.MEDIUM,
    scorer_ids=[],
    tags=["power-concentration", "inequality"],
)

MIT_6_2 = RiskCategory(
    id="MIT-6.2",
    domain="Socioeconomic & Environmental",
    subdomain="Labor market impacts",
    description=(
        "AI-driven displacement of workers, degradation of job quality, "
        "or creation of exploitative labor conditions."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.MEDIUM,
    scorer_ids=[],
    tags=["labor", "employment", "automation"],
)

MIT_6_3 = RiskCategory(
    id="MIT-6.3",
    domain="Socioeconomic & Environmental",
    subdomain="Economic and cultural devaluation of human effort",
    description=(
        "AI devaluing human creative, intellectual, or manual work by providing "
        "cheap automated alternatives without appropriate attribution or compensation."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.LOW,
    scorer_ids=[],
    tags=["creativity", "attribution", "compensation"],
)

MIT_6_4 = RiskCategory(
    id="MIT-6.4",
    domain="Socioeconomic & Environmental",
    subdomain="Environmental harm",
    description=(
        "Environmental costs of AI including energy consumption, carbon emissions, "
        "water usage, and electronic waste from training and running AI systems."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.MEDIUM,
    scorer_ids=[],
    tags=["environment", "energy", "sustainability"],
)

MIT_6_5 = RiskCategory(
    id="MIT-6.5",
    domain="Socioeconomic & Environmental",
    subdomain="Erosion of social cohesion",
    description=(
        "AI contributing to social fragmentation, polarization, erosion of "
        "shared reality, and weakening of democratic institutions."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.MEDIUM,
    scorer_ids=[],
    tags=["social-cohesion", "polarization", "democracy"],
)

MIT_6_6 = RiskCategory(
    id="MIT-6.6",
    domain="Socioeconomic & Environmental",
    subdomain="Competitive dynamics and governance failure",
    description=(
        "Race-to-the-bottom dynamics in AI development where competitive pressures "
        "lead to inadequate safety practices and governance failures."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.MEDIUM,
    scorer_ids=[],
    tags=["governance", "competition", "regulation"],
)

# Domain 7: AI System Safety, Failures & Limitations
MIT_7_1 = RiskCategory(
    id="MIT-7.1",
    domain="AI System Safety, Failures & Limitations",
    subdomain="AI system failures and unexpected behavior",
    description=(
        "AI systems failing, producing unexpected outputs, or behaving "
        "unpredictably due to edge cases, distribution shift, or software bugs."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.HIGH,
    scorer_ids=[],
    tags=["reliability", "robustness", "failures"],
)

MIT_7_2 = RiskCategory(
    id="MIT-7.2",
    domain="AI System Safety, Failures & Limitations",
    subdomain="Lack of transparency and interpretability",
    description=(
        "AI systems operating as black boxes without sufficient transparency "
        "into their reasoning, decision-making process, or limitations."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.HIGH,
    scorer_ids=["ExplainabilityJudge"],
    tags=["transparency", "explainability", "interpretability"],
)

MIT_7_3 = RiskCategory(
    id="MIT-7.3",
    domain="AI System Safety, Failures & Limitations",
    subdomain="Lack of accountability",
    description=(
        "Unclear or absent accountability structures for AI system outcomes, "
        "making it difficult to assign responsibility for harms."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.MEDIUM,
    scorer_ids=[],
    tags=["accountability", "responsibility", "liability"],
)

MIT_7_4 = RiskCategory(
    id="MIT-7.4",
    domain="AI System Safety, Failures & Limitations",
    subdomain="AI goal misalignment",
    description=(
        "AI systems pursuing objectives that diverge from intended human goals "
        "or values, including reward hacking and specification gaming."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.HIGH,
    scorer_ids=[],
    tags=["alignment", "goal-misalignment", "reward-hacking"],
)

MIT_7_5 = RiskCategory(
    id="MIT-7.5",
    domain="AI System Safety, Failures & Limitations",
    subdomain="AI possessing dangerous capabilities",
    description=(
        "AI systems developing or being given capabilities that pose inherent "
        "risks such as autonomous action, self-replication, or deception."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.CRITICAL,
    scorer_ids=[],
    tags=["dangerous-capabilities", "autonomy", "deception"],
)

MIT_7_6 = RiskCategory(
    id="MIT-7.6",
    domain="AI System Safety, Failures & Limitations",
    subdomain="Multi-agent and emergent risks",
    description=(
        "Risks arising from interactions between multiple AI systems, including "
        "emergent behaviors, coordination failures, and cascading effects."
    ),
    framework=Framework.MIT_AI_RISK,
    severity=RiskSeverity.MEDIUM,
    scorer_ids=[],
    tags=["multi-agent", "emergent", "cascading"],
)


# Complete taxonomy as a dict for easy lookup
MIT_TAXONOMY: dict[str, RiskCategory] = {
    cat.id: cat
    for cat in [
        MIT_1_1, MIT_1_2, MIT_1_3,
        MIT_2_1, MIT_2_2,
        MIT_3_1, MIT_3_2,
        MIT_4_1, MIT_4_2, MIT_4_3,
        MIT_5_1, MIT_5_2,
        MIT_6_1, MIT_6_2, MIT_6_3, MIT_6_4, MIT_6_5, MIT_6_6,
        MIT_7_1, MIT_7_2, MIT_7_3, MIT_7_4, MIT_7_5, MIT_7_6,
    ]
}

# Domains with their subcategory IDs
MIT_DOMAINS: dict[str, list[str]] = {
    "Discrimination & Toxicity": ["MIT-1.1", "MIT-1.2", "MIT-1.3"],
    "Privacy & Security": ["MIT-2.1", "MIT-2.2"],
    "Misinformation": ["MIT-3.1", "MIT-3.2"],
    "Malicious Actors & Misuse": ["MIT-4.1", "MIT-4.2", "MIT-4.3"],
    "Human-Computer Interaction": ["MIT-5.1", "MIT-5.2"],
    "Socioeconomic & Environmental": ["MIT-6.1", "MIT-6.2", "MIT-6.3", "MIT-6.4", "MIT-6.5", "MIT-6.6"],
    "AI System Safety, Failures & Limitations": ["MIT-7.1", "MIT-7.2", "MIT-7.3", "MIT-7.4", "MIT-7.5", "MIT-7.6"],
}


def get_all_categories() -> list[RiskCategory]:
    """Return all MIT AI Risk Repository categories."""
    return list(MIT_TAXONOMY.values())


def get_category(category_id: str) -> RiskCategory | None:
    """Look up a category by ID."""
    return MIT_TAXONOMY.get(category_id)


def get_categories_by_domain(domain: str) -> list[RiskCategory]:
    """Get all categories for a domain."""
    ids = MIT_DOMAINS.get(domain, [])
    return [MIT_TAXONOMY[id_] for id_ in ids if id_ in MIT_TAXONOMY]


def get_scorable_categories() -> list[RiskCategory]:
    """Return only categories that have associated scorers."""
    return [c for c in MIT_TAXONOMY.values() if c.scorer_ids]
