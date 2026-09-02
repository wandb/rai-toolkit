# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""NYC Local Law 144 compliance mapping for automated employment decision tools.

Maps NYC LL 144 requirements (bias audits, notice, impact ratios) to
RAI toolkit capabilities and MIT risk categories.

Source: NYC Local Law 144 of 2023 — Automated Employment Decision Tools.
https://www.nyc.gov/site/ccl/rules/local-law-144.page
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NYCLL144Requirement:
    """A NYC LL 144 requirement with mapping to toolkit capabilities.

    Attributes:
        id: Requirement identifier.
        section: NYC Administrative Code section reference.
        title: Short title.
        description: What the requirement mandates.
        mit_category_ids: MIT risk categories that help address this requirement.
        rai_capabilities: How the RAI toolkit helps with compliance.
    """

    id: str
    section: str
    title: str
    description: str
    mit_category_ids: list[str] = field(default_factory=list)
    rai_capabilities: list[str] = field(default_factory=list)


NYC_LL_144_REQUIREMENTS: dict[str, NYCLL144Requirement] = {
    "NYC-LL144-1": NYCLL144Requirement(
        id="NYC-LL144-1",
        section="Section 20-144(a)(1)",
        title="Bias Audit — Annual Bias Assessment",
        description=(
            "Employers and employment agencies must conduct an independent bias "
            "audit of each automated employment decision tool (AEDT) used in the "
            "city at least once every two years, prior to use or material "
            "modification. The audit must assess the impact of the tool on "
            "candidates across protected classes including race, sex, and "
            "ethnicity."
        ),
        mit_category_ids=[
            "MIT-1.1",  # Unfair discrimination and bias
            "MIT-1.3",  # Unequal performance across groups
        ],
        rai_capabilities=[
            "FairnessJudge scorer evaluates demographic bias across protected classes",
            "Bias benchmarks test impact ratios by category",
            "Evaluation pipeline quantifies disparate impact with normalized scores",
        ],
    ),
    "NYC-LL144-2": NYCLL144Requirement(
        id="NYC-LL144-2",
        section="Section 20-144(a)(2)",
        title="Bias Audit — Impact Ratio Reporting",
        description=(
            "The bias audit must calculate and report impact ratios (selection "
            "rates) for each protected category. An adverse impact threshold "
            "of 0.8 (the '80% rule') is used to determine whether a protected "
            "group is disproportionately affected by the AEDT."
        ),
        mit_category_ids=[
            "MIT-1.1",  # Unfair discrimination and bias
            "MIT-1.3",  # Unequal performance across groups
        ],
        rai_capabilities=[
            "FairnessJudge provides per-group performance metrics and impact ratios",
            "Comparative evaluations track selection-rate disparities across versions",
            "Scorer registry supports custom impact-ratio aggregation",
        ],
    ),
    "NYC-LL144-3": NYCLL144Requirement(
        id="NYC-LL144-3",
        section="Section 20-144(b)",
        title="Notice to Candidates",
        description=(
            "Employers must provide clear notice to candidates that an AEDT "
            "will be used in the assessment process. The notice must describe "
            "the category of measurements or outputs the tool is designed to "
            "evaluate and any available action to contest or opt out of the "
            "assessment."
        ),
        mit_category_ids=[
            "MIT-7.2",  # Transparency and explainability
        ],
        rai_capabilities=[
            "Transparency scorer evaluates whether system disclosures are adequate",
            "Evaluation traces document what inputs the tool processes",
            "Compliance reports include candidate-facing notices as audit artifacts",
        ],
    ),
    "NYC-LL144-4": NYCLL144Requirement(
        id="NYC-LL144-4",
        section="Section 20-144(c)",
        title="Publication of Bias Audit Results",
        description=(
            "Employers must make the bias audit results publicly available, "
            "including the impact ratios by protected category, on the "
            "employer's website or through another accessible means. Results "
            "must be retained for at least two years."
        ),
        mit_category_ids=[
            "MIT-7.2",  # Transparency and explainability
            "MIT-7.1",  # System failures and robustness
        ],
        rai_capabilities=[
            "Compliance reports generate structured bias-audit documentation",
            "Versioned evaluation traces preserve audit history for retention",
            "Evaluation pipeline exports impact-ratio summaries in machine-readable format",
        ],
    ),
    "NYC-LL144-5": NYCLL144Requirement(
        id="NYC-LL144-5",
        section="Section 20-144(a)(3)",
        title="Material Changes — Re-Audit Requirement",
        description=(
            "A new bias audit must be conducted whenever a material change is "
            "made to an AEDT that is reasonably likely to produce a different "
            "impact ratio. This includes changes to the tool's algorithms, "
            "training data, or the categories of data it evaluates."
        ),
        mit_category_ids=[
            "MIT-1.1",  # Unfair discrimination and bias
            "MIT-7.1",  # System failures and robustness
        ],
        rai_capabilities=[
            "Comparative evaluations detect performance drift after model updates",
            "Evaluation versioning tracks score changes across tool revisions",
            "Bias benchmarks re-run automatically on material-change detection",
        ],
    ),
    "NYC-LL144-6": NYCLL144Requirement(
        id="NYC-LL144-6",
        section="Section 20-144(d)",
        title="Vendor Certification and Liability",
        description=(
            "AEDT vendors must certify to the employer that the tool has "
            "undergone the required bias audit and provide the results. "
            "Both the employer and the vendor may be held liable for "
            "violations of the law."
        ),
        mit_category_ids=[
            "MIT-7.2",  # Transparency and explainability
            "MIT-2.2",  # Security and resilience
        ],
        rai_capabilities=[
            "Evaluation reports can be shared with vendors as certification evidence",
            "Trace and feedback workflows capture vendor-provided audit data",
            "Compliance engine surfaces vendor-certification status in reports",
        ],
    ),
}


def get_requirement(requirement_id: str) -> NYCLL144Requirement | None:
    """Look up a NYC LL 144 requirement."""
    return NYC_LL_144_REQUIREMENTS.get(requirement_id)


def get_all_nyc_ll144_mit_categories() -> list[str]:
    """Get all MIT risk categories relevant to NYC LL 144 compliance."""
    categories: set[str] = set()
    for req in NYC_LL_144_REQUIREMENTS.values():
        categories.update(req.mit_category_ids)
    return sorted(categories)


def get_mit_categories_for_nyc_requirement(requirement_id: str) -> list[str]:
    """Get MIT risk categories that address a specific NYC LL 144 requirement."""
    req = NYC_LL_144_REQUIREMENTS.get(requirement_id)
    return req.mit_category_ids if req else []


# Reviewer-facing subtitles for framework coverage tables (Weave panel, HTML
# report, Streamlit). Kept short so rows stay scannable in narrow panels.
NYC_SECTION_DISPLAY_SUBTITLES: dict[str, str] = {
    "NYC-LL144-1": "Annual bias audit",
    "NYC-LL144-2": "Impact ratio reporting",
    "NYC-LL144-3": "Candidate notice",
    "NYC-LL144-4": "Publish audit results",
    "NYC-LL144-5": "Re-audit on material change",
    "NYC-LL144-6": "Vendor certification",
}


def format_nyc_ll144_framework_label(
    requirement_id: str,
    *,
    section: str | None = None,
    title: str | None = None,
) -> str:
    """Build a framework row label like ``NYC LL 144: Sec 20-144(a)(1) (Annual bias audit)``."""
    section_label = section or requirement_id
    subtitle = NYC_SECTION_DISPLAY_SUBTITLES.get(requirement_id)
    if subtitle is None and title:
        subtitle = title
    if subtitle:
        return f"NYC LL 144: {section_label} ({subtitle})"
    return f"NYC LL 144: {section_label}"
