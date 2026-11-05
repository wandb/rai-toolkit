# SPDX-FileCopyrightText: 2026 EffNine
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""NYC Local Law 144 of 2021 compliance mapping for automated employment
decision tools (AEDTs).

Maps the obligations in NYC Admin. Code Sec. Sec. 20-870 through 20-874 and
the DCWP implementing rule (6 RCNY Subchapter T, Sec. Sec. 5-300 to 5-304) to
RAI toolkit capabilities and MIT risk categories.

Key facts encoded here (verified against the primary sources below):

- It is unlawful for an employer or employment agency to use an AEDT in NYC
  unless a bias audit was conducted no more than one year prior to use
  (Sec. 20-871(a)(1); rule 5-301(a)). Audit currency is time-based only;
  the law has no material-change re-audit trigger.
- The audit (an impartial evaluation by an independent auditor) computes
  selection rates (or scoring rates against the sample median) and impact
  ratios for each EEO-1 Component 1 category: sex, race/ethnicity, and
  intersectional categories (rule 5-301(b)-(c)). Neither the law nor the
  rule sets a numeric impact-ratio compliance threshold (such as 0.8) and
  DCWP states the law requires no specific action based only on audit
  results.
- A summary of the most recent audit results and the distribution date
  (rule 5-300: the date the employer or employment agency began using
  that specific AEDT) must be publicly posted on the employer's
  employment section of its website, remaining posted for at least six
  months after the latest use of the AEDT (Sec. 20-871(a)(2);
  rule 5-303).
- Candidates and employees who reside in NYC must receive notice at least
  10 business days before the AEDT is used, covering: that an AEDT will
  be used, instructions for requesting an alternative selection process
  or a reasonable accommodation under other laws (if available), and the
  job qualifications and characteristics the tool will assess
  (Sec. 20-871(b); rule 5-304).
- Obligations fall on the employer or employment agency; the vendor that
  created the tool is not responsible for the bias audit (DCWP FAQ). A
  vendor may commission an audit of its own tool, but the employer
  remains ultimately responsible.

Sources:
- Law text (Admin. Code ch. 5, subch. 25): https://codelibrary.amlegal.com/codes/newyorkcity/latest/NYCadmin
- DCWP rule (6 RCNY Subchapter T): https://rules.cityofnewyork.us/wp-content/uploads/2023/04/DCWP-NOA-for-Use-of-Automated-Employment-Decisionmaking-Tools-2.pdf
- DCWP AEDT page: https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page
- DCWP FAQ: https://www.nyc.gov/assets/dca/downloads/pdf/about/DCWP-AEDT-FAQ.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NYCLL144Requirement:
    """A NYC LL 144 obligation with mapping to toolkit capabilities.

    Attributes:
        id: Requirement identifier.
        section: NYC Administrative Code / rule section reference.
        title: Short title.
        description: What the law mandates.
        mit_category_ids: MIT risk categories relevant to this obligation.
        rai_capabilities: Implemented toolkit capabilities that assist here.
        coverage_gaps: LL 144 obligations the toolkit does not implement.
    """

    id: str
    section: str
    title: str
    description: str
    mit_category_ids: list[str] = field(default_factory=list)
    rai_capabilities: list[str] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)


NYC_LL_144_REQUIREMENTS: dict[str, NYCLL144Requirement] = {
    "NYC-LL144-1": NYCLL144Requirement(
        id="NYC-LL144-1",
        section="Section 20-871(a)(1); rule 5-301(a)",
        title="Annual Bias Audit",
        description=(
            "It is unlawful for an employer or employment agency to use an "
            "automated employment decision tool (AEDT) in New York City "
            "unless a bias audit was conducted no more than one year prior "
            "to such use. The bias audit is an impartial evaluation by an "
            "independent auditor that includes testing the AEDT for disparate "
            "impact on EEO-1 Component 1 categories (sex, race, and "
            "ethnicity). The audit uses historical AEDT-use data by default, "
            "with test data permitted only in limited circumstances per "
            "rule 5-302."
        ),
        mit_category_ids=[
            "MIT-1.1",  # Unfair discrimination and bias
            "MIT-1.3",  # Unequal performance across groups
        ],
        rai_capabilities=[
            "FairnessJudge scorer flags demographic bias and stereotyping "
            "in AEDT outputs (MIT-1.1)",
            "BBQ and BOLD example datasets provide demographic-split bias "
            "benchmarks for evaluation runs",
            "Evaluation pipeline aggregates per-scorer results for bias "
            "evidence gathering",
        ],
        coverage_gaps=[
            "Audits must be performed by an independent auditor; the "
            "toolkit's LLM-judge scores are not an independent bias audit",
            "The toolkit does not track audit dates or enforce the one-year "
            "currency window",
        ],
    ),
    "NYC-LL144-2": NYCLL144Requirement(
        id="NYC-LL144-2",
        section="Section 20-870 (definition of bias audit); rule 5-301(b)-(c)",
        title="Impact Ratio Calculation and Reporting",
        description=(
            "The bias audit must compute and report, for each EEO-1 "
            "Component 1 category (sex, race/ethnicity, and intersectional "
            "combinations), the selection rate (or, for scoring tools, the "
            "scoring rate against the sample median) and the impact ratio "
            "(the category's rate divided by the rate of the most-selected "
            "or highest-scoring category), with separate calculations for "
            "each classification group. Counts for the unknown category must "
            "be retained and shown alongside applicant counts and rates. "
            "Categories representing less than "
            "2% of the data may be excluded from impact-ratio calculations "
            "with justification, but applicant counts and rates must still "
            "be shown. The published summary must explain the data source. "
            "Neither the law nor the rule establishes a numeric "
            "impact-ratio threshold for compliance."
        ),
        mit_category_ids=[
            "MIT-1.1",  # Unfair discrimination and bias
            "MIT-1.3",  # Unequal performance across groups
        ],
        rai_capabilities=[
            "BBQ example dataset is split across demographic categories for "
            "per-group performance comparison (MIT-1.3); these are benchmark "
            "slices, not EEO-1 applicant-group outcomes",
            "Evaluation pipeline reports scores by MIT risk category (not by "
            "EEO-1 applicant groups); these do not substitute for the "
            "auditor's selection/scoring-rate and impact-ratio calculations "
            "— use only as supporting bias evidence for human review",
        ],
        coverage_gaps=[
            "The toolkit does not calculate selection rates, scoring rates, "
            "or impact ratios; these are computed by the independent auditor",
            "The toolkit does not model EEO-1 Component 1 category "
            "taxonomies or the intersectional-category requirements",
            "No four-fifths-rule or other numeric threshold is implemented; "
            "the law sets none",
        ],
    ),
    "NYC-LL144-3": NYCLL144Requirement(
        id="NYC-LL144-3",
        section="Section 20-871(b); rule 5-304",
        title="Candidate Notice",
        description=(
            "Employers and employment agencies must give each NYC-resident "
            "candidate or employee written notice at least 10 business days "
            "before an AEDT is used. The notice must state that an AEDT will "
            "be used in the assessment, include instructions for requesting "
            "an alternative selection process or a reasonable accommodation "
            "under other laws (if available), and describe the job "
            "qualifications and characteristics the AEDT will assess. "
            "Employers must also post on their website the type of data the "
            "AEDT collects, its source, and the data retention policy, "
            "together with instructions for requesting such information, and "
            "must respond in writing within 30 days to a candidate's or "
            "employee's written request about the type of data collected, "
            "its source, and the retention policy, including an explanation "
            "when disclosure is legally barred."
        ),
        mit_category_ids=[
            "MIT-5.1",  # Overreliance and false transparency
            "MIT-7.2",  # Transparency and explainability
        ],
        rai_capabilities=[
            "TransparencyJudge scorer checks general disclosure of the "
            "system's nature and limitations in outputs (MIT-5.1); it does "
            "not verify the 10-business-day AEDT notice or the job "
            "qualifications/characteristics disclosure",
            "Evaluation traces record model inputs/outputs for debugging; "
            "they do not establish the employer's data source, collection "
            "notice, or retention policy",
        ],
        coverage_gaps=[
            "Notice delivery, 10-business-day timing, and accommodation "
            "request workflows are employer-side process requirements the "
            "toolkit does not implement",
            "The toolkit does not manage the website data-disclosure "
            "posting required by rule 5-304(d)",
        ],
    ),
    "NYC-LL144-4": NYCLL144Requirement(
        id="NYC-LL144-4",
        section="Section 20-871(a)(2); rule 5-303",
        title="Publication of Audit Results",
        description=(
            "Before using an AEDT, the employer or employment agency must "
            "publicly post on the employment section of its website the date "
            "of the most recent bias audit and a summary of the results — "
            "including an explanation of the data source, the number of "
            "applicants or candidates (including unknown-category counts), "
            "and selection or scoring rates and impact ratios for each "
            "classification group and all categories — together with the "
            "distribution date (rule 5-300: the date the employer or "
            "employment agency began using that specific AEDT). The summary "
            "must remain posted for at least six "
            "months after the latest use of the AEDT."
        ),
        mit_category_ids=[
            "MIT-5.1",  # Overreliance and false transparency
            "MIT-7.2",  # Transparency and explainability
        ],
        rai_capabilities=[
            "Assessment reports list scorers, thresholds, and per-category "
            "results that a human may attach alongside — not instead of — "
            "the statutory audit summary",
            "Weave-traced evaluation runs preserve a versioned history of "
            "bias-evaluation results",
        ],
        coverage_gaps=[
            "The toolkit does not produce the statutory audit summary "
            "(dates, applicant counts, rates, impact ratios) or manage the "
            "six-month posting obligation",
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
}


def format_nyc_ll144_framework_label(
    requirement_id: str,
    *,
    section: str | None = None,
    title: str | None = None,
) -> str:
    """Build a framework row label like ``NYC LL 144: Sec 20-871(a)(1) (Annual bias audit)``."""
    section_label = section or requirement_id
    subtitle = NYC_SECTION_DISPLAY_SUBTITLES.get(requirement_id)
    if subtitle is None and title:
        subtitle = title
    if subtitle:
        return f"NYC LL 144: {section_label} ({subtitle})"
    return f"NYC LL 144: {section_label}"
