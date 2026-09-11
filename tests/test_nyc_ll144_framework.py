# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Tests for NYC LL 144 compliance framework integration.

Pins the authoritative legal facts for NYC Local Law 144 of 2021
(Admin. Code Sec. Sec. 20-870 to 20-874; DCWP rule 6 RCNY Subchapter T):
one-year bias-audit window, 10-business-day notice, six-month posting
retention, no numeric impact-ratio threshold, and employer-side
responsibility.
"""

from __future__ import annotations

from rai_toolkit.compliance.engine import ComplianceMappingEngine
from rai_toolkit.compliance.frameworks import Framework
from rai_toolkit.compliance.nyc_ll144_mapping import (
    NYC_LL_144_REQUIREMENTS,
    format_nyc_ll144_framework_label,
    get_all_nyc_ll144_mit_categories,
    get_mit_categories_for_nyc_requirement,
)


def test_nyc_ll_144_framework_is_registered() -> None:
    """NYC LL 144 appears in the framework registry."""
    eng = ComplianceMappingEngine()
    frameworks = eng.get_frameworks()
    assert Framework.NYC_LL_144 in frameworks


def test_nyc_ll_144_sections_cite_authoritative_law() -> None:
    """Requirements cite Sec. 20-870/20-871 and the DCWP rule, not Sec. 20-144."""
    for req in NYC_LL_144_REQUIREMENTS.values():
        assert "20-144" not in req.section, (
            f"{req.id} cites nonexistent section 20-144; the governing law "
            "is Admin. Code sections 20-870 through 20-874"
        )
        assert "20-87" in req.section or "5-30" in req.section, (
            f"{req.id} must cite the law (20-870/20-871) or the DCWP rule "
            f"(5-300 to 5-304); got {req.section}"
        )


def test_nyc_ll_144_bias_audit_window_is_one_year() -> None:
    """The bias-audit currency window is one year, not two."""
    audit_req = NYC_LL_144_REQUIREMENTS["NYC-LL144-1"]
    assert "one year" in audit_req.description
    assert "two years" not in audit_req.description
    # Audit currency is time-based only; there is no material-change trigger.
    assert "material change" not in audit_req.description.lower()
    assert "material modification" not in audit_req.description.lower()


def test_nyc_ll_144_no_numeric_impact_ratio_threshold() -> None:
    """The law sets no numeric impact-ratio compliance threshold."""
    audit_req = NYC_LL_144_REQUIREMENTS["NYC-LL144-2"]
    assert "0.8" not in audit_req.description
    assert "80%" not in audit_req.description
    # The description states the absence of a threshold in plain terms.
    assert "no numeric impact-ratio threshold" in audit_req.description.lower() or (
        "neither the law nor the rule establishes" in audit_req.description.lower()
    )
    joined_gaps = " ".join(audit_req.coverage_gaps).lower()
    assert "threshold" in joined_gaps and (
        "sets none" in joined_gaps or "no numeric" in joined_gaps or "none" in joined_gaps
    )


def test_nyc_ll_144_notice_is_10_business_days() -> None:
    """Notice must be provided at least 10 business days before use."""
    notice_req = NYC_LL_144_REQUIREMENTS["NYC-LL144-3"]
    assert "10 business days" in notice_req.description


def test_nyc_ll_144_posting_retention_is_six_months() -> None:
    """Audit summaries must remain posted for at least six months after latest use."""
    posting_req = NYC_LL_144_REQUIREMENTS["NYC-LL144-4"]
    assert "six months" in posting_req.description
    assert "two years" not in posting_req.description


def test_nyc_ll_144_no_material_change_or_vendor_requirements() -> None:
    """The law has no material-change re-audit trigger or vendor certification."""
    for req_id, req in NYC_LL_144_REQUIREMENTS.items():
        assert "material change" not in req.title.lower()
        assert "vendor" not in req.title.lower()
    titles = {req.title.lower() for req in NYC_LL_144_REQUIREMENTS.values()}
    assert not any("re-audit" in t for t in titles)
    assert not any("certification" in t for t in titles)


def test_nyc_ll_144_resolves_mit_categories() -> None:
    """Resolving NYC LL 144 categories returns the expected MIT categories."""
    eng = ComplianceMappingEngine()
    categories = eng.get_categories(Framework.NYC_LL_144)
    ids = {c.id for c in categories}
    # The four requirements cover: MIT-1.1, MIT-1.3 (bias), MIT-5.1, MIT-7.2
    # (transparency). MIT-2.2/7.1 were dropped with the removed vendor and
    # material-change requirements.
    assert ids == {"MIT-1.1", "MIT-1.3", "MIT-5.1", "MIT-7.2"}
    assert get_all_nyc_ll144_mit_categories() == ["MIT-1.1", "MIT-1.3", "MIT-5.1", "MIT-7.2"]


def test_nyc_ll_144_domains_are_structured() -> None:
    """Each NYC LL 144 requirement maps to its MIT category IDs."""
    eng = ComplianceMappingEngine()
    domains = eng.get_domains(Framework.NYC_LL_144)
    assert len(domains) == 4
    for title, mit_ids in domains.items():
        assert isinstance(mit_ids, list)
        assert len(mit_ids) > 0


def test_nyc_ll_144_capabilities_and_gaps_are_documented() -> None:
    """Every requirement lists implemented capabilities and honest gaps."""
    for req in NYC_LL_144_REQUIREMENTS.values():
        assert req.rai_capabilities, f"{req.id} has no capabilities"
        assert req.coverage_gaps, f"{req.id} must document coverage gaps"
        # Capability claims must not overstate: no impact-ratio calculation claims
        joined = " ".join(req.rai_capabilities)
        assert "impact ratios" not in joined or "does not" in joined


def test_nyc_ll_144_coverage_partial() -> None:
    """Coverage reflects partial category inclusion correctly."""
    eng = ComplianceMappingEngine()
    profile = eng.create_profile(
        framework=Framework.NYC_LL_144,
        category_ids=["MIT-1.1"],
        name="Partial Coverage Test",
    )
    coverage = eng.get_nyc_ll144_coverage(profile)
    # MIT-1.1 covers half of LL144-1/2 (each needs MIT-1.1 + MIT-1.3)
    assert coverage["NYC-LL144-1"]["coverage_pct"] == 50.0
    assert coverage["NYC-LL144-2"]["coverage_pct"] == 50.0
    # Notice and posting requirements need MIT-5.1 + MIT-7.2
    assert coverage["NYC-LL144-3"]["coverage_pct"] == 0.0
    assert coverage["NYC-LL144-4"]["coverage_pct"] == 0.0
    assert len(coverage["NYC-LL144-3"]["missing_ids"]) == 2


def test_nyc_ll_144_coverage_full() -> None:
    """Full category set yields 100% on every requirement."""
    eng = ComplianceMappingEngine()
    profile = eng.create_profile(
        framework=Framework.NYC_LL_144,
        category_ids=["MIT-1.1", "MIT-1.3", "MIT-5.1", "MIT-7.2"],
        name="Full Coverage Test",
    )
    coverage = eng.get_nyc_ll144_coverage(profile)
    for req_id, cov in coverage.items():
        assert cov["coverage_pct"] == 100.0, f"{req_id} not fully covered"
        assert cov["missing_ids"] == []
        assert cov["coverage_gaps"], f"{req_id} should still surface gaps even at full MIT coverage"


def test_nyc_ll_144_framework_label_format() -> None:
    """Framework row labels render like ``NYC LL 144: ... (Annual bias audit)``."""
    label = format_nyc_ll144_framework_label(
        "NYC-LL144-1",
        section="Section 20-871(a)(1); rule 5-301(a)",
    )
    assert label == "NYC LL 144: Section 20-871(a)(1); rule 5-301(a) (Annual bias audit)"
    fallback = format_nyc_ll144_framework_label("NYC-LL144-1")
    assert fallback == "NYC LL 144: NYC-LL144-1 (Annual bias audit)"


def test_nyc_ll_144_requirement_helpers() -> None:
    """Helper lookups resolve requirement MIT mappings."""
    assert get_mit_categories_for_nyc_requirement("NYC-LL144-1") == ["MIT-1.1", "MIT-1.3"]
    assert get_mit_categories_for_nyc_requirement("NYC-LL144-9") == []
    from rai_toolkit.compliance.nyc_ll144_mapping import get_requirement

    assert get_requirement("NYC-LL144-2") is not None
    assert get_requirement("NYC-LL144-9") is None
