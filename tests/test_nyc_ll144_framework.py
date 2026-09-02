# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NYC LL 144 compliance framework integration."""

from __future__ import annotations

from rai_toolkit.compliance.engine import ComplianceMappingEngine
from rai_toolkit.compliance.frameworks import Framework


def test_nyc_ll_144_framework_is_registered() -> None:
    """NYC LL 144 appears in the framework registry."""
    eng = ComplianceMappingEngine()
    frameworks = eng.get_frameworks()
    assert Framework.NYC_LL_144 in frameworks


def test_nyc_ll_144_resolves_mit_categories() -> None:
    """Resolving NYC LL 144 categories returns the expected MIT categories."""
    eng = ComplianceMappingEngine()
    categories = eng.get_categories(Framework.NYC_LL_144)
    ids = {c.id for c in categories}
    # The six requirements cover: MIT-1.1, MIT-1.3, MIT-7.2, MIT-7.1, MIT-2.2
    assert ids == {"MIT-1.1", "MIT-1.3", "MIT-2.2", "MIT-7.1", "MIT-7.2"}


def test_nyc_ll_144_domains_are_structured() -> None:
    """Each NYC LL 144 requirement maps to its MIT category IDs."""
    eng = ComplianceMappingEngine()
    domains = eng.get_domains(Framework.NYC_LL_144)
    assert len(domains) == 6
    for title, mit_ids in domains.items():
        assert isinstance(mit_ids, list)
        assert len(mit_ids) > 0


def test_nyc_ll_144_coverage_partial() -> None:
    """Coverage reflects partial category inclusion correctly."""
    eng = ComplianceMappingEngine()
    profile = eng.create_profile(
        framework=Framework.NYC_LL_144,
        category_ids=["MIT-1.1"],
        name="Partial Coverage Test",
    )
    coverage = eng.get_nyc_ll144_coverage(profile)
    # MIT-1.1 covers LL144-1 and LL144-2 fully; others are partially covered
    assert coverage["NYC-LL144-1"]["coverage_pct"] == 50.0
    assert coverage["NYC-LL144-2"]["coverage_pct"] == 50.0
    assert coverage["NYC-LL144-3"]["coverage_pct"] == 0.0
    assert len(coverage["NYC-LL144-3"]["missing_ids"]) == 1


def test_nyc_ll_144_coverage_full() -> None:
    """Full category set yields 100% on every requirement."""
    eng = ComplianceMappingEngine()
    profile = eng.create_profile(
        framework=Framework.NYC_LL_144,
        category_ids=["MIT-1.1", "MIT-1.3", "MIT-2.2", "MIT-7.1", "MIT-7.2"],
        name="Full Coverage Test",
    )
    coverage = eng.get_nyc_ll144_coverage(profile)
    for req_id, cov in coverage.items():
        assert cov["coverage_pct"] == 100.0, f"{req_id} not fully covered"
        assert cov["missing_ids"] == []
