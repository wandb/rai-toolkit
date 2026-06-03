# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Adversarial red-team suite.

Ships a catalog of attack templates and an orchestrator that probes any
BaseModel for common failure modes: prompt injection, jailbreaks, PII extraction,
bias probes, and goal hijacking.

Example::

    from rai_toolkit.redteam import AttackRunner, ATTACK_CATALOG

    runner = AttackRunner(model)
    report = await runner.run_all()

    print(f"Attack success rate: {report.overall_success_rate:.1%}")
    for family, stats in report.by_family.items():
        print(f"  {family}: {stats.success_rate:.1%} ({stats.successes}/{stats.total})")
"""

from rai_toolkit.redteam.attacks import (
    ATTACK_CATALOG,
    Attack,
    AttackCategory,
    AttackTemplate,
)
from rai_toolkit.redteam.runner import (
    AttackResult,
    AttackRunner,
    FamilyStats,
    RedTeamReport,
)

__all__ = [
    "ATTACK_CATALOG",
    "Attack",
    "AttackCategory",
    "AttackTemplate",
    "AttackResult",
    "AttackRunner",
    "FamilyStats",
    "RedTeamReport",
]
