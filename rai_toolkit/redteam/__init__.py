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

    print(report.format_summary())
"""

from rai_toolkit.redteam.attacks import (
    ATTACK_CATALOG,
    Attack,
    AttackCategory,
    AttackTemplate,
)
from rai_toolkit.redteam.runner import (
    AttackOutcome,
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
    "AttackOutcome",
    "AttackResult",
    "AttackRunner",
    "FamilyStats",
    "RedTeamReport",
]
