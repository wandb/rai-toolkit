# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional PyRIT integration for the Responsible AI toolkit.

Public import path for PyPI users:

    from integrations.pyrit_integration import run_pyrit_attacks

The core ``rai_toolkit.redteam`` package intentionally does not import PyRIT.
Install with ``pip install rai-toolkit[pyrit]`` before using this integration.
"""

from integrations.pyrit_integration.adapter import (
    PYRIT_INSTALLED,
    PyRITAttackFactory,
    RAIPromptTarget,
    default_attack_factories,
    run_pyrit_attacks,
)

__all__ = [
    "PYRIT_INSTALLED",
    "PyRITAttackFactory",
    "RAIPromptTarget",
    "default_attack_factories",
    "run_pyrit_attacks",
]
