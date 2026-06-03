# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Garak red-team integration.

The toolkit's first-party red-team API stays under ``rai_toolkit.redteam``;
the bridge to ``NVIDIA/garak`` (formerly ``leondz/garak``) lives here so
the third-party tool boundary is explicit before the PyPI release.

When Garak is installed, this module exposes :func:`run_garak_probes`,
:class:`GarakProbeSpec`, :class:`RAIGenerator`, and
:func:`default_garak_probes`. When it isn't, only ``GARAK_INSTALLED``
exists — slim installs stay importable.
"""

from integrations.garak_integration.adapter import GARAK_INSTALLED

if GARAK_INSTALLED:
    from integrations.garak_integration.adapter import (  # noqa: F401
        GarakProbeSpec,
        RAIGenerator,
        default_garak_probes,
        run_garak_probes,
    )
    __all__ = [
        "GARAK_INSTALLED",
        "GarakProbeSpec",
        "RAIGenerator",
        "default_garak_probes",
        "run_garak_probes",
    ]
else:
    __all__ = ["GARAK_INSTALLED"]
