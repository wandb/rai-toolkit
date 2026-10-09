# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Reference examples catalog.

Curated catalog of public RAI evaluation examples. Each entry lazily loads on
demand via HuggingFace datasets, caching locally. All examples are normalized
to the toolkit's standard schema: ``input_text``, ``context``, ``expected``,
``category`` plus optional scorer-specific fields such as ``rubrics`` and
``policy_expectations``.

Example::

    from rai_toolkit.examples import ExampleRegistry

    ds = ExampleRegistry.load("halueval-qa", limit=200)
    # [{"input_text": "...", "context": "...", "expected": "...", "category": "MIT-3.1"}, ...]

    print(ExampleRegistry.list_examples())
"""

from rai_toolkit.examples.registry import (
    EXAMPLE_CATALOG,
    DEMO_EXAMPLE_BUNDLES,
    ExampleDescriptor,
    ExampleRegistry,
)

__all__ = [
    "EXAMPLE_CATALOG",
    "DEMO_EXAMPLE_BUNDLES",
    "ExampleDescriptor",
    "ExampleRegistry",
]
