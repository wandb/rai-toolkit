# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for the Responsible AI toolkit.

Entry point: ``rai`` (configured in pyproject.toml).

Usage::

    rai assess <model-ref> --preset healthcare --dataset my-healthcare-eval
    rai datasets list
    rai datasets demo-datasets
    rai policies lint <policies-dir>
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rai",
        description="Responsible AI toolkit — compliance, evaluation, guardrails.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _build_assess_parser(sub)
    _build_datasets_parser(sub)
    _build_policies_parser(sub)
    _build_redteam_parser(sub)

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    return args.func(args)


def _build_assess_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("assess", help="Run a full compliance assessment on a model.")
    p.add_argument("model_ref", help="Python path to a BaseModel subclass, e.g. 'my_pkg.MyModel'.")
    p.add_argument("--preset", default="healthcare", help="Industry preset slug.")
    ds_choice = p.add_mutually_exclusive_group(required=True)
    ds_choice.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset slug. Can be repeated. Required for real assessments.",
    )
    ds_choice.add_argument(
        "--demo-datasets",
        action="store_true",
        help="Use bundled demo/smoke-test dataset bundle for the selected preset.",
    )
    p.add_argument("--policies-dir", default=None, help="Directory of YAML policy files.")
    p.add_argument("--no-redteam", action="store_true", help="Skip the red-team suite.")
    p.add_argument("--limit", type=int, default=None, help="Per-dataset row limit.")
    p.add_argument("--output", default=None, help="Path to write JSON report.")
    p.add_argument("--html", default=None, help="Path to write the HTML report.")
    p.add_argument(
        "--weave-project",
        default=None,
        help="Enable Weave tracing and push traces to this project. "
        "Requires `pip install 'rai-toolkit[weave]'` and WANDB_API_KEY.",
    )
    p.add_argument(
        "--weave-entity",
        default=None,
        help="W&B entity (team or user) for the Weave project. Optional.",
    )
    p.add_argument(
        "--weave-evaluation",
        action="store_true",
        help="Force evaluation via weave.Evaluation (requires weave + usually --weave-project).",
    )
    p.add_argument(
        "--no-weave-evaluation",
        action="store_true",
        help="Disable weave.Evaluation even when --weave-project is set (core pipeline only).",
    )
    p.set_defaults(func=_cmd_assess)


def _build_datasets_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("datasets", help="Dataset registry commands.")
    ds_sub = p.add_subparsers(dest="ds_command", required=True)
    list_p = ds_sub.add_parser("list", help="List registered datasets.")
    list_p.set_defaults(func=_cmd_datasets_list)
    show_p = ds_sub.add_parser("show", help="Show metadata for a dataset.")
    show_p.add_argument("slug")
    show_p.set_defaults(func=_cmd_datasets_show)
    demo_p = ds_sub.add_parser(
        "demo-datasets",
        help="List bundled demo dataset bundles by preset.",
    )
    demo_p.add_argument(
        "--preset",
        choices=["healthcare", "financial_services", "government", "general"],
        default=None,
        help="Only show the demo dataset bundle for one preset.",
    )
    demo_p.set_defaults(func=_cmd_datasets_demo)
    infer_p = ds_sub.add_parser(
        "infer-policy-expectations",
        help="Draft row-level policy_expectations for an existing dataset.",
    )
    infer_p.add_argument("slug", help="Registered dataset slug.")
    infer_p.add_argument("--limit", type=int, default=25, help="Rows to inspect.")
    infer_p.add_argument(
        "--domain",
        default="clinical_triage",
        help="Policy domain vocabulary to draft. Currently supports clinical_triage.",
    )
    infer_p.add_argument("--output", default=None, help="Optional JSON file to write.")
    infer_p.set_defaults(func=_cmd_datasets_infer_policy_expectations)


def _build_policies_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("policies", help="Policy-as-code commands.")
    p_sub = p.add_subparsers(dest="p_command", required=True)
    lint_p = p_sub.add_parser("lint", help="Validate policy files in a directory.")
    lint_p.add_argument("directory")
    lint_p.set_defaults(func=_cmd_policies_lint)


def _build_redteam_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("redteam", help="Red-team suite commands.")
    r_sub = p.add_subparsers(dest="r_command", required=True)
    cat_p = r_sub.add_parser("catalog", help="List attack templates in the catalog.")
    cat_p.set_defaults(func=_cmd_redteam_catalog)


def _cmd_assess(args: argparse.Namespace) -> int:
    from rai_toolkit.assessment import Assessor

    model = _load_model(args.model_ref)
    if args.dataset:
        datasets = args.dataset
    else:
        try:
            datasets = _default_datasets_for(
                args.preset,
                allow_samples=args.demo_datasets,
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    use_weave_eval: bool | None = None
    if args.weave_evaluation and args.no_weave_evaluation:
        print("error: --weave-evaluation and --no-weave-evaluation are mutually exclusive.", file=sys.stderr)
        return 2
    if args.weave_evaluation:
        use_weave_eval = True
    elif args.no_weave_evaluation:
        use_weave_eval = False

    assessor = Assessor(
        model=model,
        preset=args.preset,
        datasets=datasets,
        policies_dir=args.policies_dir,
        run_redteam=not args.no_redteam,
        dataset_limit=args.limit,
        weave_project=args.weave_project,
        weave_entity=args.weave_entity,
        use_weave_evaluation=use_weave_eval,
    )

    result = asyncio.run(assessor.run())
    print(result.format_summary())

    if args.output:
        result.to_json(args.output)
        print(f"Report written to {args.output}")

    if args.html:
        result.to_html(args.html)
        print(f"HTML report written to {args.html}")

    return 0 if result.overall_passed else 1


def _cmd_datasets_list(args: argparse.Namespace) -> int:
    from rai_toolkit.examples import EXAMPLE_CATALOG

    for slug in sorted(EXAMPLE_CATALOG):
        desc = EXAMPLE_CATALOG[slug]
        print(f"{slug:30s}  {desc.risk_category:10s}  {desc.name}")
    return 0


def _cmd_datasets_show(args: argparse.Namespace) -> int:
    from rai_toolkit.examples import ExampleRegistry

    desc = ExampleRegistry.get(args.slug)
    info = {
        "slug": desc.slug,
        "name": desc.name,
        "description": desc.description,
        "risk_category": desc.risk_category,
        "license": desc.license,
        "reference": desc.reference,
        "huggingface_path": desc.huggingface_path,
        "example_file": desc.example_file,
        "default_limit": desc.default_limit,
    }
    print(json.dumps(info, indent=2))
    return 0


def _cmd_datasets_demo(args: argparse.Namespace) -> int:
    from rai_toolkit.examples import DEMO_EXAMPLE_BUNDLES

    if args.preset:
        bundles = {args.preset: DEMO_EXAMPLE_BUNDLES.get(args.preset, [])}
    else:
        bundles = DEMO_EXAMPLE_BUNDLES
    for preset, slugs in sorted(bundles.items()):
        print(f"{preset:22s}  {', '.join(slugs) if slugs else '—'}")
    return 0


def _cmd_datasets_infer_policy_expectations(args: argparse.Namespace) -> int:
    """Draft policy expectations for review.

    This is intentionally a cold-start helper, not an authority. The output
    marks expectations as drafts so users have to review and commit them
    before they become policy-grade evidence.
    """
    from rai_toolkit.examples import ExampleRegistry

    rows = ExampleRegistry.load(args.slug, limit=args.limit)
    drafted = []
    for row in rows:
        out = dict(row)
        if out.get("policy_expectations"):
            drafted.append(out)
            continue
        out["policy_expectations"] = _infer_policy_expectations_for_row(
            out,
            domain=args.domain,
        )
        drafted.append(out)

    payload = json.dumps(drafted, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
        print(f"Draft policy expectations written to {args.output}")
    else:
        print(payload)
    return 0


def _cmd_policies_lint(args: argparse.Namespace) -> int:
    from rai_toolkit.policies.engine import PolicyEngine

    directory = Path(args.directory)
    try:
        engine = PolicyEngine.from_directory(directory)
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    total = len(engine.policies)
    print(f"OK: {total} policies loaded from {directory}")
    for policy in engine.policies:
        print(f"  - {policy.name} ({policy.severity.value})")
    return 0


def _cmd_redteam_catalog(args: argparse.Namespace) -> int:
    from rai_toolkit.redteam.attacks import ATTACK_CATALOG

    for a in ATTACK_CATALOG:
        print(f"{a.id:32s}  [{a.category.value:18s}] sev={a.severity}  {a.name}")
    return 0


def _load_model(model_ref: str):
    """Import a BaseModel subclass from a dotted path.

    Supports two forms:
      - ``pkg.module.ClassName`` — imports and instantiates with no args.
      - ``pkg.module:factory`` — imports a factory callable and calls it.
    """
    if ":" in model_ref:
        module_path, attr = model_ref.split(":", 1)
    elif "." in model_ref:
        module_path, attr = model_ref.rsplit(".", 1)
    else:
        raise ValueError(f"Invalid model ref: {model_ref}")

    module = importlib.import_module(module_path)
    obj = getattr(module, attr)
    return obj() if callable(obj) else obj


def _default_datasets_for(preset: str, *, allow_samples: bool = False) -> list[str]:
    if allow_samples:
        from rai_toolkit.examples import DEMO_EXAMPLE_BUNDLES

        samples = DEMO_EXAMPLE_BUNDLES.get(preset, [])
        if samples:
            return samples
    raise ValueError(
        f"Dataset selection is required for preset `{preset}`. Pass --dataset "
        "for a real assessment, or --demo-datasets for bundled samples."
    )


def _infer_policy_expectations_for_row(row: dict, *, domain: str) -> dict | None:
    if domain != "clinical_triage":
        return None

    text = " ".join(
        str(row.get(key) or "")
        for key in ("input_text", "input", "expected", "context")
    ).lower()
    urgent_terms = (
        "911",
        "emergency",
        "emergent",
        "urgent care",
        "red flag",
        "red-flag",
        "call immediately",
        "seek care",
        "seek medical",
        "doctor",
        "clinician",
        "physician",
        "healthcare provider",
    )
    if not any(term in text for term in urgent_terms):
        return None

    acuity = "urgent" if any(
        term in text
        for term in ("911", "emergency", "emergent", "red flag", "red-flag")
    ) else "clinician"
    severity = "critical" if acuity == "urgent" else "high"
    return {
        "draft": True,
        "generated_by": "rai datasets infer-policy-expectations",
        "domain": "clinical_triage",
        "input_acuity": acuity,
        "required_behavior": ["escalate_to_emergency_or_clinician"],
        "forbidden_behavior": ["recommend_wait_and_see_for_red_flags"],
        "violation_types": ["missing_escalation", "unsafe_reassurance"],
        "severity_if_violated": severity,
        "review_required": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
