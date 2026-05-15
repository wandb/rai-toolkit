"""Catalog of public RAI benchmark examples with lazy loaders.

Each example is registered with:
  - A ``loader`` callable that returns a list of normalized rows.
  - A ``risk_category`` indicating what this dataset primarily tests.
  - Metadata: description, license, reference, huggingface path.

Loaders are intentionally thin. They either pull from HuggingFace datasets or
load a locally-shipped JSON example for small curated samples.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen

logger = logging.getLogger(__name__)

_EXAMPLES_DIR = Path(__file__).parent
_ADV_BENCH_HARMFUL_BEHAVIORS_CSV = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)
_BBQ_SPLITS = (
    "age",
    "disabilityStatus",
    "genderIdentity",
    "nationality",
    "physicalAppearance",
    "raceEthnicity",
    "raceXSes",
    "raceXGender",
    "religion",
    "ses",
    "sexualOrientation",
)


@dataclass(init=False)
class ExampleDescriptor:
    """Metadata + loader for a registered example."""

    slug: str
    name: str
    description: str
    risk_category: str
    license: str
    reference: str
    huggingface_path: str | None = None
    example_file: str | None = None
    default_limit: int = 100
    loader: Callable[[int], list[dict[str, Any]]] | None = field(default=None, repr=False)

    def __init__(
        self,
        *,
        slug: str,
        name: str,
        description: str,
        risk_category: str,
        license: str,
        reference: str,
        huggingface_path: str | None = None,
        example_file: str | None = None,
        fixture_file: str | None = None,
        default_limit: int = 100,
        loader: Callable[[int], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.slug = slug
        self.name = name
        self.description = description
        self.risk_category = risk_category
        self.license = license
        self.reference = reference
        self.huggingface_path = huggingface_path
        self.example_file = example_file if example_file is not None else fixture_file
        self.default_limit = default_limit
        self.loader = loader

    @property
    def fixture_file(self) -> str | None:
        """Backward-compatible alias for older callers."""
        return self.example_file


class ExampleRegistry:
    """Central entry point for loading reference examples."""

    @staticmethod
    def list_examples() -> list[str]:
        return sorted(EXAMPLE_CATALOG.keys())

    @staticmethod
    def list_datasets() -> list[str]:
        """Backward-compatible alias for older callers."""
        return ExampleRegistry.list_examples()

    @staticmethod
    def get(slug: str) -> ExampleDescriptor:
        if slug not in EXAMPLE_CATALOG:
            close = _suggest(slug)
            msg = f"Example dataset '{slug}' not registered."
            if close:
                msg += f" Did you mean: {', '.join(close)}?"
            msg += f" Available: {', '.join(ExampleRegistry.list_examples())}"
            raise KeyError(msg)
        return EXAMPLE_CATALOG[slug]

    @staticmethod
    def load(slug: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Load a dataset by slug, normalized to the standard schema.

        Args:
            slug: Registered dataset identifier (see ``list_datasets()``).
            limit: Maximum number of rows to return. Defaults to the descriptor's
                ``default_limit``.

        Returns:
            List of dicts with keys ``input_text``, ``context``, ``expected``,
            and ``category``.
        """
        desc = ExampleRegistry.get(slug)
        n = limit if limit is not None else desc.default_limit

        if desc.loader is not None:
            rows = desc.loader(n)
        elif desc.example_file:
            rows = _load_example_file(desc.example_file, limit=n)
        elif desc.huggingface_path:
            rows = _load_huggingface(desc.huggingface_path, limit=n)
        else:
            raise RuntimeError(f"Dataset '{slug}' has no loader configured.")

        return [_normalize_row(r, default_category=desc.risk_category) for r in rows]


def _normalize_row(row: dict[str, Any], default_category: str) -> dict[str, Any]:
    """Normalize loader output to the standard schema.

    ``rubrics`` and ``policy_expectations`` are preserved when present.
    Scorers consume rubrics; the assessment layer consumes
    ``policy_expectations`` to decide whether policy violations are
    assessable for a row.
    """
    out: dict[str, Any] = {
        "input_text": str(row.get("input_text") or row.get("question") or row.get("prompt") or ""),
        "context": str(row.get("context") or row.get("passage") or ""),
        "expected": str(row.get("expected") or row.get("answer") or row.get("label") or ""),
        "category": str(row.get("category") or default_category),
    }
    rubrics = row.get("rubrics")
    if rubrics:
        out["rubrics"] = rubrics
    if "policy_expectations" in row:
        out["policy_expectations"] = row.get("policy_expectations")
    return out


def _load_example_file(filename: str, limit: int) -> list[dict[str, Any]]:
    path = _EXAMPLES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Example file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Example file {path} must be a JSON array")
    return rows[:limit]


_load_fixture = _load_example_file


def _load_huggingface(path: str, limit: int) -> list[dict[str, Any]]:
    """Load from HuggingFace datasets. Requires `pip install datasets`."""
    load_dataset = _require_datasets()

    logger.info("Loading %s from HuggingFace...", path)
    parts = path.split(":", 1)
    ds_name = parts[0]
    config = parts[1] if len(parts) > 1 else None

    ds = load_dataset(ds_name, config, split="train", streaming=True)
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        rows.append(dict(row))
    return rows


def _require_datasets() -> Callable[..., Iterable[Any]]:
    """Return ``datasets.load_dataset`` with a consistent optional-extra error."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "huggingface datasets is required. Install with `pip install datasets`."
        ) from e
    return load_dataset


def _halueval_qa_loader(limit: int) -> list[dict[str, Any]]:
    """Load HaluEval QA rows from the current HF config/split.

    The generic loader used to request ``split='train'`` and then normalize the
    wrong fields. HaluEval's QA config exposes a ``data`` split with
    ``knowledge``, ``question``, ``right_answer`` and ``hallucinated_answer``.
    """
    load_dataset = _require_datasets()
    ds = load_dataset("pminervini/HaluEval", "qa", split="data", streaming=True)
    rows: list[dict[str, Any]] = []
    for row in ds:
        question = str(row.get("question") or "").strip()
        expected = str(row.get("right_answer") or row.get("answer") or "").strip()
        if not question or not expected:
            continue
        rows.append({
            "input_text": question,
            "context": str(row.get("knowledge") or "").strip(),
            "expected": expected,
            "category": "MIT-3.1",
        })
        if len(rows) >= limit:
            break
    return rows


def _truthfulqa_generation_loader(limit: int) -> list[dict[str, Any]]:
    """Load TruthfulQA generation rows.

    TruthfulQA is not a retrieval-grounded dataset. To make it usable by the
    toolkit's grounding-oriented factuality scorer, we put the accepted answer
    set in ``context`` as reference answers and the canonical ``best_answer``
    in ``expected``.
    """
    load_dataset = _require_datasets()
    ds = load_dataset(
        "truthfulqa/truthful_qa",
        "generation",
        split="validation",
        streaming=True,
    )
    rows: list[dict[str, Any]] = []
    for row in ds:
        question = str(row.get("question") or "").strip()
        best_answer = str(row.get("best_answer") or "").strip()
        correct_answers = [
            str(a).strip() for a in row.get("correct_answers") or [] if str(a).strip()
        ]
        if not question or not best_answer:
            continue
        if not correct_answers:
            correct_answers = [best_answer]
        context = "Accepted truthful answers:\n" + "\n".join(f"- {a}" for a in correct_answers)
        rows.append({
            "input_text": question,
            "context": context,
            "expected": best_answer,
            "category": "MIT-3.1",
        })
        if len(rows) >= limit:
            break
    return rows


def _bbq_loader(limit: int) -> list[dict[str, Any]]:
    """Load BBQ from an accessible parquet-backed mirror.

    ``heegyu/bbq`` now requires deprecated dataset scripts. ``walledai/BBQ``
    exposes one split per social dimension with a simple multiple-choice
    schema: context, question, choices, answer, category.
    """
    load_dataset = _require_datasets()
    rows: list[dict[str, Any]] = []
    for split in _BBQ_SPLITS:
        ds = load_dataset("walledai/BBQ", split=split, streaming=True)
        for row in ds:
            choices = [str(c).strip() for c in row.get("choices") or []]
            try:
                answer_idx = int(row.get("answer"))
            except (TypeError, ValueError):
                answer_idx = -1
            expected = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""
            context = str(row.get("context") or "").strip()
            question = str(row.get("question") or "").strip()
            if not question or not choices:
                continue
            choice_lines = "\n".join(
                f"{i}. {choice}" for i, choice in enumerate(choices)
            )
            rows.append({
                "input_text": (
                    f"{context}\n\nQuestion: {question}\n\nChoices:\n{choice_lines}"
                ).strip(),
                "context": context,
                "expected": expected,
                "category": "MIT-1.1",
            })
            if len(rows) >= limit:
                return rows
    return rows


def _bold_loader(limit: int) -> list[dict[str, Any]]:
    """Load BOLD generation prompts.

    BOLD rows contain ``prompts`` as a list; the generic normalizer looked for
    ``prompt`` and produced blank inputs while overwriting MIT-1.1 with BOLD's
    internal category label. This loader expands prompt lists and keeps the RAI
    category fixed to fairness/bias.
    """
    load_dataset = _require_datasets()
    ds = load_dataset("AmazonScience/bold", split="train", streaming=True)
    rows: list[dict[str, Any]] = []
    for row in ds:
        for prompt in row.get("prompts") or []:
            prompt_text = str(prompt or "").strip()
            if not prompt_text:
                continue
            rows.append({
                "input_text": prompt_text,
                "context": "",
                "expected": "",
                "category": "MIT-1.1",
            })
            if len(rows) >= limit:
                return rows
    return rows


def _advbench_harmful_loader(limit: int) -> list[dict[str, Any]]:
    """Load AdvBench harmful behaviors from the original public CSV.

    The HF mirror is gated, so the catalog uses the upstream GitHub CSV to keep
    this slug usable without Hugging Face authentication.
    """
    rows: list[dict[str, Any]] = []
    with urlopen(_ADV_BENCH_HARMFUL_BEHAVIORS_CSV, timeout=30) as response:
        text = response.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    for raw in reader:
        goal = str(
            raw.get("goal")
            or raw.get("prompt")
            or raw.get("behavior")
            or raw.get("instruction")
            or ""
        ).strip()
        if not goal:
            values = [str(v).strip() for v in raw.values() if str(v).strip()]
            goal = values[0] if values else ""
        if not goal:
            continue
        rows.append({
            "input_text": goal,
            "context": "",
            "expected": "Refuse to comply with the harmful request.",
            "category": "MIT-4.1",
        })
        if len(rows) >= limit:
            break
    return rows


def _jailbreakbench_harmful_loader(limit: int) -> list[dict[str, Any]]:
    """Load JBB harmful behavior prompts.

    JailbreakBench is a compact, open robustness benchmark that includes
    behaviors sourced from AdvBench and HarmBench, with stable category/source
    metadata and no gated download.
    """
    load_dataset = _require_datasets()
    ds = load_dataset(
        "JailbreakBench/JBB-Behaviors",
        "behaviors",
        split="harmful",
        streaming=True,
    )
    rows: list[dict[str, Any]] = []
    for row in ds:
        goal = str(row.get("Goal") or "").strip()
        if not goal:
            continue
        metadata = [
            f"Behavior: {row.get('Behavior')}" if row.get("Behavior") else "",
            f"Misuse category: {row.get('Category')}" if row.get("Category") else "",
            f"Source: {row.get('Source')}" if row.get("Source") else "",
        ]
        rows.append({
            "input_text": goal,
            "context": "\n".join(part for part in metadata if part),
            "expected": "Refuse to comply with the harmful request.",
            "category": "MIT-4.1",
        })
        if len(rows) >= limit:
            break
    return rows


def _finqa_verified_loader(limit: int) -> list[dict[str, Any]]:
    """Load a small verified financial QA set with direct context/question/answer."""
    load_dataset = _require_datasets()
    ds = load_dataset("Aiera/finqa-verified", split="test", streaming=True)
    rows: list[dict[str, Any]] = []
    for row in ds:
        question = str(row.get("question") or "").strip()
        context = str(row.get("context") or "").strip()
        expected = _stringify_answer(row.get("answer"))
        if not question or not context or not expected:
            continue
        rows.append({
            "input_text": question,
            "context": context,
            "expected": expected,
            "category": "MIT-3.1",
        })
        if len(rows) >= limit:
            break
    return rows


def _tatqa_finance_loader(limit: int) -> list[dict[str, Any]]:
    """Load TAT-QA validation questions over hybrid table/text financial context."""
    load_dataset = _require_datasets()
    ds = load_dataset("next-tat/TAT-QA", split="validation", streaming=True)
    rows: list[dict[str, Any]] = []
    for row in ds:
        context = _render_tatqa_context(row)
        for q in row.get("questions") or []:
            question = str(q.get("question") or "").strip()
            expected = _stringify_answer(q.get("answer"))
            if not question or not expected:
                continue
            rows.append({
                "input_text": question,
                "context": context,
                "expected": expected,
                "category": "MIT-3.1",
            })
            if len(rows) >= limit:
                return rows
    return rows


def _render_tatqa_context(row: dict[str, Any]) -> str:
    """Render TAT-QA's table + paragraphs into a compact text context."""
    parts: list[str] = []
    table = row.get("table") or {}
    table_rows = table.get("table") if isinstance(table, dict) else None
    if table_rows:
        parts.append("Table:\n" + _render_table(table_rows))
    paragraphs = row.get("paragraphs") or []
    para_texts: list[str] = []
    for paragraph in paragraphs:
        if isinstance(paragraph, dict):
            text = str(paragraph.get("text") or "").strip()
        else:
            text = str(paragraph or "").strip()
        if text:
            para_texts.append(text)
    if para_texts:
        parts.append("Paragraphs:\n" + "\n".join(para_texts))
    return "\n\n".join(parts)


def _render_table(table_rows: Any) -> str:
    rendered: list[str] = []
    for row in table_rows or []:
        if isinstance(row, list):
            rendered.append(" | ".join(str(cell).strip() for cell in row))
        else:
            rendered.append(str(row).strip())
    return "\n".join(line for line in rendered if line)


def _stringify_answer(answer: Any) -> str:
    if isinstance(answer, list):
        return ", ".join(str(item).strip() for item in answer if str(item).strip())
    if answer is None:
        return ""
    return str(answer).strip()


def _healthbench_triage_loader(limit: int) -> list[dict[str, Any]]:
    """Stream HealthBench (OpenAI), filter to the ``emergency_referrals`` theme.

    HealthBench is rubric-graded — each prompt has physician-written criteria,
    not a single gold answer. We preserve those criteria in ``rubrics`` and use
    ``ideal_completions_data.ideal_completion`` (populated for ~80% of rows) as
    the toolkit's compatibility ``expected`` field, skipping rows where it is
    missing.

    HealthBench prompts have no retrieval context, so ``context`` is empty.
    The toolkit's ``FactualityJudge`` will mark these rows as un-assessed —
    that is the honest answer for a benchmark that does not include grounding
    evidence.

    Reference: https://arxiv.org/abs/2505.08775 (HealthBench, OpenAI 2025).
    License: MIT. Source: https://huggingface.co/datasets/openai/healthbench.
    """
    load_dataset = _require_datasets()
    logger.info("Loading openai/healthbench (theme:emergency_referrals)...")
    ds = load_dataset("openai/healthbench", split="test", streaming=True)
    rows: list[dict[str, Any]] = []
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > 5000:  # bail out if the theme is unexpectedly absent
            break
        tags = row.get("example_tags") or []
        if "theme:emergency_referrals" not in tags:
            continue
        ideal = row.get("ideal_completions_data")
        if not ideal or not ideal.get("ideal_completion"):
            continue
        prompt_msgs = row.get("prompt") or []
        input_text = _render_chat_prompt(prompt_msgs)
        if not input_text:
            continue
        rubrics = row.get("rubrics") or []
        rows.append({
            "input_text": input_text,
            "context": "",
            "expected": str(ideal["ideal_completion"]),
            "category": "MIT-3.1",
            "rubrics": [
                {
                    "criterion": str(r.get("criterion") or ""),
                    "points": int(r.get("points") or 0),
                    "tags": list(r.get("tags") or []),
                }
                for r in rubrics
                if r.get("criterion")
            ],
        })
        if len(rows) >= limit:
            break
    logger.info("Loaded %d HealthBench triage rows (scanned %d).", len(rows), scanned)
    return rows


def _render_chat_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten a HealthBench ``prompt`` (list of ``{role, content}``) into text.

    Single-turn user prompts are returned as-is. Multi-turn prompts are
    rendered as ``User: ...\\nAssistant: ...\\nUser: ...`` so the last user
    turn carries its conversational history into ``input_text``.
    """
    if not messages:
        return ""
    if len(messages) == 1 and messages[0].get("role") == "user":
        return str(messages[0].get("content") or "").strip()
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or "").strip().capitalize() or "User"
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _suggest(slug: str) -> list[str]:
    """Cheap fuzzy match for helpful KeyError messages."""
    slug_l = slug.lower()
    return [s for s in EXAMPLE_CATALOG if slug_l in s or s in slug_l][:3]


EXAMPLE_CATALOG: dict[str, ExampleDescriptor] = {
    "halueval-qa": ExampleDescriptor(
        slug="halueval-qa",
        name="HaluEval (QA subset)",
        description="Hallucination benchmark: questions where one of two answers is hallucinated.",
        risk_category="MIT-3.1",
        license="Apache-2.0",
        reference="https://arxiv.org/abs/2305.11747",
        huggingface_path="pminervini/HaluEval:qa",
        loader=_halueval_qa_loader,
        default_limit=100,
    ),
    "truthfulqa-gen": ExampleDescriptor(
        slug="truthfulqa-gen",
        name="TruthfulQA (generation)",
        description="Questions designed to test model truthfulness on topics humans typically get wrong.",
        risk_category="MIT-3.1",
        license="Apache-2.0",
        reference="https://arxiv.org/abs/2109.07958",
        huggingface_path="truthfulqa/truthful_qa:generation",
        loader=_truthfulqa_generation_loader,
        default_limit=100,
    ),
    "bbq": ExampleDescriptor(
        slug="bbq",
        name="BBQ (Bias Benchmark for QA)",
        description="Measures social bias in QA systems across 9 demographic axes.",
        risk_category="MIT-1.1",
        license="CC-BY-4.0",
        reference="https://arxiv.org/abs/2110.08193",
        huggingface_path="walledai/BBQ",
        loader=_bbq_loader,
        default_limit=100,
    ),
    "bold": ExampleDescriptor(
        slug="bold",
        name="BOLD (Bias in Open-ended Language Generation)",
        description="Evaluates bias across 5 domains: race, gender, religion, profession, ideology.",
        risk_category="MIT-1.1",
        license="CC-BY-SA-4.0",
        reference="https://arxiv.org/abs/2101.11718",
        huggingface_path="AmazonScience/bold",
        loader=_bold_loader,
        default_limit=100,
    ),
    "advbench-harmful": ExampleDescriptor(
        slug="advbench-harmful",
        name="AdvBench (harmful behaviors)",
        description=(
            "Adversarial prompts eliciting harmful behaviors. Loaded from the "
            "upstream public CSV because the Hugging Face mirror is gated."
        ),
        risk_category="MIT-4.1",
        license="MIT",
        reference="https://arxiv.org/abs/2307.15043",
        loader=_advbench_harmful_loader,
        default_limit=100,
    ),
    "jailbreakbench-harmful": ExampleDescriptor(
        slug="jailbreakbench-harmful",
        name="JailbreakBench harmful behaviors",
        description=(
            "Open robustness benchmark with 100 harmful behavior goals "
            "covering misuse categories sourced from AdvBench, HarmBench/TDC, "
            "and original JailbreakBench examples."
        ),
        risk_category="MIT-4.1",
        license="MIT",
        reference="https://arxiv.org/abs/2404.01318",
        huggingface_path="JailbreakBench/JBB-Behaviors:behaviors",
        loader=_jailbreakbench_harmful_loader,
        default_limit=100,
    ),
    "healthbench-triage": ExampleDescriptor(
        slug="healthbench-triage",
        name="HealthBench (emergency-referrals subset, OpenAI)",
        description=(
            "Physician-rubric-graded medical conversations from OpenAI's "
            "HealthBench, filtered to the `theme:emergency_referrals` "
            "subset (the closest match to triage). Loader uses "
            "`ideal_completions_data.ideal_completion` as the gold answer "
            "(populated for ~80% of rows) and skips rows without one. "
            "HealthBench prompts include no retrieval context, so "
            "FactualityJudge will mark rows as un-assessed; RubricScorer "
            "uses the preserved per-row HealthBench criteria."
        ),
        risk_category="MIT-3.1",
        license="MIT",
        reference="https://arxiv.org/abs/2505.08775",
        huggingface_path="openai/healthbench",
        loader=_healthbench_triage_loader,
        default_limit=20,
    ),
    "clinical-triage-policy-examples": ExampleDescriptor(
        slug="clinical-triage-policy-examples",
        name="Clinical triage policy expectations (toolkit example)",
        description=(
            "Tiny toolkit-authored example showing row-level "
            "policy_expectations for clinical triage. Use this as an "
            "authoring example for policy-ready datasets; it is not a "
            "clinical benchmark."
        ),
        risk_category="MIT-3.1",
        license="Apache-2.0",
        reference="Toolkit-authored policy_expectations example",
        example_file="clinical_triage_policy_expectations.json",
        default_limit=3,
    ),
    "finqa-sample": ExampleDescriptor(
        slug="finqa-sample",
        name="FinQA (curated sample)",
        description="Financial reasoning questions grounded in earnings reports.",
        risk_category="MIT-3.1",
        license="MIT (derived)",
        reference="https://arxiv.org/abs/2109.00122",
        example_file="finqa_sample.json",
        default_limit=10,
    ),
    "finqa-verified": ExampleDescriptor(
        slug="finqa-verified",
        name="FinQA verified subset",
        description=(
            "Small verified financial QA set with real filing context, "
            "question, and numeric answer fields."
        ),
        risk_category="MIT-3.1",
        license="MIT",
        reference="https://huggingface.co/datasets/Aiera/finqa-verified",
        huggingface_path="Aiera/finqa-verified",
        loader=_finqa_verified_loader,
        default_limit=50,
    ),
    "tatqa-finance": ExampleDescriptor(
        slug="tatqa-finance",
        name="TAT-QA (finance table/text QA)",
        description=(
            "Financial question answering over hybrid table and paragraph "
            "contexts from real reports, requiring numerical reasoning."
        ),
        risk_category="MIT-3.1",
        license="CC-BY-4.0",
        reference="https://arxiv.org/abs/2105.07624",
        huggingface_path="next-tat/TAT-QA",
        loader=_tatqa_finance_loader,
        default_limit=50,
    ),
    "pii-extraction-probes": ExampleDescriptor(
        slug="pii-extraction-probes",
        name="PII extraction probes (toolkit sample)",
        description="Curated probes targeting PII leakage via direct asks and repetition attacks.",
        risk_category="MIT-2.1",
        license="Apache-2.0",
        reference="https://arxiv.org/abs/2311.17035",
        example_file="pii_probes.json",
        default_limit=15,
    ),
    "mit-risk-smoke": ExampleDescriptor(
        slug="mit-risk-smoke",
        name="MIT Risk Smoke Test (toolkit sample)",
        description="Small cross-category smoke test covering 7 MIT AI Risk domains. Useful for quick assessment dry-runs during development.",
        risk_category="MIT-MIXED",
        license="Apache-2.0",
        reference="Toolkit-authored curated set",
        example_file="mit_risk_smoke.json",
        default_limit=8,
    ),
}


DEMO_EXAMPLE_BUNDLES: dict[str, list[str]] = {
    "healthcare": ["healthbench-triage", "pii-extraction-probes"],
    "financial_services": ["finqa-sample"],
    "government": ["mit-risk-smoke"],
    "general": ["mit-risk-smoke"],
}

