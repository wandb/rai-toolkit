# RAI Review (Streamlit)

Optional reviewer-facing UI built on `rai_toolkit.workflow`. It wraps the
assessment pipeline in an app-team → RAI-team **review gate**.

## Flow

1. **Intake** — app team fills out an `ApplicationProfile` (industry, data
   types, deployment context, capabilities, model adapter).
2. **Scoping** — `scope_assessor()` derives risk tier (escalating on
   PHI / biometric / credit / legal / `autonomous_action`), validates explicit
   datasets for real runs (or demo fixture bundles in Demo mode), red-team
   severity cap, and policies. Every choice is recorded in a `ScopingDecision`.
3. **Automated assessment** — the scoped `Assessor` pipeline runs and
   produces a `AssessmentResult`.
4. **Auto-recommendation** — `auto_decide()` turns findings into an
   APPROVE / REQUEST_CHANGES / REJECT recommendation with a per-finding
   remediation list.
5. **Interactive probing** — the Review page has a chat panel that talks to the
   *same* model the assessment ran against. The reviewer pins concerning
   turns as a `ManualFinding`; `reconcile_manual_findings()` downgrades the
   auto-recommendation accordingly (e.g. one critical pin alone → REJECT).
6. **Human decision** — reviewer signs off (approve / request_changes / reject)
   with notes. Overrides of the auto-recommendation stay in the record.

## Run

```bash
pip install "rai-toolkit[demo,weave]"
streamlit run demo/rai_review/app.py
```

Run this from the **repository root** (or ensure the repo root is on
`PYTHONPATH`) so `rai_toolkit` and `demo` resolve correctly.

## Persistence and tracing

Submissions live under `rai_workspace/` (JSON profiles, submissions, generated
HTML reports). That directory is **gitignored** in this repo; create it locally
or point `ReviewRegistry` at another path for your deployment.

The intake form has an optional **W&B Weave** toggle. When set, evaluation,
judges, red-team, and reviewer chat probes can stream to your Weave project as
part of the audit trail.

## Model adapters (intake form)

Two ways to point the UI at the system under review:

- **Python class** (`package.module:ClassName` or `package.module:factory`) —
  for code in this repo, including the small reference apps below.
- **OpenAI-compatible endpoint** — base URL, model name, API key, optional
  system prompt (OpenAI, Azure, vLLM, Ollama, LiteLLM, etc.). Implemented in
  `rai_toolkit/models/openai_compatible.py`.

### Bundled reference models (`demo_app/`)

These are **not** part of the PyPI wheel; they exist in this repo so reviewers
can run an end-to-end flow without wiring their own model first:

| Model ref | Preset hint | Notes |
|-----------|---------------|--------|
| `demo_app.triage_assistant:build_model` | `healthcare` | Small healthcare RAG over a tiny on-disk corpus |
| `demo_app.finance_advisor:build_model` | `financial_services` | Small finance RAG over a tiny on-disk corpus |

Both call OpenAI `gpt-4o-mini`, so set `OPENAI_API_KEY` before running — the
assessment is only meaningful on real model output.

## Workflow API (library)

The same types and functions are available without Streamlit — see the main
[README](../../README.md) section **Workflow package (`rai_toolkit.workflow`)**.
