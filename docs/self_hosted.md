# Self-hosted and air-gapped deployment

The core toolkit (`rai_toolkit/`) is vendor-neutral and runs entirely inside your network. This guide covers a fully self-hosted setup, for example HPC clusters with Slurm-scheduled GPU nodes serving models through vLLM. Requested in issue #5.

Install the toolkit in a Python 3.11 or newer environment. When upgrading
from Python 3.10, create a new environment and reinstall the toolkit and the
extras you use. The model-serving process can run separately with its own
runtime and dependency requirements.

## 1. Serve your model behind an OpenAI-compatible API

vLLM exposes an OpenAI-compatible server out of the box:

```bash
python -m vllm.entrypoints.openai.api_server --model <your-model> --port 8000
```

## 2. Point the model adapter at your endpoint

The toolkit's OpenAI-compatible adapter accepts a custom base URL, so the assessment target never leaves your network:

```python
from rai_toolkit.models import OpenAICompatibleModel

model = OpenAICompatibleModel(
    model="<your-model>",
    base_url="http://<internal-host>:8000/v1",
    api_key="not-needed-for-local",
)
```

## 3. Keep LLM-as-judge scoring local

Judge scorers use the same OpenAI-compatible client and accept the same `base_url` override, so scoring stays on your own hardware too. Use a strong local model for judging and calibrate against human-labeled examples (see Limitations in the README).

## 4. Skip the optional cloud extras

Weave tracing is an optional extra. If you do not install `.[weave]`, no tracing dependency is present. Reports are written as local JSON and HTML files.

## 5. No-external-calls checklist

- Model under test served in-network (vLLM or equivalent)
- `base_url` pointing at the internal endpoint for both the model adapter and judge scorers
- Weave extra not installed (or `weave.init()` never called)
- Reference datasets: bundled JSON examples work offline; HuggingFace-streamed loaders (for example HealthBench) require egress, so mirror them internally or use your own datasets
- Reports written to local paths only

If anything in the toolkit attempts a network call outside your endpoints in this configuration, please open an issue; that is treated as a bug.
