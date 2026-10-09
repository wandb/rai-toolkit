# Model adapters

A model adapter is the seam between the toolkit and whatever system actually answers a prompt. Everything downstream — evaluation, red-teaming, guardrails, the assessment report — talks to a `BaseModel` and never to a provider SDK. That only holds if every adapter behaves the same way, so this page is the contract adapters are held to.

The provider adapter contract has an executable form: `tests/model_adapter_contract.py`. A dedicated provider adapter is not finished until it passes that suite.

The provider request, channel, and standard metadata requirements below describe
dedicated provider adapters, including `OpenAICompatibleModel` and
`AnthropicModel`. `CallableModel` is a forwarding wrapper with the narrower
contract described next; it cannot enforce what an arbitrary function sends to
its provider.

## Wrapping a callable

`CallableModel` invokes `predict_fn(input_text, context=context, **kwargs)` with
the caller's values unchanged. It awaits coroutine functions, runs synchronous
functions on a worker thread, and awaits an awaitable returned by a synchronous
function. Exceptions propagate. The function must return a string or a
`ModelResponse`; other return types raise `TypeError`.

The wrapper does not serialize retrieved context, append a context handling
policy, choose message roles, or validate provider-specific options. The wrapped
function is responsible for those choices and for delivering context to the
model. If it uses retrieved context, keep that data out of system instructions
and other privileged channels. Test the actual provider request constructed by
the function, including poisoned context and empty-context behavior. The
provider adapter conformance suite does not establish those properties for an
arbitrary callable.

Document the callable's prompt construction, context handling, model settings,
and retrieval configuration alongside its assessment results. An injection test
of that callable measures the complete implementation, including its choice of
trust boundaries. It is not an adapter-only comparison with the built-ins unless
the relevant request construction and assessment conditions are aligned.

A returned `ModelResponse` is preserved, including its metadata. A returned
string is wrapped with empty metadata; `CallableModel` does not synthesize the
six provider metadata keys below. Populate them in your function when known. If
the function retrieves its own context, return the actual context used under
`metadata["retrieved_context"]` so evaluation can use it instead of the dataset's
context hint.

## When you need a dedicated adapter

Start with `OpenAICompatibleModel`. It speaks the OpenAI chat-completions protocol, which is what the public OpenAI API, Azure, vLLM, Ollama, LiteLLM, and most internal proxies expose. Pointing it at a `base_url` costs nothing and adds no code to maintain.

Write a dedicated adapter only when the provider's wire protocol genuinely differs and the difference is visible in the contract below. `AnthropicModel` exists because the Messages API carries system instructions in a top-level `system` parameter rather than a message, returns a list of content blocks rather than a single string, and names its token counts `input_tokens` / `output_tokens`. Each of those is a real translation, not a preference.

Reasons that do **not** justify a new adapter: a different default model id, a different system prompt, a different temperature, or a convenience wrapper around one that already exists. Those are configuration.

## Subclassing `BaseModel`

```python
from rai_toolkit.models.base import BaseModel, ModelResponse


class MyProviderModel(BaseModel):
    def __init__(self, model: str, api_key: str | None = None, name: str | None = None):
        super().__init__(name=name or model)
        self.model = model
        self.api_key = api_key
        self._client = None

    async def predict(self, input_text, context="", **kwargs) -> ModelResponse:
        ...
```

Call `super().__init__()` with the display name. `BaseModel` falls back to the class name when no name is given, which is rarely what a report reader wants to see; pass the model id instead.

Do not decorate `predict` yourself. `BaseModel.__init_subclass__` wraps every subclass's `predict` with the toolkit's tracing op (see [Tracing](#tracing)).

## The `predict` contract

```python
async def predict(self, input_text: str, context: str = "", **kwargs) -> ModelResponse
```

**`input_text` is the user's input and must not be silently rewritten.** Do not trim it, truncate it, prefix it, template it, or fold it into a longer instruction. Red-team and evaluation results are only meaningful if the string that was scored is the string that was sent. An adapter that needs standing instructions puts them in `system_prompt`, which is configuration the caller chose, not a rewrite of their input.

With an empty `context` — the default — that means `input_text` is passed verbatim: it reaches the provider once, unchanged, as the user content string. When a caller supplies `context`, `input_text` is not altered either, but it is encapsulated as one field of the JSON user payload described in [Retrieved context](#retrieved-context); it round-trips out of that payload as the exact string the caller passed.

**A non-empty `context` reaches the provider exactly once**, and never through a privileged channel. RAG evaluation depends on the retrieved passage actually being in the request; groundedness scoring depends on it being there once, not duplicated across a system block and a user turn. See below for where it goes.

**An empty `context` must not create a synthetic context block.** `context=""` is the default and means "no retrieval happened". Emitting an empty `retrieved_context` field, or an empty "Retrieved context:" block, tells the model a retrieval returned nothing, which is a different claim from not having retrieved at all, and it changes behaviour.

**`**kwargs` carries per-call overrides** of the adapter's own configured defaults. The built-in adapters accept only the options in this table:

| Adapter | Supported call-time options | Validation and request behaviour |
| --- | --- | --- |
| `OpenAICompatibleModel` | `temperature`, `max_tokens` | `max_tokens` must be a positive Python integer. It is omitted from the provider request when the caller does not supply it, preserving the provider default. |
| `AnthropicModel` | `max_tokens` | A positive Python integer is the portable form. The adapter also preserves its existing normalization of integer strings, `None`, and blank strings for compatibility. |

Every other call-time option raises `TypeError` before a provider request is made. The error names the adapter and lists every unsupported key in sorted order, so misspellings such as `max_token` cannot look successful while doing nothing. Adapter-controlled fields such as `model`, `messages`, `system`, and retrieved-context serialization cannot be overridden through `**kwargs`.

This is an intentional compatibility change for callers that relied on ignored keyword arguments: those calls now fail clearly. Provider SDK parameters are not forwarded automatically; adding another supported option requires an explicit adapter contract, validation, documentation, and tests.

`predict` returns a `ModelResponse`, never a bare string, `None`, or a provider object.

## Retrieved context

Retrieved context is attacker-reachable data. A retrieval index can be poisoned, a scraped page can carry "ignore your instructions", and a red-team run exists precisely to send such passages. So context never enters the system prompt, the Anthropic top-level `system` parameter, or any other channel the model is entitled to read as instructions. It travels in the user turn, structurally separated from the caller's own input.

**When `context` is non-empty**, the user message is a JSON object with exactly two keys:

```json
{"retrieved_context": "<the passage the retriever returned>", "input_text": "<the caller's input>"}
```

Both values round-trip: JSON encoding keeps quotes, braces, and newlines inside the retrieved passage from running together with the caller's input, so a scorer reading the request back gets the same two strings the caller passed.

**When context is present, a fixed context handling policy is appended to the system instructions.** `rai_toolkit/models/_prompting.build_prompt_parts` owns both halves — it builds the JSON payload and appends the policy — so every built-in adapter states the same thing. The policy names the two fields, marks `retrieved_context` as untrusted reference data rather than instructions, and tells the model to answer `input_text`. It is appended to the caller's `system_prompt` when there is one and used on its own when there is not. Adapters do not reword it; a per-adapter variant would mean two adapters answering the same poisoned passage differently.

**When `context` is empty**, none of this applies: no JSON wrapper, no policy sentence, and the user content is the bare `input_text` string. A request built with `context=""` stays byte-identical to one built with the argument omitted.

This structure is defense in depth, not a guarantee. It removes one specific and well-understood failure — retrieved text arriving in a privileged channel, or blurring into the user's own question across an ad-hoc `"Retrieved context:"` delimiter — and it gives the model a trusted statement about which field is which. It does not make prompt injection impossible: a model can still be talked into following instructions inside a JSON string value. Treat it as raising the cost of an injection, and keep the guardrail and red-team layers doing their own work rather than relying on adapter-level framing to stop attacks outright.

## Output fidelity

`ModelResponse.output` is the provider's text, in the order the provider returned it, and nothing else.

- Concatenate multi-part responses with `"".join(...)`. Do not insert newlines, spaces, or separators the provider did not send — a scorer cannot tell an invented separator from model output.
- Do not reorder, deduplicate, or filter text segments.
- Drop non-text parts (tool-use blocks, thinking blocks) rather than stringifying them into `output`. If they matter, surface them under a provider-specific metadata key.
- When the provider returns no text, `output` is `""`. Do not substitute a placeholder, an apology, or an error string.

## Standard metadata keys

Every dedicated provider adapter populates these six keys on `ModelResponse.metadata`:

| Key | Meaning |
| --- | --- |
| `model` | The model identifier sent to the provider |
| `base_url` | The endpoint override in use, or `None` for the provider default |
| `finish_reason` | Why generation stopped, in the provider's own vocabulary |
| `prompt_tokens` | Input tokens billed |
| `completion_tokens` | Output tokens billed |
| `total_tokens` | Sum of the two, or the provider's own total |

The keys are always present. Populate a key when the provider supplies the value, and set it to `None` when it does not — an absent key and a `None` value mean different things to report rendering, and the toolkit relies on the second. Read them defensively (`getattr(usage, "prompt_tokens", None)`); providers omit usage on some responses.

Do not rename these keys to match a provider's vocabulary. `AnthropicModel` maps `input_tokens` to `prompt_tokens` and `stop_reason` to `finish_reason` precisely so downstream code never learns which provider answered. `finish_reason` is the one value passed through unchanged, because it is descriptive rather than structural.

Provider-specific additions are allowed alongside the standard keys as long as they are documented in the adapter's docstring and never replace a standard key.

## Display name

`name` is a stable, non-empty string used in reports and logs. Set it once at construction and never mutate it during a call — a name that changes mid-run breaks the grouping of results. Default it to the model id.

## Credentials

An API key never appears in `repr`, in `ModelResponse.metadata`, or in an error message the adapter raises. `BaseModel.__repr__` renders only the class name and display name, so the safe thing is to leave it alone; if an adapter overrides `__repr__`, it must not add credential fields.

Storing the key as an instance attribute is fine — `AnthropicModel` does, because it creates its client lazily — but it stays out of anything user-visible. When an adapter raises a configuration error, name the field, not the value: `"api_key must not be empty"`, never the key itself.

## Error handling

**Transport and authentication failures propagate.** Do not catch a provider's timeout, rate-limit, or 401 and turn it into an empty `ModelResponse` or a generic exception. An evaluation that silently scores a swallowed error as a model answer is worse than one that fails. Retries, backoff, and universal error types are out of scope for adapters.

**Invalid configuration is rejected early**, in the constructor or in `from_args`, with a message that names the offending field. `AnthropicModel._coerce_max_tokens` is the model to follow: it normalizes `None` and blank strings to the documented default and raises `ValueError` naming `max_tokens` for anything else. Failing at construction surfaces the problem before a run starts rather than partway through it.

## Tracing

`BaseModel.__init_subclass__` wraps each subclass's `predict` with `_tracing.traced(name="rai.model.predict", kind="llm")`. Every adapter therefore emits exactly one span, under one op name, with one flattened output shape — regardless of whether it was called from the eval pipeline, a chat probe, or a red-team run.

Do not add a second wrapper. Decorating `predict` with `@_tracing.traced` yourself sets the `__rai_traced_op_name__` sentinel, which `__init_subclass__` honours by skipping its own wrap; that is the escape hatch for classes that genuinely need a different op name (`GuardedModel` uses `rai.guardrails.predict`), not something a provider adapter wants. A plain adapter should leave `predict` undecorated and let the base class do it.

## Optional SDKs and lazy imports

`import rai_toolkit` and `import rai_toolkit.models` must keep working in an environment that has none of the optional provider packages installed. That is a hard constraint: `rai_toolkit/models/__init__.py` imports every adapter, so a top-level `import some_sdk` in an adapter module would break the whole package for everyone.

So:

1. Declare the SDK in a named extra in `pyproject.toml` (`anthropic = ["anthropic>=0.40.0"]`).
2. Import it inside the function or property that needs it, not at module scope.
3. Catch `ImportError` and re-raise one that names the extra:

```python
_ANTHROPIC_IMPORT_ERROR = (
    "The 'anthropic' package is required to use AnthropicModel. "
    "Install it with: pip install -e '.[anthropic]'"
)
```

The lazy-client property in `AnthropicModel` shows the shape: `_client` starts as `None`, the `client` property imports and constructs on first use, and the error a user without the extra sees tells them exactly what to install.

`OpenAICompatibleModel` imports `openai` at module scope because `openai` is a core dependency, not an extra. That is the only reason it is allowed to.

## `from_args()`

Each adapter module exposes a `from_args(args: dict) -> Adapter` factory. It is what the Streamlit intake form under `demo/` uses, so it takes a flat dict of strings from a form or a config file rather than typed Python values.

- **Required fields** are validated explicitly and raise with a message naming the field: `raise ValueError("AnthropicModel requires a 'model' argument")`.
- **Missing or empty optional fields are unset**, not empty. Use `args.get("api_key") or None` so `""` from an untouched form field becomes `None` and the SDK's own environment-variable resolution still applies. An empty string passed through as a key produces a confusing auth failure at the first call instead of the intended default.
- **Type normalization happens here**, not at the call site. Accept the string forms a form produces (`"0.35"`, `"4000"`) and convert them, rejecting values that are not valid with a clear error. Share the coercion helper with the constructor so both entry points enforce the same rule.
- Document the accepted keys and their defaults in the factory's docstring.

## The conformance suite

`tests/model_adapter_contract.py` provides `ModelAdapterContractTests`, a base class covering everything above: inheritance, a non-empty stable name, `predict` being async and returning a `ModelResponse`, verbatim input, single-delivery context, context never reaching a privileged channel, the JSON shape of the user turn when context is present, empty-context neutrality, output fidelity, the standard metadata keys, single tracing, transport-error propagation, optional-import behaviour, and credential redaction.

Two hooks let the channel assertions stay provider-neutral. `privileged_texts(call)` returns the trusted instruction text in a recorded request and `user_texts(call)` returns the user-turn bodies; the defaults read a top-level `system` parameter and message `role` fields, which covers both built-in adapters. Override them only for a provider that spells trusted instructions some other way.

The suite is fully offline and provider-neutral. It never imports `openai` or `anthropic` and never inspects a provider request type; it walks the plain dicts the adapter handed its mocked transport. Keep it that way — an assertion that reaches into a provider's SDK objects stops being a shared contract.

To exercise it, add a `Test`-prefixed subclass to the adapter's own test module and supply the factory:

```python
class TestMyProviderContract(ModelAdapterContractTests):
    adapter_module = my_provider
    optional_sdk_module = "my_sdk"
    optional_sdk_extra = "[my-provider]"
    secret_values = ("secret-test-key",)

    def make_adapter(self):
        return my_provider.MyProviderModel(model="m", api_key="secret-test-key")

    def provider_calls(self, adapter):
        return adapter.client.requests.calls

    def set_transport_error(self, adapter, error):
        async def failing_create(**kwargs):
            raise error

        adapter.client.requests.create = failing_create

    def expected_output(self, adapter):
        return "offline answer"
```

`make_adapter` returns a configured adapter whose transport is already mocked — the existing modules do this with an autouse fixture that patches the SDK client class. `provider_calls` returns the request payloads that mock recorded. `secret_values` lists the credentials the redaction tests search for. Leave `optional_sdk_module` unset for an adapter built on a core dependency; the optional-import tests skip themselves.

Provider-specific behaviour — message ordering, block concatenation, per-call overrides, `from_args` normalization — stays in the adapter's own tests. The conformance suite covers what is shared, not what is particular.

## Checklist

- [ ] Subclasses `BaseModel`, calls `super().__init__(name=...)`
- [ ] `predict` is async, returns `ModelResponse`, leaves `input_text` verbatim
- [ ] Non-empty `context` sent once, in the JSON user payload, never in a privileged channel
- [ ] Context handling policy appended to the system instructions when context is present
- [ ] Empty `context` changes nothing; `input_text` sent as the bare user content
- [ ] `output` preserves provider text and order, invents nothing
- [ ] All six standard metadata keys present, `None` when unavailable
- [ ] No credentials in `repr`, metadata, or raised errors
- [ ] Transport failures propagate; invalid config rejected at construction
- [ ] `predict` left undecorated so `BaseModel` traces it once
- [ ] Optional SDK in a named extra, imported lazily, error names the extra
- [ ] `from_args` validates required fields and normalizes empty optionals
- [ ] `ModelAdapterContractTests` subclass added to the adapter's test module
