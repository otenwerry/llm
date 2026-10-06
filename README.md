# llm

[Human-written]
Unified interface to call models through OpenAI/Anthropic/OpenRouter APIs. Work in progress, probably not good to use in arbitrary projects yet but works well enough for what I'm working on now; will make a bit better later on

[LLM-written]
Install this directory as a local dependency (`uv add --editable /path/to/llm`) or install its built wheel.

```python
from llm import LLM

async with LLM() as llm:
    reply = await llm.ask("opus-4.6", [{"role": "user", "content": "Hi"}], max_tokens=1000)
    print(reply.text, reply.cost)
```

Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY` in the environment. Only keys for APIs you use are required. The library does **not** load `.env` files. If wanted, call `dotenv.load_dotenv()` in your application first.

## Configuration

Built-in aliases, prices, and routing live in `llm/models.toml`. Each `LLM` takes its own copy. Add or override complete model specs without editing the package:

```python
llm = LLM(models={"custom": {
    "api": "openai", "id": "your-model-id", "max_tokens": 4096,
    "caching": "free", "reasoning": False,
}})
```

A spec needs `api`, `id`, `max_tokens`, and `caching`; OpenRouter also needs `host`. Set `reasoning=False` for OpenAI models without reasoning support, or `temperature=False` to reject that setting. Anthropic's `thinks=True` requests a summary of default thinking. See the bundled table for pricing fields; missing prices produce an unknown cost.

`LLM(clients={"openai": custom_async_openai_client})` supports your own credentials, endpoint, headers, transport, and SDK configuration. Injected clients remain owned by you. Owned clients close with `async with LLM()` or `await llm.aclose()`. Use one instance per event loop.

`LLM(timeout=3600, max_retries=5, concurrency=50)` configures owned clients and simultaneous calls per API. Injected clients retain their own SDK retry/timeout settings. Defaults route Claude/GPT directly and pin OpenRouter to one host; override the model spec or native provider options to change routing.

## Messages and settings

Messages and function tools use OpenAI chat format. Content accepts strings and `text` / `image_url` parts, including base64 image URLs. Append `reply.message` to continue: it preserves native thinking, encrypted reasoning, and tool calls. Tool results use `role="tool"`, `tool_call_id`, and `content`. Anthropic accepts initial system messages and rejects unsupported roles such as `developer`, rather than silently turning them into user messages.

| Setting | Meaning |
|---|---|
| `max_tokens` | Positive output limit, reasoning included. Defaults to the configured model limit; reduce it when prompt plus output would exceed context. |
| `temperature`, `top_p`, `top_k` | Sampling controls where supported. |
| `seed`, `frequency_penalty`, `presence_penalty` | OpenRouter only. |
| `reasoning` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`; support varies by model/API. Omission preserves provider defaults. |
| `verbosity` | `low`, `medium`, or `high`, for OpenAI/OpenRouter. |
| `stop` | Stop strings, for Anthropic/OpenRouter. |
| `schema` | JSON schema for text output. OpenAI strict schemas require all properties in `required` and `additionalProperties: false`. |
| `logprobs` | Alternative token count; zero means just the chosen token. OpenAI/OpenRouter, model-dependent. |
| `tools`, `tool_choice` | Function definitions; choice is `auto`, `none`, `required`, or a function name. Explicit tool `strict` is preserved. |
| `parallel_tool_calls` | `False` limits replies to one tool call; requires tools. |
| `cache` | Prompt caching controls below. |
| `batch` | Per-call override of `LLM(batch=...)`. |
| `extra_body` | Provider-native options merged over the generated body and recorded in `reply.settings`. |

Unsupported shared options raise errors locally or at the provider. `extra_body` allows native options such as reasoning budgets, service tiers, or metadata. It cannot override model, conversation, streaming mode, or response count. Nested objects replace the corresponding object, so include every field needed. Native options may affect prices. The library returns complete replies; streaming, audio, and embeddings are outside this interface.

## Replies and costs

`Reply` includes `text`, `reasoning`, `tool_calls`, `message`, `stop_reason`, `api_stop_reason`, `host`, `usage`, `cost`, `logprobs`, `settings`, `model`, `model_id`, and provider response `id`. Usage preserves provider fields. Stop reasons normalize to `end`, `tool_calls`, `max_tokens`, `refusal`, `content_filter`, or `other`. OpenAI status `failed` raises an error.

OpenRouter cost comes from reported usage. Direct cost is an **estimate** from usage and configured prices, including cache reads/writes, long prompts, and batch discounts. Keep prices current. Premium tiers and server tools are unpriced unless the estimator covers them; a spec's `service_tier` can identify the tier its prices apply to.

**Unknown cost is `None`, never zero.** Missing usage/prices, unpriced features, or an interrupted request can make totals unknown. `llm.cost` and `llm.costs[model]` include observed charges from failed attempts and canceled callers whose batch results were retrieved. They become `None` when any recorded charge is unknown. `llm.known_cost` remains the subtotal of known charges. Check for `None` before summing or formatting costs.

Accounting happens before output parsing and delivery. A successful retried OpenRouter reply includes charges from earlier returned failures. Charges hidden by SDK retries or process crashes cannot be reconciled from responses alone; use provider billing records for invoice reconciliation. Totals are in memory, not a durable ledger.

## Prompt caching

Specs label caching `free` (automatic), `paid` (explicit writes cost extra), or `none` (unsupported). This is provider prompt/KV caching, not saved responses: each call generates a fresh reply. Paid writes are opt-in.

`cache=True` or `"5m"` marks the end of the prompt; a message's `"cache": True` marks a shared prefix. Anthropic also supports `"1h"`. Automatic caches ignore `True` and reject `False`/`"1h"`. Explicit OpenAI caches retain prompts for 30 minutes and reject assistant-message marks and `"1h"`. Actual hits depend on provider minimum lengths and expiration; inspect `reply.usage`.

`cache=False` rejects conflicting message marks. Providers with both paid and automatic caching should declare `automatic_caching=true`; the library then rejects attempts to disable caching.

## Batches and checks

`LLM(batch=True)` uses Anthropic/OpenAI batch APIs; OpenRouter rejects it. Start calls concurrently with `asyncio.gather`. Queues flush within two seconds or at 10,000 requests; replies can take hours. Settings and accounting work the same, with the configured batch discount.

Canceling a queued call prevents submission. Canceling a submitted caller does not cancel the remote request or discard other results. Closing stops local polling; remote work may continue and be billed. Batch IDs are printed; automatic recovery is not implemented. Finish active work before closing if you need all results.

`await llm.check(models)` checks listings without paid inference; `call=True` also sends tiny direct requests. Run offline regression tests with `uv run pytest tests`. The paid smoke test in `exp_test.py` exits unsuccessfully on failures, including provider rejections; see `DOCS.md`.
