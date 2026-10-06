"""Call models by configurable nickname. Defaults send Claude to Anthropic, GPT to OpenAI, and
everything else through OpenRouter, pinned to one host. Every reply records its dollar cost and who served it.

Conversations use OpenAI's chat format everywhere (see README.md); this file translates it for Anthropic. Settings
have one shared name each, translated for every API; a setting a model can't take raises ValueError."""
import asyncio
import json
import os
import tomllib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import openai

MODELS = tomllib.loads((Path(__file__).parent / "models.toml").read_text())
TIMEOUT = 3600  # Seconds per request; a reply at the full output limit can take many minutes.
CONCURRENCY = 50  # Requests in flight at once per API, unless LLM(concurrency=...) says otherwise.
NATIVE_KEYS = ("anthropic_content", "openai_output")  # Provider-specific copies kept on assistant messages.
CACHE_TIMES = {True: "5m", "5m": "5m", "1h": "1h"}  # Accepted cache marks, and how long each keeps the text.
PING = [{"role": "user", "content": "Reply with only the word: ok"}]
KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
UNSUPPORTED = {  # Shared settings each API has no way to take.
    "anthropic": {"seed", "frequency_penalty", "presence_penalty", "verbosity", "logprobs"},
    "openai": {"top_k", "seed", "frequency_penalty", "presence_penalty", "stop"},
    "openrouter": set(),
}
REASONING = {  # Reasoning levels each API takes, from least to most.
    "anthropic": ("none", "low", "medium", "high", "xhigh", "max"),
    "openai": ("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    "openrouter": ("none", "minimal", "low", "medium", "high", "xhigh"),
}
VERBOSITY = ("low", "medium", "high")
STOP_REASONS = {  # Each API's stop reasons in shared terms; anything unlisted becomes "other".
    "anthropic": {"end_turn": "end", "stop_sequence": "end", "tool_use": "tool_calls", "max_tokens": "max_tokens",
                  "model_context_window_exceeded": "max_tokens", "refusal": "refusal"},
    "openai": {"completed": "end", "max_output_tokens": "max_tokens", "content_filter": "content_filter"},
    "openrouter": {"stop": "end", "tool_calls": "tool_calls", "length": "max_tokens",
                   "content_filter": "content_filter"},
}
BATCH_APIS = ("anthropic", "openai")  # OpenRouter has no batch API.
BATCH_PRICE = 0.5  # Both batch APIs charge half of every price, cache reads and writes included (checked 2026-10-05).
BATCH_WAIT = 2  # Maximum seconds before queued requests are sent as a batch.
BATCH_POLL = 30  # Seconds between checks on a running batch.
BATCH_MAX = 10_000  # Requests per batch; large multimodal inputs may reach provider byte limits sooner.


@dataclass
class Reply:
    text: str
    reasoning: str  # Visible reasoning or its summary, where the API returns any (see README.md).
    tool_calls: list  # OpenAI format: [{"id", "type": "function", "function": {"name", "arguments"}}].
    message: dict  # This reply as an assistant message, ready to append to the conversation.
    stop_reason: str  # "end", "tool_calls", "max_tokens" (cut off), "refusal", "content_filter", or "other".
    api_stop_reason: str  # The stop reason exactly as the API reported it.
    host: str  # Who served the request.
    cost: float | None  # Dollars; None when usage or pricing is unavailable, never a fabricated zero.
    usage: dict  # Token counts exactly as the API reported them.
    logprobs: list | None = None  # With `logprobs`: [{"token", "logprob", "top_logprobs": [{"token", "logprob"}]}].
    settings: dict = field(default_factory=dict)  # Every setting used besides the model, messages, and tools.
    model: str = ""  # The nickname asked for, as in models.toml.
    model_id: str = ""  # That model's name on the API it went to.
    id: str = ""  # Provider response ID, for tracing and billing reconciliation.


def price(spec, uncached, cache_read, cache_write, output, cache_write_1h=0):
    """Dollar cost of a direct API call from its token counts and the model's prices. Above a model's long-prompt
    threshold, every token in the call is billed at the long-prompt prices."""
    if not spec.get("price") or min(uncached, cache_read, cache_write, output, cache_write_1h) < 0:
        return None
    long = spec.get("long_price")
    p = long if long and uncached + cache_read + cache_write + cache_write_1h > long["above"] else spec["price"]
    return (uncached * p["input"] + cache_read * p["cache_read"] + cache_write * p["cache_write"]
            + cache_write_1h * p.get("cache_write_1h", p["cache_write"]) + output * p["output"]) / 1e6


def text_of(content):
    """The text in a message's content, whether a string or a list of parts."""
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)


def anthropic_content(content):
    """OpenAI-format content (a string, or text and image_url parts) in Anthropic's format."""
    if isinstance(content, str):
        return content
    blocks = []
    for part in content:
        if part["type"] == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image_url":
            url = part["image_url"]["url"]
            if url.startswith("data:"):  # data:<media type>;base64,<data>
                media_type, data = url[5:].split(";base64,")
                source = {"type": "base64", "media_type": media_type, "data": data}
            else:
                source = {"type": "url", "url": url}
            blocks.append({"type": "image", "source": source})
        else:
            raise ValueError(f"Can't send a {part['type']!r} content part to Anthropic")
    return blocks


def cache_control(cache):
    """The cache mark Anthropic uses, which OpenRouter also passes on to hosts that need marks."""
    return {"type": "ephemeral", **({"ttl": "1h"} if cache == "1h" else {})}


def anthropic_mark(content, cache):
    """Anthropic-format content with a cache mark on its last block, if `cache` asks for one."""
    if not cache:
        return content
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else [dict(b) for b in content]
    if not blocks:
        raise ValueError("Can't cache an empty content list")
    blocks[-1]["cache_control"] = cache_control(cache)
    return blocks


def anthropic_messages(messages):
    """OpenAI-format messages in Anthropic's format. System messages are left out (they go in `system`), tool
    calls become tool_use blocks, and consecutive tool results become one user turn of tool_result blocks.
    An assistant message that came from Claude is sent back exactly, thinking blocks included. A message
    marked for caching gets Anthropic's cache mark on its last block."""
    out = []
    for m in messages:
        if m["role"] not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Can't translate role {m['role']!r} to Anthropic")
        if m["role"] == "system":
            if out:
                raise ValueError("Claude only takes system messages at the start of a conversation")
            continue
        if m["role"] == "tool":
            content = anthropic_content(m["content"])
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": content}
            previous = out[-1] if out else {}
            if previous.get("role") == "user" and isinstance(previous["content"], list) \
                    and previous["content"][-1]["type"] == "tool_result":
                previous["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m["role"] == "assistant" and "anthropic_content" in m:
            out.append({"role": "assistant", "content": m["anthropic_content"]})
        elif m["role"] == "assistant":
            content = anthropic_content(m.get("content") or "")
            if calls := [{"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                          "input": json.loads(c["function"]["arguments"] or "{}")} for c in m.get("tool_calls") or []]:
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}] if content else []
                content = content + calls
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": "user", "content": anthropic_content(m["content"])})
        out[-1]["content"] = anthropic_mark(out[-1]["content"], m.get("cache"))
    return out


def openai_parts(content, cache=None):
    """OpenAI-format content as Responses API input parts, the last one marked as a cache breakpoint if `cache`
    asks for one. Unmarked text stays a plain string."""
    if isinstance(content, str) and not cache:
        return content
    parts = []
    for part in [{"type": "text", "text": content}] if isinstance(content, str) else content:
        if part["type"] == "text":
            parts.append({"type": "input_text", "text": part["text"]})
        elif part["type"] == "image_url":
            image = part["image_url"]
            parts.append({"type": "input_image", "image_url": image["url"],
                          **({"detail": image["detail"]} if "detail" in image else {})})
        else:
            raise ValueError(f"Can't send a {part['type']!r} content part to OpenAI")
    if cache:
        if not parts:
            raise ValueError("Can't cache an empty content list")
        parts[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return parts


def openai_input(messages):
    """OpenAI-format chat messages as input items for OpenAI's Responses API. An assistant message that came from
    OpenAI is sent back exactly, encrypted reasoning included. OpenAI can't mark its own replies for caching."""
    items = []
    for m in messages:
        if m.get("cache") == "1h":
            raise ValueError("OpenAI only caches for 30 minutes; use cache=True")
        if m["role"] == "assistant" and m.get("cache"):
            raise ValueError("OpenAI can't cache up to an assistant message; mark the message after it instead")
        if m["role"] == "tool":
            output = openai_parts(m["content"], m.get("cache"))
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": output})
        elif m["role"] == "assistant" and "openai_output" in m:
            items += m["openai_output"]
        elif m["role"] == "assistant":
            if m.get("content"):
                items.append({"role": "assistant", "content": text_of(m["content"])})
            items += [{"type": "function_call", "call_id": c["id"], "name": c["function"]["name"],
                       "arguments": c["function"]["arguments"]} for c in m.get("tool_calls") or []]
        else:
            items.append({"role": m["role"], "content": openai_parts(m["content"], m.get("cache"))})
    return items


def openrouter_messages(messages):
    """OpenAI-format messages for OpenRouter: Claude- and OpenAI-only copies dropped, and cache marks written the
    way OpenRouter passes them on to hosts that need them."""
    out = []
    for m in messages:
        cache = m.get("cache")
        m = {k: v for k, v in m.items() if k not in (*NATIVE_KEYS, "cache")}
        if cache:
            if not m.get("content"):
                raise ValueError("Can't cache up to a message with no text; mark the message after it instead")
            parts = [{"type": "text", "text": m["content"]}] if isinstance(m["content"], str) else list(m["content"])
            parts[-1] = {**parts[-1], "cache_control": cache_control(cache)}
            m["content"] = parts
        out.append(m)
    return out

def shared_logprobs(tokens):
    """Logprobs from OpenAI or OpenRouter, which report them the same way, without the raw bytes."""
    return [{"token": t["token"], "logprob": t["logprob"],
             "top_logprobs": [{"token": a["token"], "logprob": a["logprob"]} for a in t.get("top_logprobs") or []]}
            for t in tokens]


class LLM:
    def __init__(self, max_retries=5, concurrency=CONCURRENCY, batch=False, *, models=None, clients=None, timeout=TIMEOUT):
        """batch=True sends Claude and GPT requests through their batch APIs: half price, but replies take minutes
        to hours (at most 24). `ask` works the same either way (see README.md)."""
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        self.models = deepcopy(MODELS | (models or {}))
        self.clients = dict(clients or {})  # Supplied clients remain owned by the caller.
        self.owned = set()
        self.timeout = timeout
        self.max_retries = max_retries
        self.cost = self.known_cost = 0.0
        self.costs = {}  # Totals become None if any observed charge is unknown.
        self.batch = batch
        self.concurrency = concurrency
        self.loop = None  # The event loop the clients and slots below belong to; see _use_this_loop.
        self.closed = False

    def _use_this_loop(self):
        """An instance belongs to one event loop; never silently abandon its clients or outstanding work."""
        loop = asyncio.get_running_loop()
        if self.closed or self.loop is not None and self.loop is not loop:
            raise RuntimeError("Use a new LLM after closing it or switching event loops")
        if loop is not self.loop:
            self.loop = loop
            self.slots = {api: asyncio.Semaphore(self.concurrency) for api in KEYS}
            self.waiting = {api: [] for api in BATCH_APIS}  # (request, future, model) not yet sent in a batch.
            self.timers = {}
            self.running = {}  # Batch task -> jobs; includes tasks canceled before their coroutine starts.

    async def __aenter__(self):
        self._use_this_loop()
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        """Close owned clients and stop local batch work. Submitted remote batches may still run and be billed."""
        if self.closed:
            return
        self._use_this_loop()
        self.closed = True
        for timer in self.timers.values():
            timer.cancel()
        for jobs in (*self.waiting.values(), *self.running.values()):
            for _, future, _ in jobs:
                future.cancel()
        tasks = list(self.running)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(self.clients[api].close() for api in self.owned))

    def _record(self, model, cost):
        """Record usage before parsing output or delivering it to a possibly canceled caller."""
        self.known_cost += cost or 0.0
        self.cost = None if cost is None or self.cost is None else self.cost + cost
        previous = self.costs.get(model, 0.0)
        self.costs[model] = None if cost is None or previous is None else previous + cost
        return cost

    def client(self, api):
        """The client for one API, made on first use so that only the keys actually used need to be set."""
        if api not in self.clients:
            if not os.environ.get(KEYS[api]):
                raise ValueError(f"{KEYS[api]} isn't set; it's needed for models that go through {api}")
            settings = {"api_key": os.environ[KEYS[api]], "max_retries": self.max_retries, "timeout": self.timeout}
            self.clients[api] = (anthropic.AsyncAnthropic(**settings) if api == "anthropic" else openai.AsyncOpenAI(
                **settings, **({"base_url": "https://openrouter.ai/api/v1"} if api == "openrouter" else {})))
            self.owned.add(api)
        return self.clients[api]

    async def check(self, models, call=False):
        """Raise ValueError unless every nickname is in models.toml and listed where it says, which costs nothing.
        With call=True, also send each model a tiny real request (a fraction of a cent each), which catches what
        listings can't, such as a host that only serves people with their own account."""
        self._use_this_loop()
        models = list(models)
        if unknown := sorted(set(models) - set(self.models)):
            raise ValueError(f"Unknown models: {', '.join(unknown)}")
        listed = {}  # API -> model IDs it offers, fetched once.
        for model in models:
            spec = self.models[model]
            client = self.client(spec["api"]).with_options(timeout=30, max_retries=1)
            try:
                if spec["api"] == "openrouter":
                    data = await client.get(f"/models/{spec['id']}/endpoints", cast_to=object)
                    available = spec["host"] in {e["tag"].split("/")[0] for e in data["data"]["endpoints"]}
                else:
                    if spec["api"] not in listed:
                        listed[spec["api"]] = {m.id async for m in client.models.list()}
                    available = spec["id"] in listed[spec["api"]]
            except (anthropic.APIError, openai.APIError) as e:
                raise ValueError(f"Couldn't reach {spec['api']} to check {model}: {e}") from e
            if not available:
                raise ValueError(f"{model} ({spec['id']}) is not available from {spec.get('host', spec['api'])}")
        if call:
            results = await asyncio.gather(*(self.ask(m, PING, max_tokens=16, batch=False) for m in models),
                                           return_exceptions=True)
            if failed := [f"{m}: {r}" for m, r in zip(models, results) if isinstance(r, Exception)]:
                raise ValueError("Test calls failed:\n" + "\n".join(failed))

    async def ask(self, model, messages, *, max_tokens=None, temperature=None, top_p=None, top_k=None, seed=None,
                  frequency_penalty=None, presence_penalty=None, reasoning=None, verbosity=None, stop=None,
                  schema=None, logprobs=None, tools=None, tool_choice=None, parallel_tool_calls=None, cache=None,
                  extra_body=None, batch=None):
        """Return a Reply using shared settings (README.md). Unsupported settings raise ValueError.
        `extra_body` supplies provider-native options; `batch` overrides the instance default.
        Append reply.message to continue a conversation without losing provider reasoning state.
        """
        self._use_this_loop()
        if model not in self.models:
            raise ValueError(f"Unknown model: {model!r}; add it with LLM(models=...)")
        if not messages:
            raise ValueError("messages must not be empty")
        spec = self.models[model]
        api = spec["api"]
        batch = self.batch if batch is None else batch
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
                                       or max_tokens < 1):
            raise ValueError("max_tokens must be a positive integer")
        settings = {"max_tokens": spec["max_tokens"] if max_tokens is None else max_tokens,
                    "temperature": temperature, "top_p": top_p,
                    "top_k": top_k, "seed": seed, "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty, "reasoning": reasoning, "verbosity": verbosity,
                    "stop": stop, "schema": schema, "logprobs": logprobs, "tool_choice": tool_choice,
                    "parallel_tool_calls": parallel_tool_calls}
        settings = {k: v for k, v in settings.items() if v is not None}
        if unsupported := sorted(UNSUPPORTED[api] & set(settings)):
            raise ValueError(f"{model} can't take {', '.join(unsupported)}; {api}'s API has no such setting")
        if "temperature" in settings and not spec.get("temperature", True):
            raise ValueError(f"{model} doesn't take a temperature setting; its API rejects one")
        if "reasoning" in settings and reasoning not in REASONING[api]:
            raise ValueError(f"{model} takes reasoning {', '.join(REASONING[api])}, not {reasoning!r}")
        if "reasoning" in settings and not spec.get("reasoning", True):
            raise ValueError(f"{model} does not support reasoning controls")
        if "verbosity" in settings and verbosity not in VERBOSITY:
            raise ValueError(f"verbosity must be {', '.join(VERBOSITY)}, not {verbosity!r}")
        if not tools and {"tool_choice", "parallel_tool_calls"} & set(settings):
            raise ValueError("tool_choice and parallel_tool_calls need tools")
        if batch and api not in BATCH_APIS:
            raise ValueError(f"{model} can't be batched; only Claude and GPT models have batch APIs")
        if cache is False and any(m.get("cache") for m in messages):
            raise ValueError("cache=False conflicts with a message marked for caching")
        if cache is not None:
            messages = [*messages[:-1], {**messages[-1], "cache": cache}]
        if marked := [m["cache"] for m in messages if "cache" in m]:
            if bad := set(marked) - set(CACHE_TIMES) - {False}:
                raise ValueError(f"cache must be True, '5m', '1h', or False, not {bad}")
            if (spec["caching"] == "free" or spec.get("automatic_caching")) and False in marked:
                raise ValueError(f"{model} caches automatically, so it can't be told not to cache")
            if spec["caching"] == "free" and "1h" in marked:
                raise ValueError(f"{model} caches automatically, for as long as its provider chooses, not for '1h'")
            if spec["caching"] == "none" and any(marked):
                raise ValueError(f"{model}'s host doesn't cache")
            if spec["caching"] == "free":  # It caches anyway, so the marks are dropped rather than sent.
                messages = [{k: v for k, v in m.items() if k != "cache"} for m in messages]
        marks = {i: CACHE_TIMES[m["cache"]] for i, m in enumerate(messages) if m.get("cache")}
        request = getattr(self, f"_{api}_request")(spec, messages, tools, settings)
        if extra_body:
            if forbidden := {"model", "messages", "input", "stream", "n"} & extra_body.keys():
                raise ValueError(f"extra_body cannot override {', '.join(sorted(forbidden))}")
            request["extra_body"] = request.get("extra_body", {}) | deepcopy(extra_body)
            settings["extra_body"] = deepcopy(extra_body)
        if batch:
            request.update(request.pop("extra_body", {}))  # Batch APIs take JSON bodies, not SDK kwargs.
            reply = await self._batched(api, model, request)
        else:
            async with self.slots[api]:
                try:
                    reply = await getattr(self, "_" + api)(model, request)
                except (openai.APIConnectionError, anthropic.APIConnectionError, asyncio.CancelledError):
                    self._record(model, None)  # A request may have been billed before the connection was lost.
                    raise
        reply.settings = {**settings, **({"cache": marks} if marks else {}), **({"batch": True} if batch else {})}
        reply.model, reply.model_id = model, spec["id"]
        return reply

    def _anthropic_request(self, spec, messages, tools, settings):
        if any(p["type"] != "text" for m in messages if m["role"] == "system" and not isinstance(m["content"], str)
               for p in m["content"]):
            raise ValueError("Claude only takes text in system messages")
        request = {"model": spec["id"], "max_tokens": settings["max_tokens"], "messages": anthropic_messages(messages)}
        if system := [anthropic_mark([{"type": "text", "text": text_of(m["content"])}], m.get("cache"))[0]
                      for m in messages if m["role"] == "system"]:
            request["system"] = system
        if tools:
            request["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                                 "input_schema": t["function"].get("parameters", {"type": "object"}),
                                 **({"strict": t["function"]["strict"]} if "strict" in t["function"] else {})}
                                for t in tools]
        for name in ("temperature", "top_p", "top_k"):
            if name in settings:
                request[name] = settings[name]
        if "stop" in settings:
            request["stop_sequences"] = settings["stop"]
        output_config = {}
        if settings.get("reasoning") == "none":
            request["thinking"] = {"type": "disabled"}
        elif "reasoning" in settings:
            request["thinking"] = {"type": "adaptive", "display": "summarized"}
            output_config["effort"] = settings["reasoning"]
        elif spec.get("thinks"):  # Shows the thinking it does anyway; asking would turn it on for other models.
            request["thinking"] = {"type": "adaptive", "display": "summarized"}
        if "schema" in settings:
            output_config["format"] = {"type": "json_schema", "schema": settings["schema"]}
        if output_config:
            request["output_config"] = output_config
        choice = settings.get("tool_choice")
        if choice or "parallel_tool_calls" in settings:
            request["tool_choice"] = {"auto": {"type": "auto"}, "none": {"type": "none"}, "required": {"type": "any"},
                                      None: {"type": "auto"}}.get(choice, {"type": "tool", "name": choice})
            if settings.get("parallel_tool_calls") is False:
                request["tool_choice"]["disable_parallel_tool_use"] = True
        return request

    async def _anthropic(self, model, request):
        # The SDK dropped its sampling options; the models that still take them get them as extra fields.
        extra = {k: request.pop(k) for k in ("temperature", "top_p", "top_k") if k in request}
        extra.update(request.pop("extra_body", {}))
        async with self.client("anthropic").messages.stream(**request, extra_body=extra) as stream:  # For long replies.
            return self._anthropic_reply(model, await stream.get_final_message())

    def _anthropic_reply(self, model, message, batch=False):
        spec = self.models[model]
        usage = message.usage.model_dump() if message.usage else {}
        write_1h = (usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0
        cost = price(spec, usage["input_tokens"], usage.get("cache_read_input_tokens") or 0,
                     (usage.get("cache_creation_input_tokens") or 0) - write_1h, usage["output_tokens"], write_1h) if usage else None
        tier = usage.get("service_tier") or "standard"
        if ("standard" if tier == "batch" else tier) != spec.get("service_tier", "standard") \
                or any((usage.get("server_tool_use") or {}).values()):
            cost = None  # The token price table cannot price premium tiers or server tools.
        cost = self._record(model, cost * (BATCH_PRICE if batch else 1) if cost is not None else None)
        blocks = [block.model_dump(exclude_none=True) for block in message.content]
        text = "".join(b["text"] for b in blocks if b["type"] == "text")
        tool_calls = [{"id": b["id"], "type": "function",
                       "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                      for b in blocks if b["type"] == "tool_use"]
        reply = {"role": "assistant", "content": text, "anthropic_content": blocks}
        if tool_calls:
            reply["tool_calls"] = tool_calls
        reasoning = "".join(b.get("thinking", "") for b in blocks if b["type"] == "thinking")
        stop = message.stop_reason
        return Reply(text, reasoning, tool_calls, reply, STOP_REASONS["anthropic"].get(stop, "other"), stop,
                     "Anthropic", cost, usage, id=message.id)

    def _openai_request(self, spec, messages, tools, settings):
        """OpenAI's Responses API, the only one where its newer models combine reasoning with tools and report
        reasoning summaries. Nothing is stored at OpenAI; reasoning is carried in the conversation, encrypted."""
        request = {"model": spec["id"], "input": openai_input(messages), "max_output_tokens": settings["max_tokens"],
                   "store": False, "include": []}
        if spec.get("reasoning", True):
            request["include"].append("reasoning.encrypted_content")
            request["reasoning"] = {"summary": "auto", **({"effort": settings["reasoning"]}
                                                         if "reasoning" in settings else {})}
        if tools:  # Not strict by default, matching the other APIs.
            request["tools"] = [{"type": "function", "strict": False, **t["function"]} for t in tools]
        if spec["caching"] == "paid":  # Turns off OpenAI's automatic caching, so only marked messages are cached.
            request["prompt_cache_options"] = {"mode": "explicit"}
        for name in ("temperature", "top_p", "parallel_tool_calls"):
            if name in settings:
                request[name] = settings[name]
        text = {}
        if "schema" in settings:
            text["format"] = {"type": "json_schema", "name": "reply", "schema": settings["schema"], "strict": True}
        if "verbosity" in settings:
            text["verbosity"] = settings["verbosity"]
        if text:
            request["text"] = text
        if "logprobs" in settings:
            request["top_logprobs"] = settings["logprobs"]
            request["include"].append("message.output_text.logprobs")
        if choice := settings.get("tool_choice"):
            request["tool_choice"] = choice if choice in ("auto", "none", "required") else {
                "type": "function", "name": choice}
        return request

    async def _openai(self, model, request):
        return self._openai_reply(model, await self.client("openai").responses.create(**request))

    def _openai_reply(self, model, response, batch=False):
        spec = self.models[model]
        usage = response.usage.model_dump() if response.usage else {}
        details = usage.get("input_tokens_details") or {}
        cache_read, cache_write = details.get("cached_tokens") or 0, details.get("cache_write_tokens") or 0
        cost = price(spec, usage.get("input_tokens", 0) - cache_read - cache_write, cache_read, cache_write,
                     usage.get("output_tokens", 0))
        if not usage or (response.service_tier or "default") != spec.get("service_tier", "default") \
                or any(item.type not in ("message", "reasoning", "function_call") for item in response.output):
            cost = None
        cost = self._record(model, cost * (BATCH_PRICE if batch else 1) if cost is not None else None)
        if response.status == "failed":
            raise RuntimeError(f"OpenAI response {response.id} failed: {response.error}")
        output = [item.model_dump(exclude_none=True) for item in response.output]
        parts = [part for item in output if item["type"] == "message" for part in item["content"]]
        text = "".join(part.get("text") or part.get("refusal") or "" for part in parts)
        reasoning = "".join(part["text"] for item in output if item["type"] == "reasoning"
                            for part in item.get("summary", []))
        tool_calls = [{"id": item["call_id"], "type": "function",
                       "function": {"name": item["name"], "arguments": item["arguments"]}}
                      for item in output if item["type"] == "function_call"]
        message = {"role": "assistant", "content": text, "openai_output": output}
        if tool_calls:
            message["tool_calls"] = tool_calls
        api_stop = response.incomplete_details.reason if response.incomplete_details else response.status
        stop = STOP_REASONS["openai"].get(api_stop, "other")
        if stop == "end" and any(part["type"] == "refusal" for part in parts):
            stop = "refusal"
        elif stop == "end" and tool_calls:
            stop = "tool_calls"
        logprobs = [token for part in parts for token in part.get("logprobs") or []]
        return Reply(text, reasoning, tool_calls, message, stop, api_stop, "OpenAI",
                     cost, usage, shared_logprobs(logprobs) if logprobs else None, id=response.id)

    def _openrouter_request(self, spec, messages, tools, settings):
        """Pinned to one host, with no fallback and no host that would ignore a setting."""
        extra = {"provider": {"only": [spec["host"]], "allow_fallbacks": False, "require_parameters": True}}
        if "reasoning" in settings:
            extra["reasoning"] = {"effort": settings["reasoning"]}
        for name in ("top_k", "verbosity"):
            if name in settings:
                extra[name] = settings[name]
        request = {"model": spec["id"], "messages": openrouter_messages(messages),
                   "max_tokens": settings["max_tokens"], "extra_body": extra}
        if tools:
            request["tools"] = tools
        for name in ("temperature", "top_p", "seed", "frequency_penalty", "presence_penalty", "stop",
                     "parallel_tool_calls"):
            if name in settings:
                request[name] = settings[name]
        if "schema" in settings:
            request["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "reply", "schema": settings["schema"], "strict": True}}
        if "logprobs" in settings:
            request["logprobs"], request["top_logprobs"] = True, settings["logprobs"]
        if choice := settings.get("tool_choice"):
            request["tool_choice"] = choice if choice in ("auto", "none", "required") else {
                "type": "function", "function": {"name": choice}}
        return request

    async def _openrouter(self, model, request):
        """OpenRouter adds `reasoning`, and `reasoning_details`, which some models need sent back with their tool
        calls. A host error that arrives as a normal-looking response (no reply, or finish_reason "error") is retried
        like any other temporary error, and whatever the failed tries cost is added to the reply's cost."""
        spec, total = self.models[model], 0.0
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client("openrouter").chat.completions.create(**request)
            except openai.NotFoundError as e:  # OpenRouter's way of saying the pinned host can't take a setting.
                if "No endpoints found that can handle the requested parameters" not in str(e):
                    raise
                used = [k for k in request if k not in ("model", "messages", "max_tokens", "extra_body")]
                used += [k for k in request["extra_body"] if k != "provider"]
                raise ValueError(f"{spec['host']} doesn't take one of the settings sent: {', '.join(used)}") from e
            host = getattr(response, "provider", None)
            usage = response.usage.model_dump() if response.usage else {}
            cost = self._record(model, usage.get("cost"))
            total = None if total is None or cost is None else total + cost
            if response.choices and response.choices[0].finish_reason != "error":
                break
            if attempt == self.max_retries:
                raise RuntimeError(f"No reply from {host} after {attempt + 1} tries: {getattr(response, 'error', None)}")
            await asyncio.sleep(2 ** attempt)
        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls = [call.model_dump() for call in choice.message.tool_calls or []]
        message = {"role": "assistant", "content": text}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if details := getattr(choice.message, "reasoning_details", None):
            message["reasoning_details"] = details
        reasoning = getattr(choice.message, "reasoning", None) or ""
        logprobs = [token.model_dump() for token in choice.logprobs.content or []] if choice.logprobs else []
        stop = choice.finish_reason
        return Reply(text, reasoning, tool_calls, message, STOP_REASONS["openrouter"].get(stop, "other"), stop,
                     host, total, usage, shared_logprobs(logprobs) if logprobs else None, id=response.id)

    async def _batched(self, api, model, request):
        """Queue a request, bounding both wait time and request count."""
        future = asyncio.get_running_loop().create_future()
        self.waiting[api].append((request, future, model))
        if api not in self.timers:  # Bound the wait even when requests arrive continuously.
            self.timers[api] = asyncio.get_running_loop().call_later(BATCH_WAIT, self._send_batches, api)
        if len(self.waiting[api]) >= BATCH_MAX:
            self._send_batches(api)
        return await future

    def _send_batches(self, api):
        if timer := self.timers.pop(api, None):
            timer.cancel()
        jobs, self.waiting[api] = self.waiting[api], []
        jobs = [job for job in jobs if not job[1].done()]
        for start in range(0, len(jobs), BATCH_MAX):
            chunk = jobs[start:start + BATCH_MAX]
            task = asyncio.create_task(self._run_batch(api, chunk))
            self.running[task] = chunk
            task.add_done_callback(self.running.pop)

    async def _run_batch(self, api, jobs):
        """Send one batch, wait for it to finish, and hand each request its reply (or its error)."""
        try:
            await getattr(self, f"_run_{api}_batch")(jobs)
        except (Exception, asyncio.CancelledError) as e:
            for _, future, model in jobs:
                if not future.done() or future.cancelled():
                    self._record(model, None)
                if not future.done():
                    future.set_exception(e)
            if isinstance(e, asyncio.CancelledError):
                raise
        for _, future, model in jobs:
            if not future.done():
                self._record(model, None)
                future.set_exception(RuntimeError("The batch ended without a reply to this request"))

    async def _run_anthropic_batch(self, jobs):
        client = self.client("anthropic")
        batch = await client.messages.batches.create(
            requests=[{"custom_id": str(i), "params": request} for i, (request, _, _) in enumerate(jobs)])
        print(f"Anthropic batch {batch.id}: {len(jobs)} requests sent")
        last = None
        while batch.processing_status != "ended":
            await asyncio.sleep(BATCH_POLL)
            batch = await client.messages.batches.retrieve(batch.id)
            counts = batch.request_counts
            if (now := (counts.processing, counts.succeeded)) != last:
                print(f"Anthropic batch {batch.id}: {counts.succeeded + counts.errored + counts.expired + counts.canceled}"
                      f"/{len(jobs)} done")
                last = now
        async for entry in await client.messages.batches.results(batch.id):
            _, future, model = jobs[int(entry.custom_id)]
            result = entry.result
            self._batch_result("anthropic", model, future,
                               result.message if result.type == "succeeded" else None,
                               None if result.type == "succeeded" else f"{result.type}: {getattr(result, 'error', '')}")

    def _batch_result(self, api, model, future, message, error):
        try:
            if error is not None:
                self._record(model, None)
                raise RuntimeError(f"Batch request failed: {error}")
            if api == "openai":
                message = openai.types.responses.Response.construct(**message)
            reply = getattr(self, f"_{api}_reply")(model, message, batch=True)
        except Exception as e:
            if not future.done():
                future.set_exception(e)
        else:
            if not future.done():
                future.set_result(reply)

    async def _run_openai_batch(self, jobs):
        client = self.client("openai")
        lines = [json.dumps({"custom_id": str(i), "method": "POST", "url": "/v1/responses", "body": request})
                 for i, (request, _, _) in enumerate(jobs)]
        file = await client.files.create(file=("batch.jsonl", "\n".join(lines).encode()), purpose="batch")
        batch = await client.batches.create(input_file_id=file.id, endpoint="/v1/responses", completion_window="24h")
        print(f"OpenAI batch {batch.id}: {len(jobs)} requests sent")
        last = None
        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            await asyncio.sleep(BATCH_POLL)
            batch = await client.batches.retrieve(batch.id)
            counts = batch.request_counts
            if counts and (now := (counts.completed, counts.failed)) != last:
                print(f"OpenAI batch {batch.id}: {counts.completed + counts.failed}/{len(jobs)} done")
                last = now
        for file_id in (batch.output_file_id, batch.error_file_id):
            if not file_id:
                continue
            for line in (await client.files.content(file_id)).text.splitlines():
                result = json.loads(line)
                _, future, model = jobs[int(result["custom_id"])]
                response = result.get("response") or {}
                if response.get("status_code") == 200:
                    self._batch_result("openai", model, future, response["body"], None)
                else:
                    self._batch_result("openai", model, future, None, result.get("error") or response)
        if batch.status != "completed":
            raise RuntimeError(f"OpenAI batch {batch.id} {batch.status}: {batch.errors}")
