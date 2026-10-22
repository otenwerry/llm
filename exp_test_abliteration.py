"""Paid checks of what abliteration.ai's docs leave unclear, each a few hundred tokens (well under $0.10 in all).
Deterministic regression tests live in tests/."""
import argparse
import asyncio
import json
import os

import httpx2
import openai
from dotenv import load_dotenv
from llm import LLM

MODELS = ("abliterated-large-v2", "abliterated-large", "abliterated")
LOWEST = {"abliterated-large-v2": "low", "abliterated-large": "none", "abliterated": "none"}
QUESTION = [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}]


async def checks(llm, model):
    """(name, coroutine) pairs; each coroutine returns a short note or raises AssertionError."""
    async def reasoning_saved():
        # As exp_evals.py calls it: no reasoning level, so the model's default.
        reply = await llm.ask(model, QUESTION, max_tokens=8000)
        assert reply.reasoning, f"no reasoning text (stop {reply.stop_reason}, usage {reply.usage})"
        assert reply.text.strip() == "391", f"text={reply.text!r}"
        return f"{len(reply.reasoning)} characters of reasoning"

    async def lowest_level():
        reply = await llm.ask(model, QUESTION, max_tokens=2000, reasoning=LOWEST[model])
        assert reply.text.strip() == "391", f"text={reply.text!r}"
        if LOWEST[model] == "none":
            assert not reply.reasoning, f"reasoning 'none' still returned {len(reply.reasoning)} characters of it"
            return "no reasoning, as asked"
        assert reply.reasoning, "reasoning 'low' returned no reasoning text"
        return f"{len(reply.reasoning)} characters of reasoning"

    async def second_turn():
        first = await llm.ask(model, QUESTION, max_tokens=2000, reasoning=LOWEST[model])
        messages = [*QUESTION, first.message, {"role": "user", "content": "Now add 9. Answer with just the number."}]
        second = await llm.ask(model, messages, max_tokens=2000, reasoning=LOWEST[model])
        assert second.text.strip() == "400", f"text={second.text!r}"
        return "follow-up answered from the earlier turn"

    async def cut_off():
        reply = await llm.ask(model, QUESTION, max_tokens=8)
        assert reply.stop_reason == "max_tokens", f"stop_reason={reply.stop_reason} ({reply.api_stop_reason})"
        return "stop_reason max_tokens"

    async def rejects(**settings):
        try:
            reply = await llm.ask(model, QUESTION, max_tokens=2000, **settings)
        except openai.APIStatusError as e:
            return f"rejected: HTTP {e.status_code} {e.message[:120]}"
        raise AssertionError(f"accepted {settings} without an error (text={reply.text!r})")

    found = [("reasoning text saved", reasoning_saved()), (f"reasoning {LOWEST[model]}", lowest_level()),
             ("second turn", second_turn()), ("cut off at max_tokens", cut_off())]
    if model != "abliterated":  # Their model list caps the large models' temperature at 1.
        found.append(("temperature 1.5 rejected", rejects(temperature=1.5)))
    return found


async def balance_used():
    """Dollars abliteration.ai has billed this account so far (free to ask)."""
    async with httpx2.AsyncClient() as client:
        r = await client.get("https://api.abliteration.ai/v1/credits",
                             headers={"Authorization": f"Bearer {os.environ['ABLITERATION_API_KEY']}"})
        return r.json()["data"]["total_usage"]


async def run(args):
    if args.dry_run:
        print(f"Models: {', '.join(args.models)}")
        return
    load_dotenv()
    before, failures = await balance_used(), 0
    async with LLM() as llm:
        async def one(model, name, check):
            nonlocal failures
            try:
                print(f"PASS {model} | {name}: {await check}", flush=True)
            except Exception as error:
                failures += 1
                print(f"FAIL {model} | {name}: {type(error).__name__}: {error}", flush=True)
        jobs = [one(model, name, check) for model in args.models for name, check in await checks(llm, model)]
        await asyncio.gather(*jobs)
        await asyncio.sleep(5)  # Give abliteration.ai's billing a moment to catch up.
        billed = await balance_used() - before
        print(json.dumps({"cost": llm.cost, "known_cost": llm.known_cost, "billed_by_abliteration": round(billed, 6),
                          "failures": failures}))
    return bool(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS), type=lambda value: value.split(","))
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
