"""Paid smoke test of selected models. Deterministic regression tests live in tests/."""
import argparse
import asyncio
import json

from dotenv import load_dotenv
from llm import LLM


async def run(args):
    if args.dry_run:
        print(f"Models: {', '.join(args.models)}; max_tokens={args.max_tokens}; batch={args.batch}")
        return
    load_dotenv()  # Applications choose whether and where to load credentials.
    failures = 0
    async with LLM(batch=args.batch) as llm:
        async def one(model):
            nonlocal failures
            try:
                reply = await llm.ask(model, [{"role": "user", "content": "Reply with only the word: ok"}],
                                      max_tokens=args.max_tokens)
                if reply.text.strip().lower() != 'ok' or reply.stop_reason != 'end':
                    raise AssertionError(f"text={reply.text!r}, stop={reply.stop_reason}")
                if reply.cost is None:
                    raise AssertionError('Cost unavailable')
                print(f"PASS {model}: ${reply.cost:.6f}, {reply.host}", flush=True)
            except Exception as error:
                failures += 1
                print(f"FAIL {model}: {type(error).__name__}: {error}", flush=True)
        await asyncio.gather(*(one(model) for model in args.models))
        print(json.dumps({'cost': llm.cost, 'known_cost': llm.known_cost, 'failures': failures}))
    return bool(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', required=True, type=lambda value: value.split(','))
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == '__main__':
    main()
