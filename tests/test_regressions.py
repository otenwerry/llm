"""Offline contracts: real SDK response types, mocked transports, no credentials or paid calls."""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import httpx2 as httpx
import pytest
from anthropic.types import Message
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

from llm import LLM, MODELS
from llm.api import IDLE_TIMEOUT, anthropic_messages, merge_delta, openai_parts, price

MODELS_BY_API = {'anthropic': 'opus-4.6', 'openai': 'gpt-5.5', 'openrouter': 'qwen-3.7-plus', 'abliteration': 'abliterated'}
MESSAGES = [{'role': 'user', 'content': 'hi'}]
TOOL = {'type': 'function', 'function': {'name': 'weather', 'strict': True, 'parameters': {'type': 'object'}}}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(httpx.AsyncClient, 'send', AsyncMock(side_effect=AssertionError('Network forbidden')))
    monkeypatch.setattr(httpx.Client, 'send', Mock(side_effect=AssertionError('Network forbidden')))


def response(api, **changes):
    if api == 'anthropic':
        return Message.model_validate(dict(id='msg-test', model='test', role='assistant', type='message',
            content=[{'type': 'text', 'text': 'ok'}], stop_reason='end_turn', usage={'input_tokens': 100, 'output_tokens': 20,
            'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}) | changes)
    if api == 'openai':
        return Response.model_validate(dict(id='resp-test', created_at=0, model='test', object='response',
            output=[{'id': 'msg', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                     'content': [{'type': 'output_text', 'text': 'ok', 'annotations': []}]}],
            parallel_tool_calls=True, tool_choice='auto', tools=[], status='completed',
            usage={'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
                   'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0}, 'output_tokens_details': {'reasoning_tokens': 0}}) | changes)
    return ChatCompletion.model_validate(dict(id='gen-test', model='test', created=0, object='chat.completion',
        choices=[{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'ok'}}],
        usage={'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120, 'cost': .12}, provider='mock') | changes)


class MidStream(NS):
    """A reply that starts streaming and then fails with `error`, as a stalled or dropped connection does."""


async def events(items, error=None):
    for item in items:
        yield item
    if error:
        raise error


def streamed(api, raw, error=None):
    """A whole reply as the stream its API sends. Anthropic's helper gives the final message; OpenAI's last event
    carries the response; OpenRouter's reply arrives as a delta chunk and then a usage chunk."""
    if api == 'anthropic':
        class Stream:
            def __aiter__(self):
                return events([NS(type='message_start'), NS(type='message_stop')], error).__aiter__()
            async def get_final_message(self):
                return raw
        return Stream()
    if api == 'openai':
        return events([NS(type='response.created'), NS(type=f'response.{raw.status}', response=raw)], error)
    whole = raw.model_dump(exclude_none=True)
    chunks = [{'id': whole['id'], 'provider': whole.get('provider'), 'choices': [
        {'index': c['index'], 'delta': c['message'], 'finish_reason': c.get('finish_reason')}
        for c in whole['choices']]}, {'id': whole['id'], 'choices': [], 'usage': whole.get('usage')}]
    return events([NS(to_dict=lambda c=c: c) for c in chunks], error)


def setup(api, replies=None, **options):
    """Fake clients that stream each of `replies` in turn. A reply can also be an exception, raised when the
    request is sent, or MidStream(raw=..., error=...), which fails partway through."""
    replies = iter(replies or [response(api)])
    call = AsyncMock(side_effect=lambda **kwargs: next(replies))  # Records each request's arguments.
    async def open_stream(**kwargs):
        item = await call(**kwargs)
        if isinstance(item, Exception):
            raise item
        return streamed(api, item.raw, item.error) if isinstance(item, MidStream) else streamed(api, item)
    @asynccontextmanager
    async def stream(**kwargs):
        yield await open_stream(**kwargs)
    client = NS(close=AsyncMock(), responses=NS(create=open_stream), chat=NS(completions=NS(create=open_stream)),
                messages=NS(stream=stream))
    return LLM(clients={api: client}, **options), client, call


@pytest.mark.parametrize('api', MODELS_BY_API)
def test_public_ask_and_cost(api):
    async def run():
        llm, client, call = setup(api)
        async with llm:
            reply = await llm.ask(MODELS_BY_API[api], MESSAGES, max_tokens=100, reasoning='low')
            assert reply.text == 'ok' and reply.stop_reason == 'end' and reply.id
            assert reply.cost == llm.cost == llm.known_cost == llm.costs[reply.model] > 0
            assert reply.settings['reasoning'] == 'low'
        client.close.assert_not_awaited()  # Borrowed clients stay open.
        with pytest.raises(RuntimeError):
            await llm.ask(MODELS_BY_API[api], MESSAGES)
    asyncio.run(run())


@pytest.mark.parametrize('api', MODELS_BY_API)
def test_extra_body_and_shared_settings(api):
    async def run():
        llm, _, call = setup(api)
        model = 'gpt-5.4' if api == 'openai' else MODELS_BY_API[api]
        await llm.ask(model, MESSAGES, max_tokens=42, temperature=.2, top_p=.8, tools=[TOOL],
                      parallel_tool_calls=False, extra_body={'metadata': {'test': 'value'}})
        body = call.call_args.kwargs
        assert body['extra_body']['metadata'] == {'test': 'value'}
        assert body['extra_body' if api == 'anthropic' else 'tools'] is not None
        assert body['tools'][0].get('strict', body['tools'][0].get('function', {}).get('strict')) is True
        assert body.get('max_output_tokens', body.get('max_tokens')) == 42
        assert body.get('temperature', body['extra_body'].get('temperature')) == .2
    asyncio.run(run())


@pytest.mark.parametrize('field', ['model', 'messages', 'input', 'stream', 'n'])
def test_extra_body_cannot_change_response_contract(field):
    async def run():
        llm, _, call = setup('openai')
        with pytest.raises(ValueError):
            await llm.ask('gpt-5.5', MESSAGES, extra_body={field: 'override'})
        call.assert_not_awaited()
    asyncio.run(run())


def test_custom_models_are_isolated_and_nonreasoning_supported():
    async def run():
        custom = {'local': {'api': 'openai', 'id': 'custom', 'max_tokens': 100, 'reasoning': False, 'caching': 'free'}}
        llm, _, call = setup('openai', models=custom)
        custom['local']['id'] = 'mutated'
        reply = await llm.ask('local', MESSAGES)
        assert call.call_args.kwargs['model'] == 'custom'
        assert 'reasoning' not in call.call_args.kwargs
        assert reply.cost is None and llm.cost is None and llm.known_cost == 0
        assert 'local' not in MODELS
    asyncio.run(run())


@pytest.mark.parametrize('invalid', [0, -1, True, 1.5])
def test_invalid_limits(invalid):
    with pytest.raises(ValueError):
        LLM(concurrency=invalid)
    async def run():
        llm, _, call = setup('openai')
        with pytest.raises(ValueError):
            await llm.ask('gpt-5.5', MESSAGES, max_tokens=invalid)
        call.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('invalid', [-1, True, 1.5])
def test_invalid_retries(invalid):
    with pytest.raises(ValueError):
        LLM(max_retries=invalid)


def test_cache_conflict_and_message_translation():
    async def run():
        llm, _, call = setup('anthropic')
        messages = [{'role': 'system', 'content': 'prefix', 'cache': True}, *MESSAGES]
        with pytest.raises(ValueError):
            await llm.ask('opus-4.6', messages, cache=False)
        await llm.ask('opus-4.6', messages)
        assert call.call_args.kwargs['system'][0]['cache_control'] == {'type': 'ephemeral'}
        assert messages[0]['cache'] is True
    asyncio.run(run())
    with pytest.raises(ValueError):
        anthropic_messages([{'role': 'developer', 'content': 'instructions'}])
    assert openai_parts([{'type': 'image_url', 'image_url': {'url': 'https://example.com/image', 'detail': 'low'}}])[0]['detail'] == 'low'


def test_unknown_cost_does_not_become_zero():
    async def run():
        llm, _, _ = setup('openrouter', [response('openrouter', usage=None), response('openrouter')])
        first = await llm.ask('qwen-3.7-plus', MESSAGES)
        second = await llm.ask('qwen-3.7-plus', MESSAGES)
        assert first.cost is None and second.cost == .12
        assert llm.cost is None and llm.costs['qwen-3.7-plus'] is None and llm.known_cost == .12
    asyncio.run(run())


def test_charged_failure_preserved():
    async def run():
        llm, _, _ = setup('openrouter', [response('openrouter', choices=[])], max_retries=0)
        with pytest.raises(RuntimeError):
            await llm.ask('qwen-3.7-plus', MESSAGES)
        assert llm.cost == llm.costs['qwen-3.7-plus'] == .12
    asyncio.run(run())


def test_retry_then_success_counts_each_attempt_once(monkeypatch):
    monkeypatch.setattr('llm.api.asyncio.sleep', AsyncMock())
    async def run():
        llm, _, _ = setup('openrouter', [response('openrouter', choices=[]), response('openrouter')])
        reply = await llm.ask('qwen-3.7-plus', MESSAGES)
        assert reply.cost == llm.cost == .24
    asyncio.run(run())


@pytest.mark.parametrize('api', ['openai', 'anthropic'])
def test_batch_cancelled_caller_still_accounted_and_other_result_delivered(api):
    async def run():
        llm = LLM()
        llm._use_this_loop()
        futures = [asyncio.get_running_loop().create_future() for _ in range(2)]
        futures[0].cancel()
        jobs = [({}, f, MODELS_BY_API[api]) for f in futures]
        raw = response(api)
        if api == 'openai':
            lines = '\n'.join(json.dumps({'custom_id': str(i), 'response': {'status_code': 200, 'body': raw.model_dump()}}) for i in range(2))
            llm.clients[api] = NS(files=NS(create=AsyncMock(return_value=NS(id='file')), content=AsyncMock(return_value=NS(text=lines))), batches=NS(create=AsyncMock(return_value=NS(id='batch', status='completed', output_file_id='out', error_file_id=None))))
        else:
            async def entries():
                for i in range(2):
                    yield NS(custom_id=str(i), result=NS(type='succeeded', message=raw))
            llm.clients[api] = NS(messages=NS(batches=NS(create=AsyncMock(return_value=NS(id='batch', processing_status='ended')), results=AsyncMock(return_value=entries()))))
        await llm._run_batch(api, jobs)
        result = futures[1].result()
        assert result.text == 'ok'
        assert llm.cost == result.cost * 2 > 0
    asyncio.run(run())


def test_batch_queue_deadline_and_cancellation():
    async def run():
        llm = LLM(batch=True)
        llm._use_this_loop()
        first = asyncio.create_task(llm._batched('openai', 'gpt-5.5', {}))
        await asyncio.sleep(0)
        timer = llm.timers['openai']
        second = asyncio.create_task(llm._batched('openai', 'gpt-5.5', {}))
        await asyncio.sleep(0)
        assert llm.timers['openai'] is timer
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        llm._run_batch = AsyncMock()
        llm._send_batches('openai')
        assert not llm.running
        llm._run_batch.assert_not_called()
        await llm.aclose()
    asyncio.run(run())


def test_batch_extra_body_is_flat_json():
    async def run():
        llm = LLM(batch=True)
        result = NS(settings={}, model='', model_id='')
        llm._batched = AsyncMock(return_value=result)
        await llm.ask('gpt-5.5', MESSAGES, extra_body={'metadata': {'key': 'value'}})
        body = llm._batched.call_args.args[2]
        assert body['metadata'] == {'key': 'value'} and 'extra_body' not in body
        await llm.aclose()
    asyncio.run(run())


def test_price_includes_cache_categories_and_long_threshold():
    spec = {'price': {'input': 4, 'output': 20, 'cache_read': .4, 'cache_write': 5, 'cache_write_1h': 8}}
    assert price(spec, 1000, 2000, 3000, 500, 4000) == pytest.approx(.0618)
    spec['long_price'] = spec['price'] | {'above': 100, 'input': 8}
    assert price(spec, 100, 0, 0, 0) == .0004
    assert price(spec, 101, 0, 0, 0) == .000808


def test_owned_client_closed_and_timeout_configured(monkeypatch):
    client = NS(close=AsyncMock())
    factory = Mock(return_value=client)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-not-a-real-key')
    monkeypatch.setattr('llm.api.openai.AsyncOpenAI', factory)
    async def run():
        async with LLM(timeout=7) as llm:
            assert llm.client('openai') is client
        await llm.aclose()
        client.close.assert_awaited_once()
    asyncio.run(run())
    timeout = factory.call_args.kwargs['timeout']
    assert timeout.connect == timeout.write == timeout.pool == 7 and timeout.read == IDLE_TIMEOUT


def test_failed_openai_response_raises_but_keeps_usage():
    async def run():
        llm, _, _ = setup('openai', [response('openai', status='failed', error={'code': 'server_error', 'message': 'failure'})])
        with pytest.raises(RuntimeError):
            await llm.ask('gpt-5.5', MESSAGES)
        assert llm.cost > 0
    asyncio.run(run())


@pytest.mark.parametrize('api', ['openai', 'anthropic'])
def test_unpriced_service_tier_is_unknown(api):
    async def run():
        raw = response(api)
        if api == 'openai':
            raw.service_tier = 'priority'
        else:
            raw.usage.service_tier = 'priority'
        llm, _, _ = setup(api, [raw])
        result = await llm.ask(MODELS_BY_API[api], MESSAGES)
        assert result.cost is None and llm.cost is None
    asyncio.run(run())


def test_transport_loss_marks_total_unknown():
    import openai
    async def run():
        llm, _, call = setup('openai')
        call.side_effect = openai.APITimeoutError(request=httpx.Request('POST', 'https://example.com'))
        with pytest.raises(openai.APITimeoutError):
            await llm.ask('gpt-5.5', MESSAGES)
        assert llm.cost is None and llm.known_cost == 0
    asyncio.run(run())


def test_close_cancels_queued_calls_without_sending():
    async def run():
        llm = LLM(batch=True)
        pending = asyncio.create_task(llm.ask('gpt-5.5', MESSAGES))
        await asyncio.sleep(0)
        await llm.aclose()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert llm.cost == 0 and not llm.clients
    asyncio.run(run())


def test_instance_rejects_new_loop_without_losing_clients():
    llm, client, _ = setup('openai')
    async def run():
        await llm.ask('gpt-5.5', MESSAGES)
    asyncio.run(run())
    with pytest.raises(RuntimeError):
        asyncio.run(run())
    assert llm.clients['openai'] is client


@pytest.mark.parametrize('api', MODELS_BY_API)
def test_tool_round_trip_preserves_native_state(api):
    if api == 'anthropic':
        raw = response(api, stop_reason='tool_use', content=[
            {'type': 'thinking', 'thinking': 'inspect weather', 'signature': 'sig'},
            {'type': 'tool_use', 'id': 'call-1', 'name': 'weather', 'input': {}}])
    elif api == 'openai':
        raw = response(api, output=[{'type': 'reasoning', 'id': 'reason', 'summary': [], 'encrypted_content': 'encrypted'},
            {'type': 'function_call', 'id': 'fc-1', 'call_id': 'call-1', 'name': 'weather', 'arguments': '{}'}])
    else:
        raw = response(api, choices=[{'index': 0, 'finish_reason': 'tool_calls', 'message': {'role': 'assistant',
            'content': None, 'reasoning_details': [{'type': 'reasoning.text', 'text': 'inspect weather'}],
            'tool_calls': [{'id': 'call-1', 'type': 'function', 'function': {'name': 'weather', 'arguments': '{}'}}]}}])
    async def run():
        llm, _, call = setup(api, [raw, response(api)])
        first = await llm.ask(MODELS_BY_API[api], MESSAGES, tools=[TOOL])
        assert first.stop_reason == 'tool_calls' and first.tool_calls[0]['id'] == 'call-1'
        messages = [*MESSAGES, first.message, {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'sunny'}]
        second = await llm.ask(MODELS_BY_API[api], messages, tools=[TOOL])
        assert second.text == 'ok'
        body = call.call_args.kwargs
        if api == 'anthropic':
            assert body['messages'][1]['content'][0]['signature'] == 'sig'
        elif api == 'openai':
            assert body['input'][1]['encrypted_content'] == 'encrypted'
        else:
            assert body['messages'][1]['reasoning_details'][0]['text'] == 'inspect weather'
    asyncio.run(run())


def test_automatic_paid_cache_cannot_be_disabled():
    async def run():
        llm, _, call = setup('openrouter')
        with pytest.raises(ValueError):
            await llm.ask('qwen-3.7-plus', MESSAGES, cache=False)
        call.assert_not_awaited()
    asyncio.run(run())


def test_output_parse_failure_keeps_charge():
    async def run():
        raw = response('openai')
        raw.output[0].content = None  # A malformed response after valid billing usage arrived.
        llm, _, _ = setup('openai', [raw])
        with pytest.raises(KeyError):
            await llm.ask('gpt-5.5', MESSAGES)
        assert llm.cost > 0
    asyncio.run(run())


def test_close_just_scheduled_batch_does_not_leave_waiters():
    async def run():
        llm = LLM(batch=True)
        pending = asyncio.create_task(llm.ask('gpt-5.5', MESSAGES))
        await asyncio.sleep(0)
        llm._send_batches('openai')
        await llm.aclose()  # Close before the worker gets a chance to run.
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not llm.running
    asyncio.run(run())


@pytest.mark.parametrize('api', MODELS_BY_API)
def test_stalled_reply_is_sent_again(api, monkeypatch):
    monkeypatch.setattr('llm.api.asyncio.sleep', AsyncMock())
    async def run():
        stall = MidStream(raw=response(api), error=httpx.ReadTimeout('no data'))
        llm, _, call = setup(api, [stall, response(api)])
        reply = await llm.ask(MODELS_BY_API[api], MESSAGES)
        assert reply.text == 'ok' and call.await_count == 2
        assert llm.cost is None and llm.known_cost == reply.cost > 0  # The abandoned try may have been billed.
        assert reply.seconds >= reply.longest_gap >= 0
    asyncio.run(run())


def test_stall_retries_give_up():
    async def run():
        stall = MidStream(raw=response('openai'), error=httpx.RemoteProtocolError('dropped'))
        llm, _, call = setup('openai', [stall], max_retries=0)
        with pytest.raises(httpx.RemoteProtocolError):
            await llm.ask('gpt-5.5', MESSAGES)
        assert call.await_count == 1 and llm.cost is None
    asyncio.run(run())


def test_openrouter_stream_reassembles_fragments():
    async def run():
        chunks = [
            {'id': 'gen', 'provider': 'Host', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'He',
                'reasoning': 'thi', 'reasoning_details': [{'type': 'reasoning.text', 'text': 'thi', 'index': 0}],
                'tool_calls': [{'index': 0, 'id': 'call-1', 'type': 'function',
                                'function': {'name': 'weather', 'arguments': '{"ci'}}]}}]},
            {'id': 'gen', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'llo', 'reasoning': 'nk',
                'reasoning_details': [{'type': 'reasoning.text', 'text': 'nk', 'index': 0},
                                      {'type': 'reasoning.encrypted', 'data': 'xyz', 'index': 1}],
                'tool_calls': [{'index': 0, 'id': 'call-1', 'function': {'arguments': 'ty": 1}'}}]},
                'finish_reason': 'tool_calls'}]},
            {'id': 'gen', 'choices': [], 'usage': {'prompt_tokens': 1, 'completion_tokens': 2, 'cost': .5}},
        ]
        llm, client, _ = setup('openrouter')
        async def create(**kwargs):
            assert kwargs['stream'] is True
            return events([NS(to_dict=lambda c=c: c) for c in chunks])
        client.chat.completions.create = create
        reply = await llm.ask('qwen-3.7-plus', MESSAGES, tools=[TOOL])
        assert reply.text == 'Hello' and reply.reasoning == 'think' and reply.host == 'Host' and reply.cost == .5
        assert reply.tool_calls == [{'id': 'call-1', 'type': 'function',
                                     'function': {'name': 'weather', 'arguments': '{"city": 1}'}}]
        assert reply.message['reasoning_details'] == [{'type': 'reasoning.text', 'text': 'think', 'index': 0},
                                                      {'type': 'reasoning.encrypted', 'data': 'xyz', 'index': 1}]
        assert reply.stop_reason == 'tool_calls' and reply.id == 'gen'
    asyncio.run(run())


def test_merge_delta_appends_entries_without_index():
    acc = merge_delta({}, {'reasoning_details': [{'type': 'a', 'text': 'x'}]})
    merge_delta(acc, {'reasoning_details': [{'type': 'b', 'text': 'y'}]})
    assert acc == {'reasoning_details': [{'type': 'a', 'text': 'x'}, {'type': 'b', 'text': 'y'}]}


def test_openrouter_error_mid_stream_is_retried(monkeypatch):
    import openai
    monkeypatch.setattr('llm.api.asyncio.sleep', AsyncMock())
    async def run():
        error = openai.APIError('host failed', httpx.Request('POST', 'https://openrouter.ai'), body={'code': 502})
        llm, _, call = setup('openrouter', [MidStream(raw=response('openrouter'), error=error), response('openrouter')])
        reply = await llm.ask('qwen-3.7-plus', MESSAGES)
        assert reply.text == 'ok' and call.await_count == 2 and reply.cost == llm.cost == .24  # Usage came before the error.
    asyncio.run(run())


def test_failed_attempts_are_recorded_and_reported_costs_kept(monkeypatch):
    monkeypatch.setattr('llm.api.asyncio.sleep', AsyncMock())
    async def run():
        unreported = response('openrouter', choices=[], usage=None)
        reported = response('openrouter', choices=[])
        stall = MidStream(raw=response('openrouter'), error=httpx.ReadTimeout('no data'))
        llm, _, _ = setup('openrouter', [stall, unreported, reported, response('openrouter')])
        reply = await llm.ask('qwen-3.7-plus', MESSAGES)
        assert [a['cost'] for a in reply.failed_attempts] == [None, None, .12]
        assert 'ReadTimeout' in reply.failed_attempts[0]['error'] and 'mock' in reply.failed_attempts[1]['error']
        assert reply.cost == pytest.approx(.24)  # This try and the one reported failure; unreported ones aren't guessed.
        assert llm.cost is None and llm.known_cost == pytest.approx(.24)  # The stall cut off its usage.
    asyncio.run(run())


@pytest.mark.parametrize('api', MODELS_BY_API)
def test_partial_text_survives_cancellation(api):
    async def run():
        llm, client, _ = setup(api)
        started = asyncio.Event()
        async def slow():  # Streams some text, then hangs until canceled.
            if api == 'openai':
                yield NS(type='response.output_text.delta', delta='Hel')
            elif api in ('openrouter', 'abliteration'):
                yield NS(to_dict=lambda: {'id': 'gen', 'choices': [{'index': 0, 'delta': {'content': 'Hel', 'reasoning': 'r'}}]})
            else:
                yield NS(type='content_block_delta')
            started.set()
            await asyncio.Event().wait()
        if api == 'anthropic':
            class Stream:
                current_message_snapshot = NS(content=[NS(type='text', text='Hel')])
                def __aiter__(self):
                    return slow().__aiter__()
            @asynccontextmanager
            async def stream(**kwargs):
                yield Stream()
            client.messages.stream = stream
        else:
            async def create(**kwargs):
                return slow()
            client.responses.create = client.chat.completions.create = create
        task = asyncio.create_task(llm.ask(MODELS_BY_API[api], MESSAGES))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.partial['text'] == 'Hel'
    asyncio.run(run())


def test_abliteration_reasoning_cost_and_request(monkeypatch):
    async def run():
        chunks = [
            {'id': 'chatcmpl', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'reasoning_content': 'thi'}}]},
            {'id': 'chatcmpl', 'choices': [{'index': 0, 'delta': {'reasoning_content': 'nk', 'content': 'ok'},
                                            'finish_reason': 'stop'}]},
            {'id': 'chatcmpl', 'choices': [], 'usage': {'prompt_tokens': 1_000_000, 'completion_tokens': 1_000_000,
                                                        'prompt_tokens_details': {'cached_tokens': 500_000}}},
        ]
        llm, client, _ = setup('abliteration')
        sent = {}
        async def create(**kwargs):
            sent.update(kwargs)
            return events([NS(to_dict=lambda c=c: c) for c in chunks])
        client.chat.completions.create = create
        reply = await llm.ask('abliterated-large-v2', MESSAGES, reasoning='max', temperature=0)
        assert sent['model'] == 'abliterated-model-large-v2' and sent['temperature'] == 0
        assert sent['extra_body'] == {'reasoning_effort': 'max'}  # No OpenRouter host pinning.
        assert reply.text == 'ok' and reply.reasoning == 'think' and reply.host == 'abliteration.ai'
        assert reply.stop_reason == 'end' and reply.message == {'role': 'assistant', 'content': 'ok'}
        assert reply.cost == llm.cost == pytest.approx(1.5 + .15 + 5)  # Uncached input, cached input, output.
    asyncio.run(run())


def test_reasoning_none_rejected_where_it_cant_be_turned_off():
    async def run():
        llm, _, call = setup('abliteration')
        with pytest.raises(ValueError, match="can't turn its reasoning off"):
            await llm.ask('abliterated-large-v2', MESSAGES, reasoning='none')
        await llm.ask('abliterated-large', MESSAGES, reasoning='none')  # The older large model can.
        assert call.call_args.kwargs['extra_body'] == {'reasoning_effort': 'none'} and call.await_count == 1
    asyncio.run(run())


def test_abliteration_policy_block_is_not_retried(monkeypatch):
    import openai
    monkeypatch.setattr('llm.api.asyncio.sleep', AsyncMock())
    async def run():
        error = openai.APIError('blocked', httpx.Request('POST', 'https://api.abliteration.ai'),
                                body={'type': 'policy_error', 'code': 'policy_violation'})
        llm, _, call = setup('abliteration', [MidStream(raw=response('openrouter'), error=error), response('openrouter')])
        with pytest.raises(openai.APIError) as caught:
            await llm.ask('abliterated', MESSAGES)
        assert caught.value.code == 'policy_violation' and call.await_count == 1
    asyncio.run(run())
