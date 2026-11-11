import { ApiError } from '../errors';
import { streamSse } from '../sse';
import { abortError, jsonResponse, sseResponse } from './helpers';

type Seen = [string, unknown][];

async function run(chunks: string[]) {
  const events: Seen = [];
  const fetchFn = vi.fn(async () => sseResponse(chunks));
  const outcome = await streamSse('/api/jobs/j/events', {
    token: 'tok',
    signal: new AbortController().signal,
    onEvent: (event, data) => events.push([event, data]),
    fetchFn: fetchFn as unknown as typeof fetch,
  });
  return { events, outcome, fetchFn };
}

test('sse parser handles split chunks, CRLF and keepalive comments', async () => {
  const { events, outcome } = await run([
    'event: progress\r\nda',
    'ta: {"progress":0.5}\r\n\r\n: keepalive\n\n',
    'event: done\ndata: {"state":"done"}\n\n',
  ]);
  expect(events).toEqual([
    ['progress', { progress: 0.5 }],
    ['done', { state: 'done' }],
  ]);
  expect(outcome).toBe('ended');
});

test('sse parser keeps a CRLF split across chunks as one line break', async () => {
  const { events } = await run(['event: a\r', '\ndata: 1\r', '\n\r', '\n', 'event: b\ndata: 2\n\n']);
  expect(events).toEqual([
    ['a', 1],
    ['b', 2],
  ]);
});

test('sse parser joins multi-line data, defaults the event name and keeps non-JSON text', async () => {
  const { events } = await run(['data: first\ndata: second\n\n', 'event: x\ndata:   spaced\n\n', 'data: {bad json\n\n']);
  expect(events).toEqual([
    ['message', 'first\nsecond'],
    ['x', '  spaced'],
    ['message', '{bad json'],
  ]);
});

test('sse parser drops an event that never got its blank line and blocks without data', async () => {
  const { events } = await run(['event: lonely\n\n', 'event: cut\ndata: 1']);
  expect(events).toEqual([]);
});

test('sse sends the token and asks for an event stream', async () => {
  const { fetchFn } = await run([]);
  const call = fetchFn.mock.calls[0] as unknown as [string, RequestInit];
  expect(call[0]).toBe('/api/jobs/j/events');
  const headers = call[1].headers as Record<string, string>;
  expect(headers['X-Neshter-Token']).toBe('tok');
  expect(headers['Accept']).toBe('text/event-stream');
});

test('sse throws the mapped ApiError on a non-OK response', async () => {
  const fetchFn = vi.fn(async () => jsonResponse(403, { detail: 'x', error_code: 'forbidden' }));
  await expect(
    streamSse('/e', { token: 't', signal: new AbortController().signal, onEvent: vi.fn(), fetchFn: fetchFn as unknown as typeof fetch }),
  ).rejects.toMatchObject({ code: 'forbidden', status: 403 });
});

test('sse maps a failed connection to the network code', async () => {
  const fetchFn = vi.fn(async () => {
    throw new TypeError('Failed to fetch');
  });
  const error = await streamSse('/e', {
    token: 't',
    signal: new AbortController().signal,
    onEvent: vi.fn(),
    fetchFn: fetchFn as unknown as typeof fetch,
  }).catch((e: unknown) => e);
  expect(error).toBeInstanceOf(ApiError);
  expect(error).toMatchObject({ code: 'network', status: 0 });
});

test('sse resolves "aborted" when the signal fires while reading', async () => {
  const controller = new AbortController();
  const fetchFn = vi.fn(async (_url: string, init: RequestInit) => {
    const body = new ReadableStream<Uint8Array>({
      start(stream) {
        init.signal?.addEventListener('abort', () => stream.error(abortError()));
      },
    });
    return new Response(body, { status: 200 });
  });
  const pending = streamSse('/e', {
    token: 't',
    signal: controller.signal,
    onEvent: vi.fn(),
    fetchFn: fetchFn as unknown as typeof fetch,
  });
  controller.abort();
  await expect(pending).resolves.toBe('aborted');
});

test('sse treats a connection reset mid-stream as ended', async () => {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(stream) {
      stream.enqueue(encoder.encode('event: progress\ndata: 1\n\n'));
      stream.error(new TypeError('terminated'));
    },
  });
  const events: Seen = [];
  const outcome = await streamSse('/e', {
    token: 't',
    signal: new AbortController().signal,
    onEvent: (e, d) => events.push([e, d]),
    fetchFn: (async () => new Response(body, { status: 200 })) as unknown as typeof fetch,
  });
  expect(outcome).toBe('ended');
  expect(events.length).toBeLessThanOrEqual(1);
});

test('an error thrown by onEvent surfaces and is not mistaken for a dropped connection', async () => {
  const boom = new TypeError('consumer bug');
  const fetchFn = vi.fn(async () => sseResponse(['event: progress\ndata: 1\n\n', 'event: progress\ndata: 2\n\n']));
  const onEvent = vi.fn(() => {
    throw boom;
  });
  await expect(
    streamSse('/e', { token: 't', signal: new AbortController().signal, onEvent, fetchFn: fetchFn as unknown as typeof fetch }),
  ).rejects.toBe(boom);
  expect(onEvent).toHaveBeenCalledTimes(1);
});
