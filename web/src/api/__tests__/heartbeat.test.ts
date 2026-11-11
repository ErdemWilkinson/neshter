import { createApi } from '../client';
import { startHeartbeat } from '../heartbeat';
import session from '../__fixtures__/session_loaded.json';
import { jsonResponse } from './helpers';

type Step = Response | Error;

function setup(steps: Step[]) {
  const queue = [...steps];
  const fetchFn = vi.fn(async () => {
    const next = queue.shift();
    if (!next) throw new Error('no more steps');
    if (next instanceof Error) throw next;
    return next;
  });
  const onStale = vi.fn();
  const api = createApi({ token: 't', fetchFn: fetchFn as unknown as typeof fetch, onStale });
  return { api, fetchFn, onStale };
}

const down = () => new TypeError('Failed to fetch');
const forbidden = () => jsonResponse(403, { detail: 'x', error_code: 'forbidden' });

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

test('heartbeat keeps polling after onDown and a later 403 reaches onStale', async () => {
  const { api, fetchFn, onStale } = setup([down(), down(), forbidden()]);
  const onDown = vi.fn();
  const stop = startHeartbeat(api, onDown, 10_000);

  await vi.advanceTimersByTimeAsync(10_000);
  expect(onDown).toHaveBeenCalledTimes(1);
  await vi.advanceTimersByTimeAsync(10_000);
  expect(onDown).toHaveBeenCalledTimes(1); // same outage
  expect(onStale).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(10_000);
  expect(onStale).toHaveBeenCalledTimes(1);
  expect(fetchFn).toHaveBeenCalledTimes(3);
  stop();
});

test('heartbeat reports a recovery and a second outage separately', async () => {
  const ok = () => jsonResponse(200, session);
  const { api } = setup([down(), ok(), ok(), down()]);
  const onDown = vi.fn();
  const onUp = vi.fn();
  const stop = startHeartbeat(api, onDown, { intervalMs: 1000, onUp });

  await vi.advanceTimersByTimeAsync(1000);
  expect([onDown.mock.calls.length, onUp.mock.calls.length]).toEqual([1, 0]);
  await vi.advanceTimersByTimeAsync(1000);
  expect([onDown.mock.calls.length, onUp.mock.calls.length]).toEqual([1, 1]);
  await vi.advanceTimersByTimeAsync(1000); // healthy again: no repeated onUp
  expect(onUp).toHaveBeenCalledTimes(1);
  await vi.advanceTimersByTimeAsync(1000);
  expect(onDown).toHaveBeenCalledTimes(2);
  stop();
});

test('heartbeat uses a 10 s default interval and stops when asked', async () => {
  const { api, fetchFn } = setup([jsonResponse(200, session), jsonResponse(200, session)]);
  const stop = startHeartbeat(api, vi.fn());
  await vi.advanceTimersByTimeAsync(9_999);
  expect(fetchFn).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(1);
  expect(fetchFn).toHaveBeenCalledTimes(1);
  stop();
  await vi.advanceTimersByTimeAsync(60_000);
  expect(fetchFn).toHaveBeenCalledTimes(1);
});

test('other errors (busy, 500) do not count as an outage', async () => {
  const { api } = setup([jsonResponse(500, { detail: 'x', error_code: 'internal' }), jsonResponse(409, { detail: 'x', error_code: 'busy' })]);
  const onDown = vi.fn();
  const stop = startHeartbeat(api, onDown, 1000);
  await vi.advanceTimersByTimeAsync(2000);
  expect(onDown).not.toHaveBeenCalled();
  stop();
});

test('a poll that is still in flight when stop() is called does not call onUp or onDown', async () => {
  let release: (r: Response) => void = () => undefined;
  const pending = new Promise<Response>((resolve) => (release = resolve));
  const steps: (() => Promise<Response>)[] = [
    async () => {
      throw down();
    },
    () => pending,
  ];
  const fetchFn = vi.fn(async () => {
    const next = steps.shift();
    if (!next) throw new Error('no more steps');
    return next();
  });
  const api = createApi({ token: 't', fetchFn: fetchFn as unknown as typeof fetch, onStale: vi.fn() });
  const onDown = vi.fn();
  const onUp = vi.fn();
  const stop = startHeartbeat(api, onDown, { intervalMs: 1000, onUp });

  await vi.advanceTimersByTimeAsync(1000); // outage
  expect(onDown).toHaveBeenCalledTimes(1);
  await vi.advanceTimersByTimeAsync(1000); // second poll is now in flight
  stop();
  release(jsonResponse(200, session));
  await vi.advanceTimersByTimeAsync(5000);
  expect(onUp).not.toHaveBeenCalled();
  expect(fetchFn).toHaveBeenCalledTimes(2);
});
