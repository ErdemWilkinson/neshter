import type { Api } from '../client';
import { ApiError } from '../errors';
import { attachJob } from '../jobs';
import type { JobState } from '../schemas';
import { getSessionStorage } from '@/app/sessionStorage';

type Result = { path: string };
type StreamOpts = { signal: AbortSignal; onEvent: (event: string, data: unknown) => void };

const state = (patch: Partial<JobState<Result>>): JobState<Result> => ({
  id: 'j1',
  kind: 'export',
  state: 'running',
  progress: 0,
  message: '',
  result: null,
  error: null,
  error_code: null,
  started_at: null,
  ...patch,
});

function fakeApi(jobs: JobState<Result>[], streams: ((opts: StreamOpts) => Promise<'ended' | 'aborted'>)[]) {
  const job = vi.fn(async () => {
    const next = jobs.shift();
    if (!next) throw new Error('no more job states');
    return next;
  });
  const streamJob = vi.fn(async (_id: string, opts: StreamOpts) => {
    const next = streams.shift();
    if (!next) throw new Error('no more streams');
    return next(opts);
  });
  return { api: { job, streamJob } as unknown as Api, job, streamJob };
}

const active = () => getSessionStorage().getItem('neshter.activeJob');
const fast = { retryMs: 0 };

beforeEach(() => getSessionStorage().clear());

test('attachJob re-attaches after the stream drops and resolves on done', async () => {
  const done = state({ state: 'done', progress: 1, result: { path: 'C:\\x' } });
  const { api, job, streamJob } = fakeApi(
    [state({ progress: 0.1 }), done],
    [
      async ({ onEvent }) => {
        onEvent('progress', state({ progress: 0.5 }));
        return 'ended';
      },
    ],
  );
  const seen: number[] = [];
  const result = await attachJob<Result>(api, 'j1', (s) => seen.push(s.progress), new AbortController().signal, fast);
  expect(result).toBe(done);
  expect(seen).toEqual([0.1, 0.5, 1]);
  expect(job).toHaveBeenCalledTimes(2);
  expect(streamJob).toHaveBeenCalledTimes(1);
});

test('attachJob resolves from a terminal stream event without another GET', async () => {
  const { api, job } = fakeApi(
    [state({})],
    [
      async ({ onEvent }) => {
        onEvent('progress', state({ progress: 0.9 }));
        onEvent('done', state({ state: 'done', progress: 1, result: { path: 'p' } }));
        return 'ended';
      },
    ],
  );
  const result = await attachJob<Result>(api, 'j1', () => undefined, new AbortController().signal, fast);
  expect(result.state).toBe('done');
  expect(result.result).toEqual({ path: 'p' });
  expect(job).toHaveBeenCalledTimes(1);
});

test('attachJob resolves error and cancelled states instead of rejecting', async () => {
  const failed = state({ state: 'error', error: 'bozuldu', error_code: 'disk_full' });
  const { api } = fakeApi([failed], []);
  await expect(attachJob(api, 'j1', () => undefined, new AbortController().signal, fast)).resolves.toBe(failed);

  const { api: api2 } = fakeApi(
    [state({})],
    [async ({ onEvent }) => (onEvent('done', state({ state: 'cancelled' })), 'ended' as const)],
  );
  await expect(attachJob(api2, 'j1', () => undefined, new AbortController().signal, fast)).resolves.toMatchObject({
    state: 'cancelled',
  });
});

test('attachJob stores and clears neshter.activeJob', async () => {
  const seenWhileRunning: (string | null)[] = [];
  const { api } = fakeApi(
    [state({}), state({ state: 'done', progress: 1 })],
    [async () => (seenWhileRunning.push(active()), 'ended' as const)],
  );
  expect(active()).toBeNull();
  await attachJob(api, 'j1', () => undefined, new AbortController().signal, fast);
  expect(seenWhileRunning).toEqual(['j1']);
  expect(active()).toBeNull();
});

test('attachJob gives up after maxReconnects and clears the active job', async () => {
  const dropped = async () => 'ended' as const;
  const { api, streamJob } = fakeApi(
    Array.from({ length: 10 }, () => state({})),
    Array.from({ length: 10 }, () => dropped),
  );
  const error = await attachJob(api, 'j1', () => undefined, new AbortController().signal, {
    maxReconnects: 2,
    retryMs: 0,
  }).catch((e: unknown) => e);
  expect(error).toBeInstanceOf(ApiError);
  expect(error).toMatchObject({ code: 'network' });
  expect(streamJob).toHaveBeenCalledTimes(3);
  expect(active()).toBeNull();
});

test('events reset the reconnect budget', async () => {
  const progressThenDrop = (progress: number) => async ({ onEvent }: StreamOpts) => {
    onEvent('progress', state({ progress }));
    return 'ended' as const;
  };
  const { api } = fakeApi(
    [state({}), state({}), state({}), state({ state: 'done', progress: 1 })],
    [progressThenDrop(0.2), progressThenDrop(0.4), progressThenDrop(0.6)],
  );
  const result = await attachJob(api, 'j1', () => undefined, new AbortController().signal, { maxReconnects: 1, retryMs: 0 });
  expect(result.state).toBe('done');
});

test('a network error while streaming is retried like a dropped stream', async () => {
  const { api } = fakeApi(
    [state({}), state({ state: 'done', progress: 1 })],
    [
      async () => {
        throw new ApiError(0, 'network');
      },
    ],
  );
  await expect(attachJob(api, 'j1', () => undefined, new AbortController().signal, fast)).resolves.toMatchObject({
    state: 'done',
  });
});

test('other stream errors reject (a stale token must not loop)', async () => {
  const { api } = fakeApi(
    [state({})],
    [
      async () => {
        throw new ApiError(403, 'forbidden');
      },
    ],
  );
  await expect(attachJob(api, 'j1', () => undefined, new AbortController().signal, fast)).rejects.toMatchObject({
    code: 'forbidden',
  });
  expect(active()).toBeNull();
});

test('abort rejects with AbortError and keeps the job id for a later re-attach', async () => {
  const controller = new AbortController();
  const { api } = fakeApi(
    [state({})],
    [
      async () => {
        controller.abort();
        return 'aborted';
      },
    ],
  );
  await expect(attachJob(api, 'j1', () => undefined, controller.signal, fast)).rejects.toMatchObject({
    name: 'AbortError',
  });
  expect(active()).toBe('j1');
});

test('ignores stream events that are not job states', async () => {
  const { api } = fakeApi(
    [state({}), state({ state: 'done', progress: 1 })],
    [
      async ({ onEvent }) => {
        onEvent('progress', 'garbage');
        onEvent('progress', { nope: true });
        return 'ended';
      },
    ],
  );
  const seen: number[] = [];
  await attachJob(api, 'j1', (s) => seen.push(s.progress), new AbortController().signal, fast);
  expect(seen).toEqual([0, 1]);
});
