import { act, renderHook, waitFor } from '@testing-library/react';
import { getSessionStorage } from '@/app/sessionStorage';
import { isLocked, jobsStore } from '@/core/stores/jobs';
import type { Api } from '../client';
import { setApi } from '../currentApi';
import type { JobState } from '../schemas';
import { useJob } from '../useJob';

type Result = { path: string };

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

/** `job` replays `states`; the event stream stays open until `release()` (or an abort). */
function fakeApi(states: JobState<Result>[]) {
  let release: () => void = () => undefined;
  const job = vi.fn(async () => states.shift() ?? state({}));
  const streamJob = vi.fn(
    (_id: string, opts: { signal: AbortSignal }) =>
      new Promise<'ended' | 'aborted'>((resolve) => {
        release = () => resolve('ended');
        opts.signal.addEventListener('abort', () => resolve('aborted'), { once: true });
      }),
  );
  const cancelJob = vi.fn(async () => state({ state: 'cancelled' }));
  return { api: { job, streamJob, cancelJob } as unknown as Api, job, cancelJob, release: () => release() };
}

beforeEach(() => {
  getSessionStorage().clear();
  jobsStore.setState({ active: null });
});
afterEach(() => setApi(null));

test('useJob writes the active job into jobsStore and clears it on done', async () => {
  const done = state({ state: 'done', progress: 1, result: { path: 'C:\\x' } });
  const fake = fakeApi([state({ progress: 0.2 }), done]);
  setApi(fake.api);
  const { result } = renderHook(() => useJob<Result>());
  expect(result.current.job).toBeNull();

  let finished: Promise<JobState<Result>> | undefined;
  act(() => {
    finished = result.current.start(async () => ({ job_id: 'j1' }));
  });
  await waitFor(() => expect(result.current.job?.progress).toBe(0.2));
  expect(jobsStore.getState().active?.id).toBe('j1');
  expect(isLocked(jobsStore.getState())).toBe(true);

  await act(async () => {
    fake.release(); // stream ends -> re-attach -> the next GET says done
    await finished;
  });
  expect(await finished).toBe(done);
  expect(result.current.job).toBe(done);
  expect(jobsStore.getState().active).toBeNull();
  expect(isLocked(jobsStore.getState())).toBe(false);
});

test('the model counts as taken as soon as the job is posted', async () => {
  const fake = fakeApi([state({})]);
  setApi(fake.api);
  const { result } = renderHook(() => useJob<Result>());
  let finished: Promise<JobState<Result>> | undefined;
  act(() => {
    finished = result.current.start(async () => ({ job_id: 'j1' }));
  });
  await waitFor(() => expect(isLocked(jobsStore.getState())).toBe(true));
  act(() => result.current.reset());
  await expect(finished).rejects.toMatchObject({ name: 'AbortError' });
  expect(jobsStore.getState().active).toBeNull();
  expect(result.current.job).toBeNull();
});

test('a start that is refused leaves the store and the job untouched', async () => {
  setApi(fakeApi([]).api);
  const { result } = renderHook(() => useJob<Result>());
  await act(async () => {
    await expect(result.current.start(async () => Promise.reject(new Error('busy')))).rejects.toThrow('busy');
  });
  expect(jobsStore.getState().active).toBeNull();
  expect(result.current.job).toBeNull();
});

test('cancel asks the server to cancel the running job, and is a no-op without one', async () => {
  const fake = fakeApi([state({})]);
  setApi(fake.api);
  const { result } = renderHook(() => useJob<Result>());
  await act(() => result.current.cancel());
  expect(fake.cancelJob).not.toHaveBeenCalled();

  let finished: Promise<JobState<Result>> | undefined;
  act(() => {
    finished = result.current.start(async () => ({ job_id: 'j1' }));
  });
  await waitFor(() => expect(result.current.job).not.toBeNull());
  await act(() => result.current.cancel());
  expect(fake.cancelJob).toHaveBeenCalledWith('j1');
  act(() => result.current.reset());
  await expect(finished).rejects.toMatchObject({ name: 'AbortError' });
});

test('unmounting does not stop following: the store still ends unlocked', async () => {
  const done = state({ state: 'done', progress: 1 });
  const fake = fakeApi([state({}), done]);
  setApi(fake.api);
  const { result, unmount } = renderHook(() => useJob<Result>());
  let finished: Promise<JobState<Result>> | undefined;
  act(() => {
    finished = result.current.start(async () => ({ job_id: 'j1' }));
  });
  await waitFor(() => expect(isLocked(jobsStore.getState())).toBe(true));
  unmount();
  fake.release();
  expect(await finished).toBe(done);
  expect(jobsStore.getState().active).toBeNull();
});

test('an error state ends the job and unlocks', async () => {
  const failed = state({ state: 'error', error: 'x', error_code: 'oom' });
  setApi(fakeApi([failed]).api);
  const { result } = renderHook(() => useJob<Result>());
  await act(async () => {
    expect(await result.current.start(async () => ({ job_id: 'j1' }))).toBe(failed);
  });
  expect(result.current.job).toBe(failed);
  expect(isLocked(jobsStore.getState())).toBe(false);
});
