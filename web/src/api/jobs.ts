import { clearActiveJob, setActiveJob } from '@/app/sessionStorage';
import type { Api } from './client';
import { ApiError, isAbortError } from './errors';
import { jobStateSchema, type JobState } from './schemas';
import { z } from '@/core/zod';

export type { JobState } from './schemas';

const anyJob = jobStateSchema(z.unknown());

export const isTerminal = (s: Pick<JobState, 'state'>): boolean =>
  s.state === 'done' || s.state === 'error' || s.state === 'cancelled';

export interface AttachOptions {
  /** Stream drops without a terminal event before giving up (reset whenever a stream delivered events). */
  maxReconnects?: number;
  retryMs?: number;
}

function abortError(): Error {
  const error = new Error('Aborted');
  error.name = 'AbortError';
  return error;
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(abortError());
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError());
    };
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal.addEventListener('abort', onAbort, { once: true });
  });
}

/**
 * Follows a job to its end: `GET /api/jobs/{id}` first (so a reload or a late attach sees the current state), then
 * the event stream. A stream that ends without a terminal event is re-attached after `retryMs` (default 5 tries,
 * 1000 ms). Resolves with the terminal state (`done`, `error` or `cancelled`: errors are states, not rejections).
 * Rejects with `ApiError('network')` when the budget is spent, with the API error for anything else (a 403 must
 * not loop), and with an `AbortError` when `signal` fires. The id is in `neshter.activeJob` while it runs; it is
 * cleared at the end, but kept on abort because the job may still run (a reload can re-attach).
 */
export async function attachJob<R>(
  api: Api,
  id: string,
  onUpdate: (state: JobState<R>) => void,
  signal: AbortSignal,
  opts: AttachOptions = {},
): Promise<JobState<R>> {
  const maxReconnects = opts.maxReconnects ?? 5;
  const retryMs = opts.retryMs ?? 1000;
  const finish = (state: JobState<R>): JobState<R> => {
    clearActiveJob();
    return state;
  };
  setActiveJob(id);
  try {
    let reconnects = 0;
    for (;;) {
      if (signal.aborted) throw abortError();
      const current = await api.job<R>(id, signal);
      onUpdate(current);
      if (isTerminal(current)) return finish(current);

      const seen: { events: number; final: JobState<R> | null } = { events: 0, final: null };
      try {
        await api.streamJob(id, {
          signal,
          onEvent: (_event, data) => {
            const parsed = anyJob.safeParse(data);
            if (!parsed.success) return;
            const state = parsed.data as JobState<R>;
            seen.events += 1;
            onUpdate(state);
            if (isTerminal(state)) seen.final = state;
          },
        });
      } catch (error) {
        if (!(error instanceof ApiError && error.code === 'network')) throw error;
      }
      if (signal.aborted) throw abortError();
      if (seen.final) return finish(seen.final);

      reconnects = seen.events > 0 ? 1 : reconnects + 1;
      if (reconnects > maxReconnects) throw new ApiError(0, 'network', 'The job event stream was lost.');
      await sleep(retryMs, signal);
    }
  } catch (error) {
    if (isAbortError(error)) throw error;
    clearActiveJob();
    throw error;
  }
}
