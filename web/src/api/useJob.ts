import { useCallback, useEffect, useRef, useState } from 'react';
import { jobsStore } from '@/core/stores/jobs';
import { getApi } from './currentApi';
import { isAbortError } from './errors';
import { attachJob, isTerminal } from './jobs';
import type { JobState } from './schemas';

export interface UseJob<R> {
  job: JobState<R> | null;
  /** Starts a job (`begin` posts it) and follows it to its end; resolves with the terminal state. */
  start(begin: () => Promise<{ job_id: string }>): Promise<JobState<R>>;
  cancel(): Promise<void>;
  /** Forgets the job in this component (a job that still runs on the server is not stopped; use `cancel`). */
  reset(): void;
}

/**
 * Runs a job and mirrors the one that holds the model into `jobsStore.active` (set as soon as the job exists,
 * cleared on a terminal state). Leaving the screen does not stop following: the lock in the store stays truthful.
 * Job results are typed but not runtime-validated here: validate them with the result schema before use.
 */
export function useJob<R = unknown>(): UseJob<R> {
  const [job, setJob] = useState<JobState<R> | null>(null);
  const mounted = useRef(true);
  const current = useRef<{ id: string; abort: AbortController } | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const start = useCallback(async (begin: () => Promise<{ job_id: string }>): Promise<JobState<R>> => {
    const api = getApi();
    const { job_id: id } = await begin();
    current.current?.abort.abort(); // only once the new job exists: a refused start (409) keeps following the old one
    const abort = new AbortController();
    const mine = { id, abort };
    current.current = mine;
    const store = jobsStore.getState();
    // Between "posted" and the first state from the server the model is already taken.
    store.setActive({
      id,
      kind: '',
      state: 'queued',
      progress: 0,
      message: '',
      result: null,
      error: null,
      error_code: null,
      started_at: null,
    });
    const publish = (state: JobState<R>) => {
      if (current.current !== mine) return; // replaced or reset: a late update must not touch the newer job
      if (mounted.current) setJob(state);
      if (isTerminal(state)) jobsStore.getState().clearActive(id);
      else jobsStore.getState().setActive(state);
    };
    try {
      const final = await attachJob<R>(api, id, publish, abort.signal);
      jobsStore.getState().clearActive(id);
      if (current.current === mine) current.current = null;
      return final;
    } catch (error) {
      jobsStore.getState().clearActive(id);
      if (!isAbortError(error) && current.current === mine) current.current = null;
      throw error;
    }
  }, []);

  const cancel = useCallback(async () => {
    const running = current.current;
    if (running) await getApi().cancelJob(running.id);
  }, []);

  const reset = useCallback(() => {
    const running = current.current;
    if (running) {
      running.abort.abort();
      jobsStore.getState().clearActive(running.id);
      current.current = null;
    }
    setJob(null);
  }, []);

  return { job, start, cancel, reset };
}
