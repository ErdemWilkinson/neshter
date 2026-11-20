import { useEffect, useState } from 'react';
import { ApiError, BusyError, type ErrorCode } from '@/api/errors';
import { sessionStore } from '@/core/stores/session';

export type Insight<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; data: T }
  | { status: 'failed'; code: ErrorCode; detail: unknown };

/** A busy model (409) is usually a short job, e.g. the heatmap re-running for a new token: wait and ask again. */
export const BUSY_RETRY_MS = 800;
export const BUSY_RETRIES = 20;

/**
 * Runs `load` while `open` and whenever `key` changes; a late answer for an older key is dropped. Busy (409) is
 * retried quietly for a while (the panel keeps "loading"); only if the model stays busy does it hand over to the
 * banner and go back to idle.
 */
export function useInsight<T>(open: boolean, key: string, load: () => Promise<T>): Insight<T> {
  // Only async callbacks set state; "loading" is derived: open, and no answer yet for this key.
  const [answer, setAnswer] = useState<{ key: string; value: Insight<T> } | null>(null);
  useEffect(() => {
    if (!open) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const attempt = (left: number): void => {
      load().then(
        (data) => {
          if (live) setAnswer({ key, value: { status: 'ready', data } });
        },
        (error: unknown) => {
          if (!live) return;
          if (error instanceof BusyError) {
            if (left > 0) {
              timer = setTimeout(() => attempt(left - 1), BUSY_RETRY_MS);
              return;
            }
            sessionStore.getState().setBusy(true);
            setAnswer({ key, value: { status: 'idle' } });
          } else if (error instanceof ApiError) setAnswer({ key, value: { status: 'failed', code: error.code, detail: error.detail } });
          else setAnswer({ key, value: { status: 'failed', code: 'internal', detail: String(error) } });
        },
      );
    };
    attempt(BUSY_RETRIES);
    return () => {
      live = false;
      clearTimeout(timer);
    };
    // `load` is a fresh closure every render; `key` names what it loads.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, key]);
  if (!open) return { status: 'idle' };
  return answer?.key === key ? answer.value : { status: 'loading' };
}
