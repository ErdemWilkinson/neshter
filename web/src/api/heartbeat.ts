import type { Api } from './client';
import { ApiError } from './errors';

export interface HeartbeatOptions {
  intervalMs?: number;
  /** A poll succeeded after `onDown` fired (the same server is reachable again). */
  onUp?: () => void;
}

/**
 * Polls `GET /api/session` every interval. A 403 already reaches the api's `onStale` (a restarted server with a new
 * token). A network error calls `onDown` once per outage; polling continues so a server that comes back on the
 * same origin is noticed: with a new token the next poll is a 403 (stale moves from "down" to "restarted"), with
 * the same token `onUp` fires. Returns a function that stops polling.
 */
export function startHeartbeat(api: Api, onDown: () => void, options: number | HeartbeatOptions = {}): () => void {
  const { intervalMs = 10_000, onUp } = typeof options === 'number' ? { intervalMs: options } : options;
  let stopped = false;
  let down = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const poll = async () => {
    try {
      await api.session();
      if (down) {
        down = false;
        if (!stopped) onUp?.();
      }
    } catch (error) {
      if (error instanceof ApiError && error.code === 'network') {
        if (!down) {
          down = true;
          if (!stopped) onDown();
        }
      } else if (error instanceof ApiError) {
        down = false; // the server answered (403, 500, busy...): it is up
      }
    }
    if (!stopped) timer = setTimeout(() => void poll(), intervalMs);
  };

  timer = setTimeout(() => void poll(), intervalMs);
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}
