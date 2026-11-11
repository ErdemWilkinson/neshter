// A stand-in for `fetch` that answers the Task 7 boot routes from the contract fixtures (not a test file itself).
import prefs from '@/api/__fixtures__/prefs.json';
import sessionLoaded from '@/api/__fixtures__/session_loaded.json';
import system from '@/api/__fixtures__/system.json';
import { jsonResponse } from '@/api/__tests__/helpers';

export { jsonResponse };
export { prefs, sessionLoaded, system };

export type Handler = (init: RequestInit | undefined) => Response | Promise<Response>;

export interface Call {
  key: string;
  init: RequestInit | undefined;
}

/** `overrides` are keyed `METHOD /path` and replace a default; unknown routes answer 404. */
export function fakeServer(overrides: Record<string, Handler> = {}) {
  const calls: Call[] = [];
  const defaults: Record<string, Handler> = {
    'GET /api/system': () => jsonResponse(200, system),
    'GET /api/session': () => jsonResponse(200, sessionLoaded),
    'GET /api/prefs': () => jsonResponse(200, prefs),
    'PUT /api/prefs': (init) => jsonResponse(200, JSON.parse(String(init?.body))),
    'GET /api/jobs': () => jsonResponse(200, []),
    'GET /api/drafts': () => jsonResponse(200, []),
    'POST /api/shutdown': () => jsonResponse(200, { ok: true }),
  };
  const handlers = { ...defaults, ...overrides };
  const fetchFn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const key = `${init?.method ?? 'GET'} ${String(input)}`;
    calls.push({ key, init });
    const handler = handlers[key];
    return handler ? handler(init) : jsonResponse(404, { detail: `no handler for ${key}`, error_code: 'not_found' });
  });
  return { fetchFn, calls, count: (key: string) => calls.filter((call) => call.key === key).length };
}

export function headersOf(init: RequestInit | undefined): Record<string, string> {
  return (init?.headers ?? {}) as Record<string, string>;
}

export function bodyOf(init: RequestInit | undefined): unknown {
  return JSON.parse(String(init?.body));
}
