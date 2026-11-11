import { ApiError, BusyError } from '../errors';
import { createApi } from '../client';
import session from '../__fixtures__/session_loaded.json';
import { jsonResponse } from './helpers';

function setup(...responses: (Response | Error)[]) {
  const queue = [...responses];
  const fetchFn = vi.fn<typeof fetch>(async () => {
    const next = queue.shift();
    if (next === undefined) throw new Error('unexpected request');
    if (next instanceof Error) throw next;
    return next;
  });
  const onStale = vi.fn();
  const onBusy = vi.fn();
  const api = createApi({ token: 'tok', fetchFn, onStale, onBusy });
  return { api, fetchFn, onStale, onBusy };
}

test('client sends X-Neshter-Token and parses the response', async () => {
  const { api, fetchFn } = setup(jsonResponse(200, session));
  const result = await api.session();
  expect(result).toMatchObject({ loaded: true, name: 'abc123' });
  const [url, init] = fetchFn.mock.calls[0] ?? [];
  expect(url).toBe('/api/session');
  expect((init?.headers as Record<string, string>)['X-Neshter-Token']).toBe('tok');
});

test('client maps error_code', async () => {
  const { api } = setup(jsonResponse(422, { detail: 'kötü', error_code: 'invalid_input' }));
  await expect(api.system()).rejects.toMatchObject({ status: 422, code: 'invalid_input', detail: 'kötü' });
});

test('unknown error codes and bodies fall back to the status', async () => {
  const { api } = setup(
    jsonResponse(500, { detail: 'x', error_code: 'weird_new_code' }),
    new Response('<html>nope</html>', { status: 404 }),
  );
  await expect(api.system()).rejects.toMatchObject({ status: 500, code: 'internal' });
  await expect(api.system()).rejects.toMatchObject({ status: 404, code: 'not_found' });
});

test('403 marks the session stale once', async () => {
  const body = { detail: 'Geçersiz', error_code: 'forbidden' };
  const { api, onStale } = setup(jsonResponse(403, body), jsonResponse(403, body));
  await expect(api.system()).rejects.toMatchObject({ code: 'forbidden', status: 403 });
  await expect(api.models()).rejects.toMatchObject({ code: 'forbidden' });
  expect(onStale).toHaveBeenCalledTimes(1);
});

test('409 busy throws BusyError and calls onBusy', async () => {
  const { api, onBusy } = setup(jsonResponse(409, { detail: 'meşgul', error_code: 'busy' }));
  const error = await api.session().catch((e: unknown) => e);
  expect(error).toBeInstanceOf(BusyError);
  expect(error).toBeInstanceOf(ApiError);
  expect(onBusy).toHaveBeenCalledTimes(1);
});

test('409 no_model is a plain ApiError', async () => {
  const { api, onBusy } = setup(jsonResponse(409, { detail: 'yok', error_code: 'no_model' }));
  const error = await api.session().catch((e: unknown) => e);
  expect(error).toBeInstanceOf(ApiError);
  expect(error).not.toBeInstanceOf(BusyError);
  expect((error as ApiError).code).toBe('no_model');
  expect(onBusy).not.toHaveBeenCalled();
});

test('network failure maps to network code', async () => {
  const { api, onStale } = setup(new TypeError('Failed to fetch'));
  await expect(api.system()).rejects.toMatchObject({ status: 0, code: 'network' });
  expect(onStale).not.toHaveBeenCalled();
});

test('a response that breaks the schema is an internal error', async () => {
  const { api } = setup(jsonResponse(200, { loaded: 'yes' }));
  await expect(api.session()).rejects.toMatchObject({ code: 'internal', status: 200 });
});

test('an aborted request rethrows the abort', async () => {
  const abort = new Error('aborted');
  abort.name = 'AbortError';
  const { api } = setup(abort);
  await expect(api.job('j1', new AbortController().signal)).rejects.toBe(abort);
});

test('POST routes send JSON and return the job id', async () => {
  const { api, fetchFn } = setup(jsonResponse(200, { job_id: 'j1' }));
  await expect(api.effect({ heads: [[0, 1]], n_max: 39 })).resolves.toEqual({ job_id: 'j1' });
  const [url, init] = fetchFn.mock.calls[0] ?? [];
  expect(url).toBe('/api/effect');
  expect(init?.method).toBe('POST');
  expect(init?.body).toBe(JSON.stringify({ heads: [[0, 1]], n_max: 39 }));
  expect((init?.headers as Record<string, string>)['Content-Type']).toBe('application/json');
});

test('draft routes encode the sha and handle 204', async () => {
  const sha = 'a'.repeat(64);
  const { api, fetchFn } = setup(new Response(null, { status: 204 }), jsonResponse(404, { detail: 'yok', error_code: 'not_found' }));
  await expect(api.deleteDraft(sha)).resolves.toBeUndefined();
  expect(fetchFn.mock.calls[0]?.[0]).toBe(`/api/draft/${sha}`);
  expect(fetchFn.mock.calls[0]?.[1]?.method).toBe('DELETE');
  await expect(api.draft(sha)).rejects.toMatchObject({ code: 'not_found' });
});

test('job ids are URL-encoded', async () => {
  const { api, fetchFn } = setup(jsonResponse(404, { detail: 'yok', error_code: 'not_found' }));
  await api.cancelJob('a/b').catch(() => undefined);
  expect(fetchFn.mock.calls[0]?.[0]).toBe('/api/jobs/a%2Fb/cancel');
});
