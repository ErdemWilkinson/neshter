import type { Api } from '@/api/client';
import { ApiError } from '@/api/errors';
import type { LoadedSession } from '@/api/schemas';
import { sessionLoaded } from '@/app/__tests__/fakeServer';
import { createBasketStore } from '@/core/stores/basket';
import { createSessionStore } from '@/core/stores/session';
import { head } from '@/core/__tests__/fixtures/items';
import { startAutosave } from '../autosave';
import { draftStatusStore } from '../draftStatus';

const model = sessionLoaded as unknown as LoadedSession;
const SHA = model.source_sha256;

function setup(prompts: string[] = ['merhaba']) {
  const basket = createBasketStore();
  const session = createSessionStore();
  session.getState().setModel(model);
  const api = { putDraft: vi.fn<Api['putDraft']>(async () => ({}) as never), deleteDraft: vi.fn(async () => undefined) };
  const stop = startAutosave({ api, basket, session, prompts: () => prompts });
  return { basket, session, api, stop };
}

const putBody = (api: ReturnType<typeof setup>['api'], call = 0) => (api.putDraft.mock.calls[call] as unknown[])[1] as Record<string, unknown>;

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  vi.useRealTimers();
  draftStatusStore.setState({ hasDraft: false, tooLarge: false, pendingSha: null });
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' });
});

test('autosave debounces to one PUT after 500 ms', async () => {
  const { basket, api, stop } = setup();
  basket.getState().add([head(0, 1)]);
  await vi.advanceTimersByTimeAsync(100);
  basket.getState().add([head(0, 2)]);
  await vi.advanceTimersByTimeAsync(100);
  basket.getState().add([head(0, 3)]);
  await vi.advanceTimersByTimeAsync(499);
  expect(api.putDraft).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(2);
  expect(api.putDraft).toHaveBeenCalledTimes(1);
  expect(api.putDraft.mock.calls[0]?.[0]).toBe(SHA);
  expect(putBody(api)).toMatchObject({ unsaved_count: 3, prompts: ['merhaba'], model_name: model.name });
  expect(draftStatusStore.getState().hasDraft).toBe(true);
  stop();
});

test('autosave flushes on visibilitychange hidden', async () => {
  const { basket, api, stop } = setup();
  basket.getState().add([head(0, 1)]);
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' });
  document.dispatchEvent(new Event('visibilitychange'));
  await vi.advanceTimersByTimeAsync(0);
  expect(api.putDraft).toHaveBeenCalledTimes(1);
  await vi.advanceTimersByTimeAsync(1000);
  expect(api.putDraft).toHaveBeenCalledTimes(1); // the debounce timer was consumed by the flush
  stop();
});

test('hidden with nothing pending writes nothing (a draft awaiting restore is not touched)', async () => {
  const { api, stop } = setup();
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' });
  document.dispatchEvent(new Event('visibilitychange'));
  await vi.advanceTimersByTimeAsync(1000);
  expect(api.putDraft).not.toHaveBeenCalled();
  expect(api.deleteDraft).not.toHaveBeenCalled();
  stop();
});

test('autosave deletes the draft when basket and shelf become empty', async () => {
  const { basket, api, stop } = setup();
  const cut = head(0, 1);
  basket.getState().add([cut]);
  await vi.advanceTimersByTimeAsync(600);
  expect(api.putDraft).toHaveBeenCalledTimes(1);
  basket.getState().remove(cut.id);
  await vi.advanceTimersByTimeAsync(600);
  expect(api.deleteDraft).toHaveBeenCalledWith(SHA);
  expect(api.putDraft).toHaveBeenCalledTimes(1);
  expect(draftStatusStore.getState().hasDraft).toBe(false);
  stop();
});

test('autosave deletes the draft after markSaved (unsaved_count 0, shelf empty)', async () => {
  const { basket, api, stop } = setup();
  basket.getState().add([head(0, 1), head(0, 2)]);
  await vi.advanceTimersByTimeAsync(600);
  basket.getState().markSaved();
  await vi.advanceTimersByTimeAsync(600);
  expect(api.deleteDraft).toHaveBeenCalledTimes(1);
  expect(api.putDraft).toHaveBeenCalledTimes(1);
  stop();
});

test('a non-empty shelf keeps the draft even when nothing is unsaved', async () => {
  const { basket, api, stop } = setup();
  const cut = head(0, 1);
  basket.getState().add([cut, head(0, 2)]);
  await vi.advanceTimersByTimeAsync(600);
  basket.getState().moveToShelf([cut.id]);
  basket.getState().markSaved();
  await vi.advanceTimersByTimeAsync(600);
  expect(api.deleteDraft).not.toHaveBeenCalled();
  expect(api.putDraft).toHaveBeenCalledTimes(2);
  expect(putBody(api, 1)).toMatchObject({ unsaved_count: 0 });
  expect(draftStatusStore.getState().hasDraft).toBe(false);
  stop();
});

test('autosave does nothing while no model is loaded', async () => {
  const { basket, session, api, stop } = setup();
  session.getState().setModel(null);
  basket.getState().add([head(0, 1)]);
  await vi.advanceTimersByTimeAsync(1000);
  expect(api.putDraft).not.toHaveBeenCalled();
  stop();
});

test('a 413 raises the too-large notice without throwing, and the next good write clears it', async () => {
  const { basket, api, stop } = setup();
  api.putDraft.mockRejectedValueOnce(new ApiError(413, 'too_large', 'x'));
  basket.getState().add([head(0, 1)]);
  await vi.advanceTimersByTimeAsync(600);
  expect(draftStatusStore.getState().tooLarge).toBe(true);
  expect(draftStatusStore.getState().hasDraft).toBe(false);
  basket.getState().add([head(0, 2)]);
  await vi.advanceTimersByTimeAsync(600);
  expect(draftStatusStore.getState().tooLarge).toBe(false);
  expect(draftStatusStore.getState().hasDraft).toBe(true);
  stop();
});

test('a network failure is swallowed (the heartbeat reports a dead server)', async () => {
  const { basket, api, stop } = setup();
  api.putDraft.mockRejectedValueOnce(new ApiError(0, 'network', 'x'));
  basket.getState().add([head(0, 1)]);
  await vi.advanceTimersByTimeAsync(600);
  expect(draftStatusStore.getState().tooLarge).toBe(false);
  stop();
});

test('a model switch writes the pending cuts under the old model and never deletes its draft', async () => {
  const { basket, session, api, stop } = setup();
  basket.getState().add([head(0, 1)]);
  await vi.advanceTimersByTimeAsync(600);
  basket.getState().add([head(0, 2)]); // pending
  session.getState().setModel({ ...model, name: 'other', source_sha256: 'b'.repeat(64) });
  basket.getState().replaceAll({ items: [], shelf: [], savedOps: [] }); // the models screen empties the basket
  await vi.advanceTimersByTimeAsync(1000);
  expect(api.putDraft).toHaveBeenCalledTimes(2);
  expect(api.putDraft.mock.calls[1]?.[0]).toBe(SHA);
  expect(putBody(api, 1)).toMatchObject({ unsaved_count: 2 });
  expect(api.deleteDraft).not.toHaveBeenCalled();
  stop();
});

test('stop removes the listeners', async () => {
  const { basket, api, stop } = setup();
  stop();
  basket.getState().add([head(0, 1)]);
  await vi.advanceTimersByTimeAsync(1000);
  expect(api.putDraft).not.toHaveBeenCalled();
});

test('an unresolved earlier draft is never overwritten or deleted; the write goes out once it is resolved', async () => {
  draftStatusStore.getState().setPendingSha(SHA);
  const { basket, api, stop } = setup();
  const cut = head(0, 1);
  basket.getState().add([cut]);
  await vi.advanceTimersByTimeAsync(600);
  expect(api.putDraft).not.toHaveBeenCalled();
  basket.getState().remove(cut.id);
  await vi.advanceTimersByTimeAsync(600);
  expect(api.deleteDraft).not.toHaveBeenCalled();
  basket.getState().add([head(0, 2)]);
  await vi.advanceTimersByTimeAsync(600);
  expect(api.putDraft).not.toHaveBeenCalled();
  draftStatusStore.getState().setPendingSha(null); // restored or deleted
  await vi.advanceTimersByTimeAsync(600);
  expect(api.putDraft).toHaveBeenCalledTimes(1);
  expect(putBody(api)).toMatchObject({ unsaved_count: 1 });
  stop();
});

test('a pending draft of another model does not block this model', async () => {
  draftStatusStore.getState().setPendingSha('c'.repeat(64));
  const { basket, api, stop } = setup();
  basket.getState().add([head(0, 1)]);
  await vi.advanceTimersByTimeAsync(600);
  expect(api.putDraft).toHaveBeenCalledTimes(1);
  stop();
});
