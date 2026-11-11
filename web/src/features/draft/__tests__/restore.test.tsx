import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { Api } from '@/api/client';
import { ApiError } from '@/api/errors';
import type { DraftBody, LoadedSession } from '@/api/schemas';
import { setApi } from '@/api/currentApi';
import { sessionLoaded } from '@/app/__tests__/fakeServer';
import { encodeDraft } from '@/core/draft';
import { basketStore } from '@/core/stores/basket';
import { compareStore } from '@/core/stores/compare';
import { sessionStore } from '@/core/stores/session';
import { head, item } from '@/core/__tests__/fixtures/items';
import { setLang } from '@/i18n';
import { draftStatusStore } from '../draftStatus';
import { RestorePrompt } from '../RestorePrompt';
import { UnsavedWork } from '../UnsavedWork';

const model = sessionLoaded as unknown as LoadedSession;
const SHA = model.source_sha256;

function draftBody(items = [head(0, 1), head(0, 2)], extra: Partial<DraftBody> = {}): DraftBody {
  return { ...encodeDraft({ items, shelf: [], savedOps: [], model: { name: model.name, sha256: SHA }, prompts: ['merhaba'] }), updated_at: 'x', ...extra };
}

function mockApi(patch: Partial<Record<keyof Api, unknown>>) {
  const api = {
    draft: vi.fn(async () => draftBody()),
    deleteDraft: vi.fn(async () => undefined),
    recipeCheck: vi.fn(async () => ({ ok: true, errors: [], sha_mismatch: false })),
    drafts: vi.fn(async () => []),
    ...patch,
  };
  setApi(api as unknown as Api);
  return api;
}

beforeEach(() => {
  setLang('tr');
  sessionStore.getState().setModel(model);
});
afterEach(() => {
  setApi(null);
  sessionStore.setState({ model: null });
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
  compareStore.setState({ lastPrompt: null });
  draftStatusStore.setState({ hasDraft: false, tooLarge: false });
});

test('restore prompt shows count and restores after recipe check', async () => {
  const api = mockApi({});
  render(<RestorePrompt />);
  expect(await screen.findByText('Önceki oturumdan kaydedilmemiş 2 kesim var: geri yükle?')).toBeInTheDocument();
  expect(api.draft).toHaveBeenCalledWith(SHA);
  expect(draftStatusStore.getState().hasDraft).toBe(true);
  await userEvent.click(screen.getByRole('button', { name: 'Geri yükle' }));
  await waitFor(() => expect(basketStore.getState().items).toHaveLength(2));
  expect(api.recipeCheck).toHaveBeenCalledTimes(1);
  expect(compareStore.getState().lastPrompt).toBe('merhaba');
  expect(screen.queryByText(/geri yükle\?/)).toBeNull();
});

test('restore keeps an unknown op as an unsupported row', async () => {
  const body = draftBody([head(0, 1), item({ type: 'edge_read_v9', edge: 3 })]);
  mockApi({ draft: vi.fn(async () => body) });
  render(<RestorePrompt />);
  expect(await screen.findByText('1 kesim bu sürümde desteklenmiyor; sepette gri görünür')).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Geri yükle' }));
  await waitFor(() => expect(basketStore.getState().items).toHaveLength(2));
  expect(basketStore.getState().items.some((i) => 'opaque' in i.op)).toBe(true);
});

test('a failed recipe check lists the errors and restores nothing', async () => {
  mockApi({ recipeCheck: vi.fn(async () => ({ ok: false, errors: ['head 9 yok <b>x</b>'], sha_mismatch: false })) });
  render(<RestorePrompt />);
  await userEvent.click(await screen.findByRole('button', { name: 'Geri yükle' }));
  expect(await screen.findByText('Kesimler geri yüklenemedi')).toBeInTheDocument();
  expect(screen.getByText(/head 9 yok <b>x<\/b>/).closest('details')).not.toBeNull();
  expect(basketStore.getState().items).toHaveLength(0);
  expect(screen.getByRole('button', { name: 'Geri yükle' })).toBeInTheDocument(); // can be retried or discarded
});

test('Sil deletes the draft and Sonra just closes the prompt', async () => {
  const api = mockApi({});
  const { unmount } = render(<RestorePrompt />);
  await userEvent.click(await screen.findByRole('button', { name: 'Sonra' }));
  expect(screen.queryByRole('region', { name: 'Önceki oturum' })).toBeNull();
  expect(api.deleteDraft).not.toHaveBeenCalled();
  unmount();
  render(<RestorePrompt />);
  await userEvent.click(await screen.findByRole('button', { name: 'Sil' }));
  expect(api.deleteDraft).toHaveBeenCalledWith(SHA);
  await waitFor(() => expect(screen.queryByRole('region', { name: 'Önceki oturum' })).toBeNull());
  expect(draftStatusStore.getState().hasDraft).toBe(false);
});

test('restore is not offered when the basket is not empty', async () => {
  basketStore.getState().add([head(3, 3)]);
  const api = mockApi({});
  render(<RestorePrompt />);
  await waitFor(() => expect(api.draft).toHaveBeenCalled());
  await act(async () => undefined);
  expect(screen.queryByRole('region', { name: 'Önceki oturum' })).toBeNull();
});

test('restore is not offered for an empty, undecodable or missing draft', async () => {
  for (const body of [draftBody([]), { ...draftBody(), basket: [{ id: 5 }] } as DraftBody, { ...draftBody(), schema_version: 2 } as unknown as DraftBody]) {
    const api = mockApi({ draft: vi.fn(async () => body) });
    const { unmount } = render(<RestorePrompt />);
    await waitFor(() => expect(api.draft).toHaveBeenCalled());
    await act(async () => undefined);
    expect(screen.queryByRole('region', { name: 'Önceki oturum' })).toBeNull();
    unmount();
  }
  const api = mockApi({ draft: vi.fn(async () => Promise.reject(new ApiError(404, 'not_found'))) });
  render(<RestorePrompt />);
  await waitFor(() => expect(api.draft).toHaveBeenCalled());
  expect(screen.queryByRole('region', { name: 'Önceki oturum' })).toBeNull();
});

test('restore and UnsavedWork ignore drafts with unsaved_count 0', async () => {
  const zero = draftBody([head(0, 1)], { unsaved_count: 0 });
  mockApi({
    draft: vi.fn(async () => zero),
    drafts: vi.fn(async () => [{ sha256: SHA, model_name: 'abc123', updated_at: 'x', unsaved_count: 0 }]),
  });
  const { container } = render(
    <>
      <RestorePrompt />
      <UnsavedWork />
    </>,
  );
  await act(async () => undefined);
  await act(async () => undefined);
  expect(container).toBeEmptyDOMElement();
  expect(document.body.textContent).not.toMatch(/\b0 kesim/);
});

test('UnsavedWork lists rows as text, notes that prompts are stored, and deletes after confirmation', async () => {
  const api = mockApi({
    drafts: vi.fn(async () => [
      { sha256: SHA, model_name: '<img src=x onerror=alert(1)>', updated_at: 'x', unsaved_count: 3 },
      { sha256: 'c'.repeat(64), model_name: 'eski', updated_at: 'x', unsaved_count: 0 },
    ]),
  });
  render(<UnsavedWork />);
  expect(await screen.findByText('Kaydedilmemiş çalışmalar: <img src=x onerror=alert(1)> · 3 kesim')).toBeInTheDocument();
  expect(document.querySelector('img')).toBeNull();
  expect(screen.queryByText(/eski/)).toBeNull();
  expect(screen.getByText(/istemleri de saklar/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Taslakları sil' }));
  expect(api.deleteDraft).not.toHaveBeenCalled();
  const dialog = screen.getByRole('dialog', { name: 'Tüm taslaklar silinsin mi?' });
  await userEvent.click(within(dialog).getByRole('button', { name: 'Sil' }));
  await waitFor(() => expect(api.deleteDraft).toHaveBeenCalledWith(SHA));
  expect(api.deleteDraft).toHaveBeenCalledTimes(1);
  await waitFor(() => expect(screen.queryByText(/Kaydedilmemiş çalışmalar/)).toBeNull());
});
