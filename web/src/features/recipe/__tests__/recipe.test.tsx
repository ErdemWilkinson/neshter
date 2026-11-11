import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { Api } from '@/api/client';
import { setApi } from '@/api/currentApi';
import type { LoadedSession } from '@/api/schemas';
import { sessionLoaded } from '@/app/__tests__/fakeServer';
import { basketStore } from '@/core/stores/basket';
import { sessionStore } from '@/core/stores/session';
import { uiStore } from '@/core/stores/ui';
import { directionOp, head, headOp, item, NOW, SHA } from '@/core/__tests__/fixtures/items';
import { setLang } from '@/i18n';
import { ImportRecipe } from '../ImportRecipe';
import { pendingImportStore } from '../pendingImport';
import { isUnverifiedDirection } from '../unverified';

const recipeText = (interventions: unknown[]) =>
  JSON.stringify({
    schema_version: 1,
    neshter_version: 'test',
    source_model: { name: 'neshter/tiny-e2e', sha256: SHA },
    concept: null,
    method: 'ui',
    seed: 0,
    dataset_sha256: null,
    head_mode: 'zero',
    interventions,
    created_at: NOW,
  });

const file = (text: string) => new File([text], 'tarif.json', { type: 'application/json' });

function mockApi(check: { ok: boolean; errors: string[]; sha_mismatch: boolean }) {
  const api = { recipeCheck: vi.fn(async () => check) };
  setApi(api as unknown as Api);
  return api;
}

const upload = (text: string) => userEvent.upload(screen.getByLabelText('Tarif içe aktar…'), file(text));
const loadModel = () => sessionStore.getState().setModel(sessionLoaded as unknown as LoadedSession);

beforeEach(() => {
  setLang('tr');
  window.location.hash = '#/';
});
afterEach(() => {
  setApi(null);
  pendingImportStore.getState().set(null);
  sessionStore.setState({ model: null });
  uiStore.setState({ route: 'home' });
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
});

test('isUnverifiedDirection is true for an imported direction only', () => {
  expect(isUnverifiedDirection(item(directionOp([1, 2]), { origin: 'import' }))).toBe(true);
  expect(isUnverifiedDirection(item(directionOp([1, 2]), { origin: 'concept' }))).toBe(false);
  expect(isUnverifiedDirection(item(headOp(0, 1), { origin: 'import' }))).toBe(false);
  expect(isUnverifiedDirection(head(0, 1))).toBe(false);
});

test('imported direction ops carry the doğrulanmamış yön badge; head ops do not', async () => {
  render(<ImportRecipe />);
  await upload(recipeText([headOp(0, 1), headOp(1, 2), directionOp([1, 2, 3])]));
  const list = await screen.findByRole('list', { name: 'Kesimler' });
  const rows = within(list).getAllByRole('listitem');
  expect(rows).toHaveLength(3);
  expect(rows.map((row) => within(row).queryByText('doğrulanmamış yön') !== null)).toEqual([false, false, true]);
  expect(screen.getByText('2 head kesimi · 1 yön kesimi')).toBeInTheDocument();
  expect(screen.getByText('Kaynak model: neshter/tiny-e2e')).toBeInTheDocument();
});

test('the badge stays on the item once it is in the basket', async () => {
  const api = mockApi({ ok: true, errors: [], sha_mismatch: false });
  loadModel();
  render(<ImportRecipe />);
  await upload(recipeText([headOp(0, 1), directionOp([1, 2, 3])]));
  await userEvent.click(await screen.findByRole('button', { name: 'Sepete ekle' }));
  await waitFor(() => expect(basketStore.getState().items).toHaveLength(2));
  expect(api.recipeCheck).toHaveBeenCalledTimes(1);
  const items = basketStore.getState().items;
  expect(items.every((i) => i.origin === 'import')).toBe(true);
  expect(items.map(isUnverifiedDirection)).toEqual([false, true]);
  expect(pendingImportStore.getState().recipe).toBeNull();
});

test('recipe import with sha mismatch asks before adding', async () => {
  mockApi({ ok: true, errors: [], sha_mismatch: true });
  loadModel();
  render(<ImportRecipe />);
  await upload(recipeText([headOp(0, 1)]));
  await userEvent.click(await screen.findByRole('button', { name: 'Sepete ekle' }));
  const dialog = await screen.findByRole('dialog');
  expect(within(dialog).getByText('Bu tarif başka bir model sürümü için hazırlanmış olabilir. Yine de ekle?')).toBeInTheDocument();
  expect(basketStore.getState().items).toHaveLength(0);
  await userEvent.click(within(dialog).getByRole('button', { name: 'Yine de ekle' }));
  expect(basketStore.getState().items).toHaveLength(1);
  expect(basketStore.getState().items[0]?.origin).toBe('import');
});

test('declining the sha mismatch adds nothing and drops the recipe', async () => {
  mockApi({ ok: true, errors: [], sha_mismatch: true });
  loadModel();
  render(<ImportRecipe />);
  await upload(recipeText([headOp(0, 1)]));
  await userEvent.click(await screen.findByRole('button', { name: 'Sepete ekle' }));
  await userEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Ekleme' }));
  expect(basketStore.getState().items).toHaveLength(0);
  expect(pendingImportStore.getState().recipe).toBeNull();
});

test('a recipe the server rejects shows an error card and adds nothing', async () => {
  mockApi({ ok: false, errors: ['katman 9 yok'], sha_mismatch: false });
  loadModel();
  render(<ImportRecipe />);
  await upload(recipeText([headOp(9, 1)]));
  await userEvent.click(await screen.findByRole('button', { name: 'Sepete ekle' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Girdi geçersiz');
  expect(screen.getByText(/katman 9 yok/)).toBeInTheDocument();
  expect(basketStore.getState().items).toHaveLength(0);
});

test('without a loaded model the recipe waits in memory and offers the model picker', async () => {
  render(<ImportRecipe />);
  await upload(recipeText([headOp(0, 1)]));
  expect(await screen.findByText('Tarifi eklemek için önce bir model yükleyin.')).toBeInTheDocument();
  expect(pendingImportStore.getState().recipe).not.toBeNull();
  await userEvent.click(screen.getByRole('button', { name: 'Model seç' }));
  expect(uiStore.getState().route).toBe('models');
  expect(pendingImportStore.getState().recipe).not.toBeNull();
});

test.each([
  ['{not json', 'Dosya geçerli bir JSON değil.'],
  ['{"schema_version":2}', 'Dosya bir Neşter tarifi gibi görünmüyor.'],
])('a bad recipe file is explained (%s)', async (text, message) => {
  render(<ImportRecipe />);
  await upload(text);
  expect(await screen.findByRole('alert')).toHaveTextContent(message);
  expect(pendingImportStore.getState().recipe).toBeNull();
});
