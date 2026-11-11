import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { opKey } from '@/core/ops';
import { basketStore } from '@/core/stores/basket';
import { uiStore } from '@/core/stores/ui';
import { saveFlowStore } from '@/features/saveflow/saveFlowStore';
import { FAKE_EXPLORE_TYPE, registerFakeExploreOp } from '@/core/__tests__/fixtures/fakeExploreOp';
import { directionOp, head, item } from '@/core/__tests__/fixtures/items';
import { setLang } from '@/i18n';
import { BasketPanel } from '../BasketPanel';
import { itemLabel } from '../labels';
import { ShelfView } from '../ShelfView';

const explore = () => item({ type: FAKE_EXPLORE_TYPE, src: 'a' });
const direction = (origin: 'import' | 'concept') => item(directionOp([1, 2, 3, 4]), { origin });

beforeAll(registerFakeExploreOp);
beforeEach(() => {
  setLang('tr');
  uiStore.setState({ route: 'explore' });
});
afterEach(() => {
  cleanup();
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
});

const rowOf = (text: string) => {
  const row = screen.getByText(text).closest('li');
  if (row === null) throw new Error(`no row for ${text}`);
  return row;
};

test('rows carry a savability badge: kaydedilebilir, yalnızca keşif, desteklenmeyen', () => {
  basketStore.getState().add([head(3, 5), explore(), item({ type: 'future_op', x: 1 })]);
  render(<BasketPanel />);
  expect(within(rowOf('K3 · head 5')).getByText('kaydedilebilir')).toBeTruthy();
  expect(within(rowOf(`Kesim: ${FAKE_EXPLORE_TYPE}`)).getByText('yalnızca keşif')).toBeTruthy();
  expect(within(rowOf('desteklenmeyen kesim (future_op)')).getByText('desteklenmeyen')).toBeTruthy();
});

test('opaque op type renders as text', () => {
  const evil = '<img src=x onerror=alert(1)>';
  const opaque = item({ type: evil });
  expect(itemLabel(opaque)).toBe(`desteklenmeyen kesim (${evil})`);
  basketStore.getState().add([opaque]);
  const { container } = render(<BasketPanel />);
  expect(container.querySelector('img')).toBeNull();
});

test('remove then Geri al restores the row', async () => {
  const user = userEvent.setup();
  basketStore.getState().add([head(3, 5), head(4, 1)]);
  render(<BasketPanel />);
  await user.click(screen.getByRole('button', { name: 'K3 · head 5 sepetten çıkar' }));
  expect(screen.queryByText('K3 · head 5')).toBeNull();
  await user.click(screen.getByRole('button', { name: 'Geri al' }));
  expect(screen.getByText('K3 · head 5')).toBeTruthy();
  expect(screen.queryByRole('button', { name: 'Geri al' })).toBeNull();
});

test('basket line counts savable and explore-only with the fake explore op', () => {
  basketStore.getState().add([head(3, 5), head(4, 1), explore()]);
  render(<BasketPanel />);
  expect(screen.getByText('Kaydedilebilir kesim: 2 · Yalnızca keşif: 1')).toBeTruthy();
});

test('pending-discard rows are struck through and marked atılacak', () => {
  const pending = explore();
  basketStore.getState().add([head(3, 5), pending]);
  basketStore.getState().markPendingDiscard([pending.id]);
  render(<BasketPanel />);
  const row = rowOf(`Kesim: ${FAKE_EXPLORE_TYPE}`);
  expect(row.className).toContain('basket-row-pending');
  expect(within(row).getByText('atılacak')).toBeTruthy();
  expect(screen.getByText('Kaydedilebilir kesim: 1 · Yalnızca keşif: 0')).toBeTruthy();
});

test('Modeli kaydet label counts savable items and navigates to save without dispatching', async () => {
  const user = userEvent.setup();
  basketStore.getState().add([head(3, 5), head(4, 1), explore()]);
  render(<BasketPanel />);
  await user.click(screen.getByRole('button', { name: 'Modeli kaydet (2 kesim)' }));
  expect(uiStore.getState().route).toBe('save');
  expect(saveFlowStore.getState().state.kind).toBe('idle'); // F19: SAVE_CLICKED belongs to the Save screen
});

test('Modeli kaydet is disabled with nothing savable; Karşılaştır goes to compare', async () => {
  const user = userEvent.setup();
  basketStore.getState().add([explore()]);
  render(<BasketPanel />);
  expect((screen.getByRole('button', { name: 'Modeli kaydet (0 kesim)' }) as HTMLButtonElement).disabled).toBe(true);
  await user.click(screen.getByRole('button', { name: 'Karşılaştır' }));
  expect(uiStore.getState().route).toBe('compare');
});

test('shelf restore moves an item back to the basket', async () => {
  const user = userEvent.setup();
  const shelved = explore();
  basketStore.getState().replaceAll({ items: [head(3, 5)], shelf: [shelved], savedOps: [] });
  render(
    <>
      <BasketPanel />
      <ShelfView />
    </>,
  );
  expect(screen.getByRole('heading', { name: 'Keşif rafı' })).toBeTruthy();
  await user.click(screen.getByRole('button', { name: `Kesim: ${FAKE_EXPLORE_TYPE} sepete geri al` }));
  expect(basketStore.getState().items).toHaveLength(2);
  expect(basketStore.getState().shelf).toHaveLength(0);
  expect(screen.queryByRole('heading', { name: 'Keşif rafı' })).toBeNull();
});

test('imported direction row shows doğrulanmamış yön; concept direction and head rows do not', () => {
  basketStore.getState().add([direction('import'), head(3, 5)]);
  const { rerender } = render(<BasketPanel />);
  expect(screen.getAllByText('doğrulanmamış yön')).toHaveLength(1);
  expect(within(rowOf('Yön · 4 boyut')).getByText('doğrulanmamış yön')).toBeTruthy();
  expect(within(rowOf('K3 · head 5')).queryByText('doğrulanmamış yön')).toBeNull();

  basketStore.getState().replaceAll({ items: [direction('concept')], shelf: [], savedOps: [] });
  rerender(<BasketPanel />);
  expect(screen.queryByText('doğrulanmamış yön')).toBeNull();
});

test('after a save the savable cuts read as saved, not as still to save', () => {
  const saved = head(3, 5);
  basketStore.getState().add([saved, head(4, 1)]);
  basketStore.setState({ savedOps: [opKey(saved.op)] });
  render(<BasketPanel />);
  expect(screen.getByText(/Kaydedilebilir kesim: 2 · Yalnızca keşif: 0 · 1 kesim kaydedildi/)).toBeTruthy();
  expect(within(rowOf('K3 · head 5')).getByText('kaydedildi')).toBeTruthy();
  expect(within(rowOf('K4 · head 1')).getByText('kaydedilebilir')).toBeTruthy();
});
