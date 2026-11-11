import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { HuntResult } from '@/api/schemas';
import { basketStore } from '@/core/stores/basket';
import { selectionStore } from '@/core/stores/selection';
import { uiStore } from '@/core/stores/ui';
import { directionOp, vectorB64 } from '@/core/__tests__/fixtures/items';
import { itemLabel } from '@/features/basket/labels';
import { setLang } from '@/i18n';
import { ExampleEditor, parseLines } from '../ExampleEditor';
import { FoundDirection } from '../FoundDirection';
import { FoundHeads } from '../FoundHeads';
import { LayerChart, ValidationLine } from '../LayerChart';
import { MethodPicker } from '../MethodPicker';
import { slugify } from '../slug';

beforeEach(() => {
  setLang('tr');
  window.location.hash = '#/';
  uiStore.setState({ route: 'concept' });
  selectionStore.setState({ selected: new Set(), focus: null });
});
afterEach(() => {
  cleanup();
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
});

type Head = HuntResult['heads'][number];
const heads = (n: number): Head[] =>
  Array.from({ length: n }, (_, i) => ({ layer: i, head: 0, score: i / n, measured: null }));
const rowsOf = () => screen.getAllByRole('row').slice(1);

test('slugify makes a valid id from Turkish text', () => {
  expect(slugify('Şiir yazma ığüöç')).toBe('Siir-yazma-iguoc');
  expect(slugify('İĞÜŞÖÇ  çok   boşluk')).toBe('IGUSOC-cok-bosluk');
  expect(slugify('a/b<c>.d_e')).toBe('abc.d_e');
  expect(slugify('x'.repeat(90))).toHaveLength(64);
  expect(slugify('???')).toBe('kavram');
  expect(slugify('')).toBe('kavram');
  for (const text of ['Şiir yazma', '🙂🙂', 'a b']) expect(slugify(text)).toMatch(/^[A-Za-z0-9._-]{1,64}$/);
});

test('parseLines ignores empty lines and counts update live', async () => {
  expect(parseLines('a\n\n  \n b \r\nc')).toEqual(['a', 'b', 'c']);
  const onPositive = vi.fn();
  render(<ExampleEditor positive={'a\nb\n\nc'} negative="" readOnly={false} onPositive={onPositive} onNegative={vi.fn()} />);
  expect(screen.getByText('3 / en az 32')).toBeInTheDocument();
  expect(screen.getByText('0 / en az 32')).toBeInTheDocument();
  expect(screen.getByText('Örnekler istektir, cevap değil · %20 doğrulamaya ayrılır')).toBeInTheDocument();
  await userEvent.type(screen.getAllByRole('textbox')[0] as HTMLElement, 'x');
  expect(onPositive).toHaveBeenCalled();
});

test('method default follows the prop and shows no durations', () => {
  render(<MethodPicker value="direction" onChange={vi.fn()} />);
  expect(screen.getByRole('radio', { name: /Davranış yönü/ })).toBeChecked();
  expect(screen.getByRole('radio', { name: /Tahmini etki/ })).not.toBeChecked();
  expect(screen.getByText("GPU'da önerilen; CPU'da çok yavaş")).toBeInTheDocument();
  expect(screen.getByRole('radio', { name: /Etkinlik farkı/ })).toBeInTheDocument();
  expect(document.body.textContent).not.toMatch(/\d+\s*(dk|sn)/);
});

test('layer chart labels n_layers + 1 bars', () => {
  const { container } = render(<LayerChart scores={[0, 0.5, 1]} refLayer={2} />);
  const labels = [...container.querySelectorAll('.layer-chart-label')].map((el) => el.textContent);
  expect(labels).toEqual(['gömme', 'K0 sonrası', 'K1 sonrası']);
  const rows = screen.getAllByRole('listitem');
  expect(rows[2]).toHaveTextContent('referans');
  expect(rows[0]).not.toHaveTextContent('referans');
});

test('layer chart stays in bounds for a single layer and out-of-range scores', () => {
  const { container } = render(<LayerChart scores={[Number.NaN, 7]} refLayer={1} />);
  const widths = [...container.querySelectorAll<HTMLElement>('.layer-chart-bar')].map((el) => el.style.width);
  expect(widths).toEqual(['0%', '100%']);
  expect(container.querySelectorAll('.layer-chart-label')).toHaveLength(2);
});

test('validation line shows accuracy, interval and n', () => {
  render(<ValidationLine result={{ val_accuracy: 0.5625, val_ci: [0.3318, 0.769], val_n: 16 }} />);
  expect(screen.getByText('doğrulamada doğruluk %56 (GA %33–%77, n=16)')).toBeInTheDocument();
});

test('found heads shows top 8 by score then 8 more', async () => {
  render(<FoundHeads heads={heads(20)} />);
  expect(rowsOf()).toHaveLength(8);
  expect(rowsOf()[0]).toHaveTextContent('19'); // the highest score first
  await userEvent.click(screen.getByRole('button', { name: 'Daha fazla göster' }));
  expect(rowsOf()).toHaveLength(16);
  await userEvent.click(screen.getByRole('button', { name: 'Daha fazla göster' }));
  expect(rowsOf()).toHaveLength(20);
  expect(screen.queryByRole('button', { name: 'Daha fazla göster' })).toBeNull();
});

test('unmeasured head shows ölçülmedi; a measured one shows its numbers', () => {
  const measured = { signal_drop: 0.42, neutral_kl: 0.0021 };
  render(
    <FoundHeads
      heads={[
        { layer: 3, head: 1, score: 0.9, measured: null },
        { layer: 2, head: 0, score: 0.5, measured },
        { layer: 4, head: 2, score: 0.1, measured: { signal_drop: null, neutral_kl: 0.5 } },
      ]}
    />,
  );
  expect(screen.getAllByText('ölçülmedi')).toHaveLength(1);
  expect(screen.getByText('sinyal düşüşü 0,42 · nötr KL 0,0021')).toBeInTheDocument();
  expect(screen.getByText('sinyali etkileyemez · nötr KL 0,5')).toBeInTheDocument();
});

test('Haritada göster ve ölç selects the shown heads and opens the map', async () => {
  render(<FoundHeads heads={heads(12)} />);
  await userEvent.click(screen.getByRole('button', { name: 'Haritada göster ve ölç' }));
  expect([...selectionStore.getState().selected].sort()).toEqual(
    [11, 10, 9, 8, 7, 6, 5, 4].map((layer) => `h:${layer}:0`).sort(),
  );
  expect(uiStore.getState().route).toBe('explore');
});

const result = (patch: Partial<HuntResult> = {}): HuntResult => ({
  method: 'direction',
  layer_scores: [0, 1, 0.5],
  heads: [],
  direction: vectorB64([1, 2, 3, 4]),
  direction_shape: [4],
  direction_layer: 2,
  ref_layer: 2,
  val_accuracy: 1,
  val_ci: [0.8, 1],
  val_n: 16,
  ...patch,
});

test('direction maps to DirectionOp with d_model from direction_shape', async () => {
  render(<FoundDirection result={result()} conceptId="code" />);
  await userEvent.click(screen.getByRole('button', { name: 'Sepete ekle (geçici)' }));
  const [added, ...rest] = basketStore.getState().items;
  expect(rest).toEqual([]);
  expect(added).toMatchObject({
    op: { type: 'direction', d_model: 4, vector_b64: directionOp([1, 2, 3, 4]).vector_b64 },
    semantics: 'direction',
    origin: 'concept',
    concept: 'code',
  });
  expect(added && itemLabel(added)).toBe('Yön · K1 sonrası'); // display only: the op itself keeps no layer
  expect(Object.keys(added?.op ?? {}).sort()).toEqual(['d_model', 'type', 'vector_b64']);
  expect(screen.getByRole('status')).toHaveTextContent('Sepete eklendi.');
});

test('adding the same direction twice says it is already there', async () => {
  render(<FoundDirection result={result()} conceptId="code" />);
  const add = screen.getByRole('button', { name: 'Sepete ekle (geçici)' });
  await userEvent.click(add);
  await userEvent.click(add);
  expect(basketStore.getState().items).toHaveLength(1);
  expect(screen.getByRole('status')).toHaveTextContent('Yön sepette zaten var');
});

test('a hunt without a direction shows nothing, a malformed vector shows an error', () => {
  const { container, rerender } = render(<FoundDirection result={result({ direction: null })} conceptId="code" />);
  expect(container).toBeEmptyDOMElement();
  rerender(<FoundDirection result={result({ direction: 'AAAA' })} conceptId="code" />);
  expect(screen.getByRole('alert')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Sepete ekle (geçici)' })).toBeNull();
});
