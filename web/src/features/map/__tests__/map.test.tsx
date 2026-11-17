import { act, cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { headId, type NodeId } from '@/core/ids';
import { selectionStore } from '@/core/stores/selection';
import { setLang } from '@/i18n';
import { LiveRegion } from '@/ui/LiveRegion';
import { makeCellLayout, hitTest } from '../cellLayout';
import { heatColor, makeColorScale } from '../colorScale';
import { drawHeatmap, TILE_GAP } from '../drawHeatmap';
import { Heatmap } from '../Heatmap';
import { HeatmapTable } from '../HeatmapTable';
import { TokenStrip } from '../TokenStrip';
import { nextCell } from '../useHeatmapKeys';

const rect = (left: number, top: number, width: number, height: number) => ({ left, top, width, height });
const zeros = (l: number, h: number) => Array.from({ length: l }, () => Array<number>(h).fill(0));
const gray = (v: number): [number, number, number] => [v * 255, v * 255, v * 255];
const scale = makeColorScale([0, 0, 0], [255, 255, 255]);

/** A recording stand-in for a 2d context: jsdom has no canvas. Tiles are `fillRect`, outlines `strokeRect`. */
function fakeCtx() {
  const tiles: { args: number[]; style: string }[] = [];
  const strokes: { style: string; args: number[] }[] = [];
  const ctx = {
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    fillRect: (...args: number[]) => void tiles.push({ args, style: ctx.fillStyle }),
    strokeRect: (...args: number[]) => void strokes.push({ style: ctx.strokeStyle, args }),
    setTransform: () => undefined,
  };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, tiles, strokes };
}

beforeEach(() => {
  setLang('tr');
  selectionStore.setState({ selected: new Set<NodeId>(), focus: null });
});
afterEach(() => vi.restoreAllMocks());

test('cellLayout returns rects for 3x5 and null outside', () => {
  const L = makeCellLayout({ n_layers: 3, n_heads: 5 }, rect(0, 0, 500, 300), zeros(3, 5), gray);
  expect(L.cellRect(headId(2, 4))).toEqual({ x: 400, y: 200, w: 100, h: 100 });
  expect(L.cellRect(headId(0, 0))).toEqual({ x: 0, y: 0, w: 100, h: 100 });
  expect(L.cellRect(headId(3, 0))).toBeNull();
  expect(L.cellRect(headId(0, 5))).toBeNull();
  expect(L.cellRect('n:0:1' as NodeId)).toBeNull();
  expect(L.cellRect('h:01:1' as NodeId)).toBeNull();
});

test('cellLayout is in viewport px (offset box) and colours come from the values', () => {
  const values = [[0, 1]];
  const L = makeCellLayout({ n_layers: 1, n_heads: 2 }, rect(10, 20, 200, 50), values, gray);
  expect(L.cellRect(headId(0, 1))).toEqual({ x: 110, y: 20, w: 100, h: 50 });
  expect(L.cellColor(headId(0, 1))).toEqual([255, 255, 255]);
  expect(L.cellColor(headId(0, 0))).toEqual([0, 0, 0]);
});

test('hit-testing maps points to head ids and rejects points outside the grid', () => {
  const L = makeCellLayout({ n_layers: 3, n_heads: 5 }, rect(10, 20, 500, 300), zeros(3, 5), gray);
  expect(hitTest(L, 10, 20)).toBe(headId(0, 0));
  expect(hitTest(L, 509, 319)).toBe(headId(2, 4));
  expect(hitTest(L, 215, 130)).toBe(headId(1, 2));
  expect(hitTest(L, 9, 20)).toBeNull();
  expect(hitTest(L, 510, 100)).toBeNull();
  expect(hitTest(L, 100, 320)).toBeNull();
  const empty = makeCellLayout({ n_layers: 0, n_heads: 0 }, rect(0, 0, 10, 10), [], gray);
  expect(hitTest(empty, 1, 1)).toBeNull();
});

test('colour scale clamps and treats NaN as zero', () => {
  expect(scale(-1)).toEqual([0, 0, 0]);
  expect(scale(2)).toEqual([255, 255, 255]);
  expect(scale(Number.NaN)).toEqual([0, 0, 0]);
});

test('drawHeatmap draws one tile per cell and outlines selected and basket cells', () => {
  const values = zeros(3, 5);
  const layout = makeCellLayout({ n_layers: 3, n_heads: 5 }, rect(0, 0, 500, 300), values, scale);
  const { ctx, tiles, strokes } = fakeCtx();
  drawHeatmap(ctx, layout, values, {
    selected: new Set([headId(0, 1)]),
    basket: new Set([headId(2, 4)]),
    focus: null,
  });
  expect(tiles).toHaveLength(15);
  const g = TILE_GAP / 2;
  expect(tiles[0]?.args).toEqual([g, g, 100 - TILE_GAP, 100 - TILE_GAP]); // cell (0,0), a hair smaller than its cell
  expect(strokes.map((s) => s.style)).toEqual([
    '#ff3b30', // --select-ring
    '#ff7a45', // --accent-basket
  ]);
  expect(strokes[0]?.args.slice(0, 2)).toEqual([101.5, 1.5]); // inside cell (0,1)
  expect(strokes[1]?.args.slice(0, 2)).toEqual([401.5, 201.5]); // inside cell (2,4)
});

test('a cell that is selected and in the basket gets both outlines, the keyboard focus its edge', () => {
  const values = zeros(1, 2);
  const layout = makeCellLayout({ n_layers: 1, n_heads: 2 }, rect(0, 0, 100, 50), values, scale);
  const { ctx, strokes } = fakeCtx();
  drawHeatmap(ctx, layout, values, {
    selected: new Set([headId(0, 0)]),
    basket: new Set([headId(0, 0)]),
    focus: headId(0, 0),
  });
  expect(strokes).toHaveLength(3);
  expect(strokes[1]?.args[0]).toBeGreaterThan(strokes[0]?.args[0] ?? Infinity); // basket outline inside the selection
  expect(strokes[2]?.args.slice(0, 2)).toEqual([0, 0]); // the keyboard focus on the cell's edge
});

test('the heat colour runs along viridis, clamped', () => {
  expect(heatColor(0)).toEqual([68, 1, 84]);
  expect(heatColor(1)).toEqual([253, 231, 37]);
  expect(heatColor(0.5)).toEqual([33, 145, 140]);
  expect(heatColor(-3)).toEqual(heatColor(0));
  expect(heatColor(Number.NaN)).toEqual(heatColor(0));
});

test('keyboard navigation clamps at edges for 1x1 and 3x5', () => {
  const one = { n_layers: 1, n_heads: 1 };
  for (const key of ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End']) {
    expect(nextCell(key, { layer: 0, head: 0 }, one)).toEqual({ layer: 0, head: 0 });
  }
  const grid = { n_layers: 3, n_heads: 5 };
  expect(nextCell('ArrowUp', { layer: 0, head: 2 }, grid)).toEqual({ layer: 0, head: 2 });
  expect(nextCell('ArrowDown', { layer: 2, head: 2 }, grid)).toEqual({ layer: 2, head: 2 });
  expect(nextCell('ArrowLeft', { layer: 1, head: 0 }, grid)).toEqual({ layer: 1, head: 0 });
  expect(nextCell('ArrowRight', { layer: 1, head: 4 }, grid)).toEqual({ layer: 1, head: 4 });
  expect(nextCell('End', { layer: 1, head: 1 }, grid)).toEqual({ layer: 1, head: 4 });
  expect(nextCell('Home', { layer: 1, head: 3 }, grid)).toEqual({ layer: 1, head: 0 });
  expect(nextCell('a', { layer: 1, head: 3 }, grid)).toBeNull();
});

function renderHeatmap(values: number[][], basket: ReadonlySet<NodeId> = new Set()) {
  const fills: number[] = [];
  const ctx = {
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    fillRect: () => void fills.push(1),
    strokeRect: () => undefined,
    setTransform: () => void (fills.length = 0), // each draw starts with a transform: count the last draw only
  };
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
  const dims = { n_layers: values.length, n_heads: values[0]?.length ?? 0 };
  render(
    <>
      <p id="legend">legend</p>
      <Heatmap dims={dims} values={values} basket={basket} describedBy="legend" />
      <LiveRegion />
    </>,
  );
  return { fills, canvas: screen.getByRole('group') };
}

test('the canvas is a labelled focusable group described by the legend', () => {
  const { canvas } = renderHeatmap(zeros(3, 5));
  expect(canvas.tagName).toBe('CANVAS');
  expect(canvas).toHaveAttribute('tabindex', '0');
  expect(canvas).toHaveAttribute('aria-roledescription', 'ısı haritası');
  expect(canvas).toHaveAttribute('aria-describedby', 'legend');
});

test('the heatmap is framed by Katman / Head axis labels and numbered ticks', () => {
  renderHeatmap(zeros(3, 5));
  expect(screen.getByText('Katman')).toBeInTheDocument();
  expect(screen.getByText(/^Head →$/)).toBeInTheDocument();
  expect(document.querySelectorAll('.heatmap-xticks span')).toHaveLength(5);
  expect(document.querySelectorAll('.heatmap-yticks span')).toHaveLength(3);
});

test('arrow keys move focus inside the grid and announce the cell', async () => {
  const user = userEvent.setup();
  const values = [
    [0.1, 0.5, 1],
    [0, 0.25, 0.75],
  ];
  const { canvas } = renderHeatmap(values);
  await user.click(canvas); // jsdom's canvas box is 0 px, so the click hits no cell; it only focuses
  expect(selectionStore.getState().focus).toBe(headId(0, 0));
  await user.keyboard('{ArrowRight}{ArrowRight}{ArrowRight}');
  expect(selectionStore.getState().focus).toBe(headId(0, 2));
  expect(screen.getByRole('status')).toHaveTextContent('K0 · H2 · güç 1,00 · seçili değil');
  await user.keyboard('{ArrowDown}{ArrowDown}');
  expect(selectionStore.getState().focus).toBe(headId(1, 2));
  await user.keyboard('{Home}');
  expect(selectionStore.getState().focus).toBe(headId(1, 0));
  await user.keyboard('{End}');
  expect(selectionStore.getState().focus).toBe(headId(1, 2));
});

test('Space toggles h:L:H in the selection store and announces it', async () => {
  const user = userEvent.setup();
  const { canvas } = renderHeatmap([[0.1, 0.5]]);
  canvas.focus();
  await user.keyboard('{ArrowRight}');
  await user.keyboard(' ');
  expect([...selectionStore.getState().selected]).toEqual([headId(0, 1)]);
  expect(screen.getByRole('status')).toHaveTextContent('K0 · H1 · güç 0,50 · seçili');
  expect(screen.getByRole('status')).not.toHaveTextContent('seçili değil');
  await user.keyboard('{Enter}');
  expect(selectionStore.getState().selected.size).toBe(0);
  expect(screen.getByRole('status')).toHaveTextContent('K0 · H1 · güç 0,50 · seçili değil');
});

test('a 1x1 grid keeps the focus on its only cell', async () => {
  const user = userEvent.setup();
  const { canvas } = renderHeatmap([[1]]);
  canvas.focus();
  await user.keyboard('{ArrowDown}{ArrowRight}{End}{Home}{ArrowUp}{ArrowLeft}');
  expect(selectionStore.getState().focus).toBe(headId(0, 0));
});

test('a stale focus id outside the grid is pulled back inside', async () => {
  const user = userEvent.setup();
  selectionStore.setState({ focus: headId(9, 9) });
  const { canvas } = renderHeatmap(zeros(3, 5));
  canvas.focus();
  await user.keyboard('{ArrowUp}');
  expect(selectionStore.getState().focus).toBe(headId(1, 4));
});

test('Liste view selection is the same store as the heatmap', async () => {
  const user = userEvent.setup();
  const values = [
    [0.2, 1],
    [0, 0.5],
  ];
  render(<HeatmapTable values={values} basket={new Set([headId(1, 1)])} />);
  await user.click(screen.getByRole('checkbox', { name: 'K1 · H0 seç' }));
  expect(selectionStore.getState().selected.has(headId(1, 0))).toBe(true);
  act(() => selectionStore.getState().toggle(headId(0, 1))); // from the heatmap side
  expect(screen.getByRole('checkbox', { name: 'K0 · H1 seç' })).toBeChecked();
  const rows = screen.getAllByRole('row');
  expect(within(rows[4] as HTMLElement).getByText('sepette')).toBeInTheDocument();
});

test('the table sorts by power, descending then ascending then back', async () => {
  const user = userEvent.setup();
  render(<HeatmapTable values={[[0.2, 1], [0, 0.5]]} basket={new Set()} />);
  const power = () => screen.getAllByRole('row').slice(1).map((r) => within(r).getAllByRole('cell')[2]?.textContent);
  expect(power()).toEqual(['0,20', '1,00', '0,00', '0,50']);
  await user.click(screen.getByRole('button', { name: /Güç/ }));
  expect(power()).toEqual(['1,00', '0,50', '0,20', '0,00']);
  expect(screen.getByRole('columnheader', { name: /Güç/ })).toHaveAttribute('aria-sort', 'descending');
  await user.click(screen.getByRole('button', { name: /Güç/ }));
  expect(power()).toEqual(['0,00', '0,20', '0,50', '1,00']);
  await user.click(screen.getByRole('button', { name: /Güç/ }));
  expect(power()).toEqual(['0,20', '1,00', '0,00', '0,50']);
});

test('token strip renders markup as text', () => {
  render(<TokenStrip tokens={['<img src=x onerror=alert(1)>', ' dün']} index={1} disabled={false} onSelect={() => undefined} />);
  expect(document.querySelector('img')).toBeNull();
  expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeTruthy();
  expect(screen.getAllByRole('button')[1]).toHaveAttribute('aria-pressed', 'true');
});

test('a blank token still has an accessible name', () => {
  render(<TokenStrip tokens={['a', ' ', '\n']} index={0} disabled={false} onSelect={() => undefined} />);
  expect(screen.getByRole('button', { name: 'boş belirteç 2' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'boş belirteç 3' })).toBeInTheDocument();
});

test('works for a 32x32 model', () => {
  const values = Array.from({ length: 32 }, (_, l) => Array.from({ length: 32 }, (_, h) => (l * 32 + h) / 1023));
  const { fills } = renderHeatmap(values);
  expect(fills).toHaveLength(1024);
  cleanup();
  render(<HeatmapTable values={values} basket={new Set()} />);
  expect(document.querySelectorAll('tbody tr')).toHaveLength(1024);
}, 20_000); // 1024 table rows in jsdom: slow on a busy machine, not a hang
