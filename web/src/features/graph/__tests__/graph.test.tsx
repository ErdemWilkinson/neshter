import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createApi } from '@/api/client';
import { setApi } from '@/api/currentApi';
import type { GraphResult } from '@/api/schemas';
import { bodyOf, fakeServer, jsonResponse, type Handler } from '@/app/__tests__/fakeServer';
import type { NodeId } from '@/core/ids';
import { jobsStore } from '@/core/stores/jobs';
import { selectionStore } from '@/core/stores/selection';
import { setLang } from '@/i18n';
import { forgetKeptGraph, GraphPanel } from '../GraphPanel';
import { PITCH_MAX, project, tiltMix, VIEWS } from '../camera';
import { hitTest } from '../drawFlow';
import { dotKinds, HundredDots } from '../HundredDots';
import { extent, incomingShare, liveWeights, placeNodes, trunkWidths, visibleEdges } from '../layout';

// jsdom has no canvas: the scene is replaced by buttons that pick what the real scene would.
vi.mock('../FlowScene', () => ({
  default: (props: { edges: { id: string }[]; cut: ReadonlySet<string>; mode: string; onSelect(p: unknown): void }) => (
    <div data-mode={props.mode}>
      {props.edges.map((e) => (
        <button key={e.id} type="button" onClick={() => props.onSelect({ kind: 'edge', id: e.id })}>
          sahne {e.id}
          {props.cut.has(e.id) ? ' kesik' : ''}
        </button>
      ))}
      <button type="button" onClick={() => props.onSelect({ kind: 'node', id: 'h:0:1' })}>
        sahne düğüm
      </button>
      <button type="button" onClick={() => props.onSelect({ kind: 'cell', id: 'h:1:1' })}>
        sahne hücre
      </button>
    </div>
  ),
}));

const GRAPH: GraphResult = {
  graph_id: 'g1',
  estimated: true,
  warnings: [],
  metric: { kind: 'logit_diff', token_id: 7, text: ' sorted', value: 3.2 },
  candidates: [{ token_id: 7, text: ' sorted' }],
  tokens: [
    { pos: 0, token_id: 1, text: 'liste' },
    { pos: 1, token_id: 2, text: ' sırala' },
  ],
  n_layers: 2,
  n_heads: 2,
  nodes: [
    { id: 't:0', kind: 'token', layer: -1, index: 0, score: 0.1 },
    { id: 't:1', kind: 'token', layer: -1, index: 1, score: 0.2 },
    { id: 'h:0:1', kind: 'head', layer: 0, index: 1, score: 0.9 },
    { id: 'h:1:0', kind: 'head', layer: 1, index: 0, score: -0.4 },
    { id: 'n:1:42', kind: 'neuron', layer: 1, index: 42, score: 0.3 },
    { id: 'out', kind: 'output', layer: 2, index: 7, score: 3.2 },
  ],
  edges: [
    { id: 'e0', src: 'h:0:1', dst: 'h:1:0', channel: 'v', weight: 0.6, badge: 'yalnızca keşif' },
    { id: 'e1', src: 'h:1:0', dst: 'out', channel: 'logit', weight: -0.3, badge: 'yalnızca keşif' },
    { id: 'e2', src: 't:1', dst: 'h:1:0', channel: 'v', weight: 0.2, badge: 'yalnızca keşif' },
    { id: 'e3', src: 'h:0:1', dst: 'n:1:42', channel: 'resid', weight: 0.1, badge: 'yalnızca keşif' },
  ],
  rivers: { trunk: [0, 1, -2, 4], norms: [1, 1, 1, 1] },
  pruning: { kept_nodes: 6, kept_edges: 4, candidate_edges: 30 },
};

const doneJob = { id: 'g1', kind: 'graph', state: 'done', progress: 1, message: '', result: GRAPH, error: null, error_code: null, started_at: null };
const cutReport = {
  edge_ids: ['e0'],
  metric_before: 3.2,
  metric_after: 2.1,
  delta_metric: -1.1,
  delta_estimated: -0.6,
  p_before: 0.41,
  p_after: 0.2,
  kl: 0.0312,
  top_before: [{ token_id: 7, text: ' sorted', p: 0.41 }],
  top_after: [{ token_id: 9, text: ' sort', p: 0.3 }],
};

const HEAT = [
  [0.1, 0.9],
  [0.4, 0.2],
];

function mount(overrides: Record<string, Handler> = {}, heat: number[][] | null = null) {
  const server = fakeServer({
    'POST /api/graph': () => jsonResponse(200, { job_id: 'g1' }),
    'GET /api/jobs/g1': () => jsonResponse(200, doneJob),
    'POST /api/graph/g1/cut': () => jsonResponse(200, cutReport),
    ...overrides,
  });
  setApi(createApi({ token: 't', fetchFn: server.fetchFn as unknown as typeof fetch, onStale: () => undefined }));
  render(<GraphPanel prompt="liste sırala" heat={heat} basket={new Set<NodeId>()} />);
  return server;
}

beforeEach(() => setLang('tr'));
afterEach(() => {
  cleanup();
  forgetKeptGraph();
  jobsStore.setState({ active: null });
  selectionStore.getState().clear();
});

describe('layout', () => {
  test('the floor is the heatmap: rows follow the computation, heads sit on their heatmap column', () => {
    const placed = new Map(placeNodes(GRAPH).map((n) => [n.id, n.pos]));
    const z = (id: string) => placed.get(id)?.[2] ?? NaN;
    expect(z('t:0')).toBeGreaterThan(z('h:0:1'));
    expect(z('h:0:1')).toBeGreaterThan(z('h:1:0'));
    expect(z('h:1:0')).toBeGreaterThan(z('n:1:42')); // a layer's MLP comes after its attention
    expect(z('n:1:42')).toBeGreaterThan(z('out'));
    expect(placed.get('h:0:1')?.[0]).toBeGreaterThan(placed.get('h:1:0')?.[0] ?? Infinity);
  });

  test('seen from straight above, the floor fits the stage and its centre is the stage centre', () => {
    const vp = { w: 800, h: 600, world: extent({ nLayers: 2, nHeads: 2 }) };
    const [cx, cy] = project(VIEWS.heat, vp, [0, 0, 0]);
    expect(cx).toBeCloseTo(400);
    expect(cy).toBeCloseTo(300);
    const [x] = project(VIEWS.heat, vp, [vp.world.w / 2, 0, 0]);
    expect(x).toBeLessThanOrEqual(800);
  });

  test('a cut starves what flows downstream of it, in proportion', () => {
    const live = liveWeights(GRAPH.nodes, GRAPH.edges, new Set(['e0']));
    expect(live.get('e0')).toBe(0);
    expect(live.get('e1')).toBeCloseTo(-0.3 * (0.2 / 0.8));
    expect(live.get('e3')).toBeCloseTo(0.1); // the source itself still writes elsewhere
  });

  test('hit testing: a heatmap cell, then a node, then the nearest river', () => {
    const hits = {
      cells: [{ id: 'h:0:0', corners: [[0, 0], [10, 0], [10, 10], [0, 10]] as [number, number][] }],
      nodes: [{ id: 'h:1:0', x: 50, y: 50, r: 9 }],
      edges: [{ id: 'e1', pts: [[100, 0], [100, 100]] as [number, number][], w: 2 }],
    };
    expect(hitTest(hits, 5, 5)).toEqual({ kind: 'cell', id: 'h:0:0' });
    expect(hitTest(hits, 52, 48)).toEqual({ kind: 'node', id: 'h:1:0' });
    expect(hitTest(hits, 106, 40)).toEqual({ kind: 'edge', id: 'e1' });
    expect(hitTest(hits, 200, 200)).toBeNull();
    // a click inside a thick river picks it, even when a thin one passes closer to the pointer
    const crossing = {
      cells: [],
      nodes: [],
      edges: [
        { id: 'thick', pts: [[300, 0], [300, 100]] as [number, number][], w: 20 },
        { id: 'thin', pts: [[306, 0], [306, 100]] as [number, number][], w: 1 },
      ],
    };
    expect(hitTest(crossing, 305, 50)).toEqual({ kind: 'edge', id: 'thick' });
  });

  test('trunk widths are scaled to the largest positive projection; negative ones are 0', () => {
    expect(trunkWidths([0, 1, -2, 4])).toEqual([0, 0.25, 0, 1]);
    expect(trunkWidths([-1, -2])).toEqual([0, 0]);
  });

  test('a cut edge stays visible below the strength limit', () => {
    expect(visibleEdges(GRAPH.edges, 1, new Set(['e3'])).map((e) => e.id)).toEqual(['e0', 'e3']);
  });

  test('incoming share counts only the same target and channel', () => {
    const e0 = GRAPH.edges[0];
    if (e0 === undefined) throw new Error('fixture');
    expect(incomingShare(GRAPH.edges, e0)).toBeCloseTo(0.6 / 0.8);
  });
});

test('opens on the heatmap; a cell picks the head and the graph view waits for a graph', async () => {
  const user = userEvent.setup();
  mount({}, HEAT);
  expect(screen.getByText(/Isı haritası burada yukarıdan görünüyor/)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Bağlar 3B' })).toBeDisabled();
  await user.click(await screen.findByRole('button', { name: 'sahne hücre' }));
  expect(selectionStore.getState().selected.has('h:1:1' as NodeId)).toBe(true);
});

test('a drawn graph turns the scene into the graph view; the heatmap is one click back', async () => {
  const user = userEvent.setup();
  mount({}, HEAT);
  await user.click(screen.getByRole('button', { name: 'Bağ grafiğini çiz' }));
  await screen.findByRole('button', { name: 'sahne e0' });
  expect(document.querySelector('[data-mode="flow"]')).not.toBeNull();
  await user.click(screen.getByRole('button', { name: 'Isı haritası' }));
  expect(document.querySelector('[data-mode="heat"]')).not.toBeNull();
});

test('a head picked in the graph can join the selection', async () => {
  const user = userEvent.setup();
  mount();
  await user.click(screen.getByRole('button', { name: 'Bağ grafiğini çiz' }));
  await user.click(await screen.findByRole('button', { name: 'sahne düğüm' }));
  await user.click(screen.getByRole('button', { name: 'Seçime ekle' }));
  expect(selectionStore.getState().selected.has('h:0:1' as NodeId)).toBe(true);
  expect(screen.getByRole('button', { name: 'Seçimden çıkar' })).toBeInTheDocument();
});

test('draws the graph, cuts a river and shows the real effect', async () => {
  const user = userEvent.setup();
  const server = mount();
  await user.click(screen.getByRole('button', { name: 'Bağ grafiğini çiz' }));
  await user.click(await screen.findByRole('button', { name: 'sahne e0' }));
  expect(screen.getByText(/%75 kadarını taşıyor/)).toBeInTheDocument();
  expect(screen.getByText(/değer \(V\)/)).toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'Bağı kes (geçici)' }));
  expect(await screen.findByText('"␣sorted" olasılığı: %41 → %20'.replace('␣', ' '))).toBeInTheDocument();
  const cutCall = server.calls.find((c) => c.key === 'POST /api/graph/g1/cut');
  expect(bodyOf(cutCall?.init)).toEqual({ edge_ids: ['e0'] });
  expect(screen.getByRole('button', { name: 'Kesildi' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'sahne e0 kesik' })).toBeInTheDocument();
  expect(screen.getByText('Yol kesimi yalnızca keşif içindir: model dosyası değişmez ve bu kesim kaydedilemez.')).toBeInTheDocument();
  // hundred dots: 41 → 20 is 20 kept, 21 taken by the cut, 59 never there; the new top guess changed
  const report = document.querySelector('.graph-report') as HTMLElement;
  expect(report.querySelectorAll('.hundred-grid .hundred-dot')).toHaveLength(100);
  expect(within(report).getByText('kalan: 20')).toBeInTheDocument();
  expect(within(report).getByText('kesimle giden: 21')).toBeInTheDocument();
  expect(within(report).getByText('zaten yoktu: 59')).toBeInTheDocument();
  expect(within(report).getByText('· değişti')).toBeInTheDocument();
});

test('hundred dots: kept, lost and gained always add up to a hundred', () => {
  expect(dotKinds(0.41, 0.2).filter((k) => k === 'lost')).toHaveLength(21);
  const up = dotKinds(0.1, 0.35);
  expect(up.filter((k) => k === 'kept')).toHaveLength(10);
  expect(up.filter((k) => k === 'gained')).toHaveLength(25);
  expect(dotKinds(2, -1).filter((k) => k === 'lost')).toHaveLength(100);
  expect(dotKinds(Number.NaN, 0.5)).toHaveLength(100);
});

test('undoing the last cut clears the report without another request', async () => {
  const user = userEvent.setup();
  const server = mount();
  await user.click(screen.getByRole('button', { name: 'Bağ grafiğini çiz' }));
  await user.click(await screen.findByRole('button', { name: 'sahne e0' }));
  await user.click(screen.getByRole('button', { name: 'Bağı kes (geçici)' }));
  await screen.findByText(/%41 → %20/);
  const before = server.calls.filter((c) => c.key === 'POST /api/graph/g1/cut').length;
  await user.click(screen.getByRole('button', { name: /kesimini geri al/ }));
  await waitFor(() => expect(screen.queryByText(/%41 → %20/)).toBeNull());
  expect(server.calls.filter((c) => c.key === 'POST /api/graph/g1/cut')).toHaveLength(before);
});

test('cutting all outputs of a node sends every outgoing edge', async () => {
  const user = userEvent.setup();
  const server = mount();
  await user.click(screen.getByRole('button', { name: 'Bağ grafiğini çiz' }));
  await user.click(await screen.findByRole('button', { name: 'sahne düğüm' }));
  await user.click(screen.getByRole('button', { name: 'Bütün çıkışlarını kes (2)' }));
  await screen.findByText(/%41 → %20/);
  const call = server.calls.find((c) => c.key === 'POST /api/graph/g1/cut');
  expect(bodyOf(call?.init)).toEqual({ edge_ids: ['e0', 'e3'] });
});

test('a failed graph job shows an error card', async () => {
  const user = userEvent.setup();
  mount({
    'GET /api/jobs/g1': () =>
      jsonResponse(200, { ...doneJob, state: 'error', result: null, error: 'Bağ grafiği şimdilik yalnızca Llama ve Qwen2', error_code: 'unsupported' }),
  });
  await user.click(screen.getByRole('button', { name: 'Bağ grafiğini çiz' }));
  expect(await screen.findByRole('alert')).toBeInTheDocument();
});

test('a new run (a new prompt) drops the old graph and its cuts', async () => {
  const user = userEvent.setup();
  const server = fakeServer({
    'POST /api/graph': () => jsonResponse(200, { job_id: 'g1' }),
    'GET /api/jobs/g1': () => jsonResponse(200, doneJob),
    'POST /api/graph/g1/cut': () => jsonResponse(200, cutReport),
  });
  setApi(createApi({ token: 't', fetchFn: server.fetchFn as unknown as typeof fetch, onStale: () => undefined }));
  const { rerender } = render(<GraphPanel prompt="liste sırala" heat={HEAT} basket={new Set<NodeId>()} />);
  await user.click(screen.getByRole('button', { name: 'Bağ grafiğini çiz' }));
  await user.click(await screen.findByRole('button', { name: 'sahne e0' }));
  await user.click(screen.getByRole('button', { name: 'Bağı kes (geçici)' }));
  await screen.findByText(/%41 → %20/);
  rerender(<GraphPanel prompt="başka bir cümle" heat={HEAT} basket={new Set<NodeId>()} />);
  expect(screen.queryByRole('button', { name: 'sahne e0' })).toBeNull();
  expect(screen.queryByText(/%41 → %20/)).toBeNull();
  expect(document.querySelector('[data-mode="heat"]')).not.toBeNull();
  expect(screen.getByRole('button', { name: 'Bağ grafiğini çiz' })).toBeEnabled();
});

test('the mode buttons are reachable: the stage is a group, not a picture', async () => {
  mount({}, HEAT);
  const stage = screen.getByRole('group', { name: /3B bağ grafiği/ });
  expect(stage).not.toHaveAttribute('role', 'img');
  expect(screen.getByRole('button', { name: 'Isı haritası' })).toBeInTheDocument();
});

test('leaving the 3D view and coming back keeps the drawn graph and its cuts', async () => {
  const user = userEvent.setup();
  const server = mount();
  await user.click(screen.getByRole('button', { name: 'Bağ grafiğini çiz' }));
  await user.click(await screen.findByRole('button', { name: 'sahne e0' }));
  await user.click(screen.getByRole('button', { name: 'Bağı kes (geçici)' }));
  await screen.findByText(/%41 → %20/);
  cleanup(); // the 2D map (or another screen) replaces the panel
  render(<GraphPanel prompt="liste sırala" heat={null} basket={new Set<NodeId>()} />);
  expect(screen.getByRole('button', { name: 'sahne e0 kesik' })).toBeInTheDocument();
  expect(screen.getByText(/%41 → %20/)).toBeInTheDocument();
  expect(server.calls.filter((c) => c.key === 'POST /api/graph')).toHaveLength(1);
});

test('the graph fades up with the tilt: straight above is the dotted map, a tilted view the full graph', () => {
  expect(tiltMix(PITCH_MAX)).toBe(0);
  expect(tiltMix(VIEWS.heat.pitch)).toBe(0);
  expect(tiltMix(VIEWS.tilt.pitch)).toBe(1);
  expect(tiltMix(VIEWS.low.pitch)).toBe(1);
  expect(tiltMix(1.2)).toBeGreaterThan(0);
  expect(tiltMix(1.2)).toBeLessThan(1);
});

test('overlapping towers: the one drawn last (the nearest) is picked', () => {
  const square = [[0, 0], [10, 0], [10, 10], [0, 10]] as [number, number][];
  const hits = { cells: [{ id: 'h:0:0', corners: square }, { id: 'h:5:0', corners: square }], nodes: [], edges: [] };
  expect(hitTest(hits, 5, 5)).toEqual({ kind: 'cell', id: 'h:5:0' });
});

test('the heatmap can be seen as towers without drawing a graph', async () => {
  const user = userEvent.setup();
  mount({}, HEAT);
  await user.click(screen.getByRole('button', { name: 'Isı 3B' }));
  expect(document.querySelector('[data-mode="towers"]')).not.toBeNull();
  expect(screen.getByText(/sütun yüksekliği/)).toBeInTheDocument();
});

describe('on the phone', () => {
  beforeEach(() => {
    (window as unknown as { NeshterNative?: unknown }).NeshterNative = { call: () => undefined };
  });
  afterEach(() => {
    delete (window as unknown as { NeshterNative?: unknown }).NeshterNative;
  });

  test('opens on the towers and offers no graph (it needs gradients)', () => {
    mount({}, HEAT);
    expect(screen.getByRole('heading', { name: '3B ısı haritası' })).toBeInTheDocument();
    expect(document.querySelector('[data-mode="towers"]')).not.toBeNull();
    expect(screen.queryByRole('button', { name: 'Bağ grafiğini çiz' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Bağlar 3B' })).toBeNull();
    expect(screen.getByText(/bilgisayardaki Neşter'de çizilebilir/)).toBeInTheDocument();
  });
});

test('the big before/after numbers follow the language: 48% in English, %48 in Turkish', () => {
  setLang('en');
  const { container, unmount } = render(<HundredDots before={0.48} after={0.2} sentence="s" cuts={[]} guess="the" changed />);
  expect(container.querySelector('.hundred-head')?.textContent).toBe('48% 20%');
  unmount();
  setLang('tr');
  const tr = render(<HundredDots before={0.48} after={0.2} sentence="s" cuts={[]} guess="the" changed />);
  expect(tr.container.querySelector('.hundred-head')?.textContent).toBe('%48 %20');
});
