import type { GraphEdge } from '@/api/schemas';
import { heatColor, rgbCss } from '@/features/map/colorScale';
import { clamp01, easeOut, project, type Camera, type Viewport } from './camera';
import { CURVE_STEPS, headX, layerZ, COL, ROW, type Grid, type PlacedNode, type Vec3 } from './layout';

export type Pick = { kind: 'node'; id: string } | { kind: 'edge'; id: string } | { kind: 'cell'; id: string };
export interface Hover {
  pick: Pick;
  x: number;
  y: number;
}

export interface FlowPalette {
  bg: string;
  line: string;
  text1: string;
  text2: string;
  inc: string;
  dec: string;
  cut: string;
  basket: string;
  focus: string;
  /** The bright dots of light travelling along the rivers. */
  light: string;
  /** Rings around selected heads (red, as on the dotted map). */
  select: string;
  mono: string;
}

const NEON = { river: '#2ee6c5', light: '#c8fff5', cut: '#ff7a45' } as const;

/** The app's look tokens; fallbacks are the dark values in tokens.css (jsdom has no stylesheet). */
export function readFlowPalette(): FlowPalette {
  const style = typeof document === 'undefined' ? null : getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => style?.getPropertyValue(name).trim() || fallback;
  return {
    bg: v('--surface-0', '#0b0a14'),
    line: v('--border', '#272545'),
    text1: v('--text-1', '#e6eaf2'),
    text2: v('--text-2', '#a7b0c0'),
    // the scene keeps its own neon look (the one picked in the design catalog): teal rivers, pale lights, orange cuts
    inc: NEON.river,
    dec: v('--accent-decrease', '#a38cff'),
    cut: NEON.cut,
    basket: v('--accent-basket', '#ff7a45'),
    focus: v('--focus-ring', '#fde725'),
    light: NEON.light,
    select: v('--select-ring', '#ff3b30'),
    mono: v('--font-mono', 'ui-monospace, monospace'),
  };
}

export interface SceneEdge extends GraphEdge {
  pts: Vec3[];
  /** 0..1 depth of the source along the computation, so rivers grow from the words toward the answer. */
  from: number;
}

export interface SceneData {
  grid: Grid;
  nodes: readonly PlacedNode[];
  /** Visible edges, weakest first so strong rivers are drawn on top. */
  edges: readonly SceneEdge[];
  trunk: readonly number[];
  /** Normalised heatmap values [layer][head], 0..1; null before a heatmap exists. */
  heat: readonly (readonly number[])[] | null;
  maxWeight: number;
}

export interface FrameState {
  t: number;
  /** 0 = heatmap seen from above … 1 = connection graph. */
  mix: number;
  /** 0 = flat heatmap … 1 = every head a tower as tall as its power (the 3D heatmap; needs no graph). */
  rise: number;
  cam: Camera;
  vp: Viewport;
  /** Eased drawn |weight| per edge. */
  widths: ReadonlyMap<string, number>;
  cut: ReadonlySet<string>;
  cutAt: ReadonlyMap<string, number>;
  basket: ReadonlySet<string>;
  basketAt: ReadonlyMap<string, number>;
  chosen: ReadonlySet<string>;
  focus: ReadonlySet<string>;
  highlight: ReadonlySet<string>;
  labels: ReadonlySet<string>;
  reduced: boolean;
  palette: FlowPalette;
  name(id: string): string;
  /** The short layer label on the axis ("L5" in English, "K5" in Turkish). */
  layerName(layer: number): string;
}

export interface Hits {
  cells: { id: string; corners: [number, number][] }[];
  nodes: { id: string; x: number; y: number; r: number }[];
  /** `w`: the drawn width in px, so a click on a thick river is not taken by a thin one beneath it. */
  edges: { id: string; pts: [number, number][]; w: number }[];
}

const CUT_SECONDS = 0.9;
const REMOVE_SECONDS = 0.7;

function polyline(ctx: CanvasRenderingContext2D, ps: readonly (readonly number[])[]) {
  ctx.beginPath();
  ps.forEach((p, i) => (i === 0 ? ctx.moveTo(p[0] ?? 0, p[1] ?? 0) : ctx.lineTo(p[0] ?? 0, p[1] ?? 0)));
}

function cross(ctx: CanvasRenderingContext2D, x: number, y: number, d: number, color: string, width: number) {
  if (d <= 0) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x - d, y - d);
  ctx.lineTo(x + d, y + d);
  ctx.moveTo(x - d, y + d);
  ctx.lineTo(x + d, y - d);
  ctx.stroke();
}

/** The soft neon glow on rivers, their lights, head rings and the answer, in px of blur; kept low on purpose. */
const GLOW = { river: 4, light: 5, ring: 3, answer: 6 } as const;

/** Draws one frame and returns what can be hit at the positions just drawn. */
export function drawFlow(ctx: CanvasRenderingContext2D, scene: SceneData, f: FrameState): Hits {
  const { grid, nodes } = scene;
  const { palette: pal, mix, t } = f;
  const P = (p: Vec3) => project(f.cam, f.vp, p);
  const heatA = 1 - easeOut(mix * 1.5);
  const flowA = easeOut((mix - 0.35) / 0.65);
  const nodeA = easeOut((mix - 0.3) / 0.5);
  const ringA = easeOut((mix - 0.2) / 0.5);
  const hits: Hits = { cells: [], nodes: [], edges: [] };
  const color = heatColor;
  const px = P([0, 0, 0])[2] * Math.min(1.4, Math.max(0.55, (P([COL, 0, 0])[0] - P([0, 0, 0])[0]) / 60));

  ctx.fillStyle = pal.bg;
  ctx.fillRect(0, 0, f.vp.w, f.vp.h);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // rows: floor lines in the graph view (brighter where the main stream is closer to the answer), labels in both
  const rowPx = Math.abs(P([0, 0, layerZ(0, grid.nLayers)])[1] - P([0, 0, layerZ(1, grid.nLayers)])[1]);
  const rowStep = rowPx >= 12 || grid.nLayers <= 1 ? 1 : 5;
  const half = (grid.nHeads * COL) / 2 + 20;
  for (let l = 0; l < grid.nLayers; l++) {
    const z = layerZ(l, grid.nLayers);
    const a = P([-half, 0, z]);
    const b = P([half, 0, z]);
    if (flowA > 0) {
      // fine floor lines, brighter toward the answer
      const depth = (l + 1) / Math.max(1, grid.nLayers);
      ctx.globalAlpha = flowA * (0.12 + 0.3 * depth);
      ctx.strokeStyle = depth > 0.3 ? pal.inc : pal.line;
      ctx.lineWidth = 0.5 + depth * 0.6;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }
    if (l % rowStep === 0) {
      ctx.globalAlpha = 1;
      ctx.fillStyle = pal.text2;
      ctx.textAlign = 'right';
      ctx.font = `11px ${pal.mono}`;
      ctx.fillText(f.layerName(l), a[0] - 6, a[1] + 4);
    }
  }
  ctx.globalAlpha = 1;
  if (heatA > 0) {
    const colStep = grid.nHeads > 24 ? 4 : 1;
    ctx.globalAlpha = heatA;
    ctx.fillStyle = pal.text2;
    ctx.textAlign = 'center';
    for (let h = 0; h < grid.nHeads; h += colStep) {
      const p = P([headX(h, grid.nHeads), 0, layerZ(0, grid.nLayers) + ROW * 0.72]);
      ctx.fillText(`H${h}`, p[0], p[1]);
    }
    ctx.globalAlpha = 1;
  }

  // the floor is the dotted map: one small dot per head, larger and brighter the stronger it is. A head in the
  // graph pulls its dot into its ring as the scene tilts; the others stay on the floor under the rivers.
  const graphHeads = new Set(nodes.filter((n) => n.kind === 'head').map((n) => n.id));
  const o = P([0, 0, 0]);
  const unit = P([COL, 0, 0]);
  const rmax = 0.46 * Math.hypot(unit[0] - o[0], unit[1] - o[1]) * Math.min(1, ROW / COL);
  const squash = Math.max(0.35, Math.sin(f.cam.pitch));
  // Towers are drawn far to near so the nearer ones cover the ones behind them.
  const towerH = f.rise * Math.max(4 * ROW, 0.26 * grid.nLayers * ROW);
  const cells: [number, number][] = [];
  for (let l = 0; l < grid.nLayers; l++) for (let h = 0; h < grid.nHeads; h++) cells.push([l, h]);
  if (towerH > 0) {
    const far = (l: number, h: number) => headX(h, grid.nHeads) * Math.sin(f.cam.yaw) + layerZ(l, grid.nLayers) * Math.cos(f.cam.yaw);
    cells.sort((a, b) => far(b[0], b[1]) - far(a[0], a[1]));
  }
  for (const [l, h] of cells) {
    const id = `h:${l}:${h}`;
    const x = headX(h, grid.nHeads);
    const z = layerZ(l, grid.nLayers);
    const top = towerH * clamp01(scene.heat?.[l]?.[h] ?? 0); // a tower is picked by its lid
    const corners = ([[-1, -1], [1, -1], [1, 1], [-1, 1]] as const).map(([i, j]): [number, number] => {
      const p = P([x + (i * COL * 0.88) / 2, top, z + (j * ROW * 0.84) / 2]);
      return [p[0], p[1]];
    });
    if (mix < 0.5) hits.cells.push({ id, corners });
    const inGraph = graphHeads.has(id);
    const shrink = inGraph ? 1 - 0.88 * easeOut(mix * 1.3) : 1 - 0.3 * easeOut(mix);
    const alpha = inGraph ? 1 - easeOut((mix - 0.25) / 0.5) : Math.max(heatA, 0.7);
    if (alpha <= 0) continue;
    const inBasket = f.basket.has(id);
    const v = clamp01(scene.heat?.[l]?.[h] ?? 0);
    const base = P([x, 0, z]);
    const c = towerH > 0 && !inGraph ? P([x, towerH * v, z]) : base;
    const rx = Math.max(1, rmax * (0.3 + 0.7 * Math.sqrt(v)) * shrink);
    ctx.globalAlpha = alpha * (0.65 + 0.35 * v) * (inBasket ? 0.45 : 1);
    if (c !== base) {
      // the tower's side: a band from its foot to its top, darker than the top so the top reads as a lid
      ctx.fillStyle = rgbCss(color(0.1 + 0.6 * v));
      ctx.beginPath();
      ctx.moveTo(base[0] - rx, base[1]);
      ctx.lineTo(c[0] - rx, c[1]);
      ctx.lineTo(c[0] + rx, c[1]);
      ctx.lineTo(base[0] + rx, base[1]);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.ellipse(base[0], base[1], rx, rx * squash, 0, 0, Math.PI);
      ctx.fill();
    }
    ctx.fillStyle = rgbCss(color(0.18 + 0.82 * v)); // the darkest violet would vanish on the dark floor
    ctx.beginPath();
    ctx.ellipse(c[0], c[1], rx, rx * squash, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = alpha;
    if (f.chosen.has(id) || f.focus.has(id)) {
      ctx.strokeStyle = f.focus.has(id) ? pal.text1 : pal.select;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(c[0], c[1], rmax + 1.5, (rmax + 1.5) * squash, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    const cw = COL * 0.88;
    const cd = ROW * 0.84;
    if (inBasket) {
      const k = f.reduced ? 1 : easeOut((t - (f.basketAt.get(id) ?? -9)) / 0.35);
      cross(ctx, c[0], c[1], Math.min(cw, cd) * 0.22 * fitPx(f) * k, pal.basket, 2);
    }
    ctx.globalAlpha = 1;
  }

  // rivers
  if (flowA > 0) {
    for (const e of scene.edges) {
      const grow = clamp01((mix - 0.35 - 0.3 * e.from) / 0.35);
      if (grow <= 0) continue;
      const ps = e.pts.map(P);
      const k = ((ps[0]?.[2] ?? 1) + (ps[CURVE_STEPS]?.[2] ?? 1)) / 2;
      const col = pal.inc;
      const live = (f.widths.get(e.id) ?? Math.abs(e.weight)) / scene.maxWeight;
      const w = (1.1 + 7 * live) * k * px;
      if (mix >= 0.95) hits.edges.push({ id: e.id, pts: ps.map((p) => [p[0], p[1]]), w });
      if (f.cut.has(e.id)) {
        drawCut(ctx, ps, w, col, f, e.id, flowA);
        continue;
      }
      const shown = ps.slice(0, Math.max(1, Math.round(grow * CURVE_STEPS)) + 1);
      if (f.highlight.has(e.id)) {
        ctx.globalAlpha = flowA;
        ctx.strokeStyle = pal.text1;
        ctx.lineWidth = w + 3;
        polyline(ctx, shown);
        ctx.stroke();
      }
      ctx.globalAlpha = flowA * (0.3 + 0.55 * Math.min(1, live * 2.2));
      ctx.strokeStyle = col;
      ctx.lineWidth = w;
      ctx.shadowColor = col;
      ctx.shadowBlur = GLOW.river;
      polyline(ctx, shown);
      ctx.stroke();
      // light travelling along the river; count and speed follow what still flows
      if (!f.reduced && grow >= 1 && live > 0.04) {
        ctx.globalAlpha = flowA;
        ctx.fillStyle = pal.light;
        ctx.shadowBlur = GLOW.light;
        const n = Math.round(1 + live * 5);
        const seed = e.id.length * 0.37 + (e.id.charCodeAt(e.id.length - 1) % 10) * 0.11;
        for (let j = 0; j < n; j++) {
          const u = (t * (0.18 + live * 0.12) + j / n + seed) % 1;
          const fi = u * CURVE_STEPS;
          const i0 = Math.floor(fi);
          const a = ps[i0];
          const b = ps[Math.min(CURVE_STEPS, i0 + 1)];
          if (a === undefined || b === undefined) continue;
          const q = fi - i0;
          ctx.beginPath();
          ctx.arc(a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q, Math.max(1.2, w * 0.32), 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;
    }
  }

  // nodes and their labels
  const labels: [PlacedNode, [number, number, number], number][] = [];
  for (const n of nodes) {
    const p = P(n.pos);
    const k = p[2] * px;
    if (n.kind === 'head') {
      if (ringA <= 0) continue;
      const r = 6.5 * k;
      if (mix >= 0.95) hits.nodes.push({ id: n.id, x: p[0], y: p[1], r: Math.max(9, r + 3) });
      ctx.globalAlpha = ringA;
      if (f.basket.has(n.id)) drawRemoved(ctx, p, r, f, n.id);
      else {
        const lit = f.focus.has(n.id);
        ctx.fillStyle = pal.bg;
        ctx.strokeStyle = lit ? pal.text1 : f.chosen.has(n.id) ? pal.select : pal.inc;
        ctx.lineWidth = lit ? 2.6 : 2.2;
        ctx.shadowColor = lit ? pal.text1 : f.chosen.has(n.id) ? pal.select : pal.inc;
        ctx.shadowBlur = GLOW.ring;
        ctx.beginPath();
        ctx.arc(p[0], p[1], lit ? r + 1 : r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.fillStyle = rgbCss(color(scene.heat?.[n.layer]?.[n.index] ?? 0.5));
        ctx.beginPath();
        ctx.arc(p[0], p[1], r * 0.45, 0, Math.PI * 2);
        ctx.fill();
      }
      if (f.labels.has(n.id) || f.focus.has(n.id) || f.basket.has(n.id)) labels.push([n, p, ringA]);
      continue;
    }
    if (nodeA <= 0) continue;
    if (mix >= 0.95 && n.kind !== 'token') hits.nodes.push({ id: n.id, x: p[0], y: p[1], r: Math.max(9, 8 * k) });
    ctx.globalAlpha = nodeA;
    if (n.kind === 'neuron') {
      const d = 3.5 * k;
      ctx.globalAlpha = nodeA * 0.7;
      ctx.fillStyle = pal.bg;
      ctx.strokeStyle = f.focus.has(n.id) ? pal.text1 : pal.text2;
      ctx.lineWidth = 2;
      ctx.fillRect(p[0] - d, p[1] - d, 2 * d, 2 * d);
      ctx.strokeRect(p[0] - d, p[1] - d, 2 * d, 2 * d);
      if (f.focus.has(n.id)) labels.push([n, p, nodeA]);
    } else if (n.kind === 'output') {
      ctx.fillStyle = pal.bg;
      ctx.strokeStyle = pal.text1;
      ctx.lineWidth = 2.5;
      ctx.shadowColor = pal.inc;
      ctx.shadowBlur = GLOW.answer;
      ctx.beginPath();
      ctx.arc(p[0], p[1], 10 * k, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = pal.inc;
      ctx.beginPath();
      ctx.arc(p[0], p[1], 4.5 * k, 0, Math.PI * 2);
      ctx.fill();
      labels.push([n, p, nodeA]);
    } else {
      ctx.fillStyle = pal.text1;
      ctx.fillRect(p[0] - 3 * k, p[1] - 3 * k, 6 * k, 6 * k);
      labels.push([n, p, nodeA]);
    }
    ctx.globalAlpha = 1;
  }
  // Labels that would overlap one already placed are dropped; the lit ones and the answer are placed first.
  const placed: [number, number, number, number][] = [];
  const rank = (n: PlacedNode) => (f.focus.has(n.id) ? 0 : n.kind === 'output' ? 1 : n.kind === 'token' ? 3 : 2);
  labels.sort((x, y) => rank(x[0]) - rank(y[0]));
  for (const [n, p, a] of labels) {
    const lit = f.focus.has(n.id);
    const text = f.name(n.id);
    ctx.font = `${lit || n.kind === 'output' ? 600 : 500} ${n.kind === 'output' ? 13 : 11}px ${pal.mono}`;
    const centred = n.kind === 'token' || n.kind === 'output';
    const x = centred ? p[0] : p[0] + 10 * p[2] * px;
    const y = centred ? (n.kind === 'output' ? p[1] + 26 * p[2] * px : p[1] - 10) : p[1] + 4;
    const tw = ctx.measureText(text).width;
    const box: [number, number, number, number] = [centred ? x - tw / 2 - 2 : x - 2, y - 11, tw + 4, 14];
    const clash = placed.some((b) => box[0] < b[0] + b[2] && b[0] < box[0] + box[2] && box[1] < b[1] + b[3] && b[1] < box[1] + box[3]);
    if (clash && !lit && n.kind !== 'output') continue;
    placed.push(box);
    ctx.globalAlpha = a;
    ctx.textAlign = centred ? 'center' : 'left';
    ctx.fillStyle = centred || lit ? pal.text1 : f.basket.has(n.id) ? pal.basket : pal.text2;
    ctx.fillText(text, x, y);
  }
  ctx.globalAlpha = 1;
  ctx.textAlign = 'left';
  return hits;
}

/** Cell size in px at the floor's centre, as a 0.5..1.5 factor (so marks keep a sane size on any model). */
function fitPx(f: FrameState): number {
  const a = project(f.cam, f.vp, [0, 0, 0]);
  const b = project(f.cam, f.vp, [COL, 0, 0]);
  return Math.min(1.5, Math.max(0.5, (b[0] - a[0]) / COL));
}

/** A cut: the river snaps at its middle, a short ring marks the spot, both halves pull back to their ends, and a
 * dashed trace with a ✕ stays. */
function drawCut(
  ctx: CanvasRenderingContext2D,
  ps: readonly [number, number, number][],
  w: number,
  col: string,
  f: FrameState,
  id: string,
  alpha: number,
) {
  const since = f.t - (f.cutAt.get(id) ?? -9);
  const k = f.reduced ? 1 : easeOut(since / CUT_SECONDS);
  const mid = CURVE_STEPS / 2;
  const m = ps[mid];
  ctx.globalAlpha = alpha * 0.7;
  ctx.setLineDash([4, 6]);
  ctx.strokeStyle = f.palette.cut;
  ctx.lineWidth = 1.4;
  polyline(ctx, ps);
  ctx.stroke();
  ctx.setLineDash([]);
  if (k < 1) {
    const up = Math.round(mid - mid * k);
    const down = Math.round(mid + mid * k);
    ctx.globalAlpha = alpha * (1 - 0.8 * k);
    ctx.strokeStyle = col;
    ctx.lineWidth = w * (1 - 0.5 * k);
    if (up > 1) {
      polyline(ctx, ps.slice(0, up));
      ctx.stroke();
    }
    if (down < CURVE_STEPS - 1) {
      polyline(ctx, ps.slice(down));
      ctx.stroke();
    }
    const ring = easeOut(since / 0.35);
    if (ring < 1 && m !== undefined) {
      ctx.globalAlpha = alpha * (1 - ring);
      ctx.strokeStyle = f.palette.text1;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(m[0], m[1], 4 + 16 * ring, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = alpha;
  if (m !== undefined) cross(ctx, m[0], m[1], 5.5 * Math.min(1, k * 3), f.palette.cut, 2.4);
  ctx.globalAlpha = 1;
}

/** A head in the basket: its ring closes in on itself, then a dashed outline and a ✕ stay where it was. */
function drawRemoved(ctx: CanvasRenderingContext2D, p: readonly number[], r: number, f: FrameState, id: string) {
  const [x = 0, y = 0] = p;
  const k = f.reduced ? 1 : easeOut((f.t - (f.basketAt.get(id) ?? -9)) / REMOVE_SECONDS);
  const base = ctx.globalAlpha;
  if (k < 1) {
    ctx.strokeStyle = f.palette.inc;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.arc(x, y, Math.max(0.1, r * (1 - k)), 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = base * (1 - k);
    ctx.strokeStyle = f.palette.text1;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(x, y, r + 18 * k, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = base;
  }
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = f.palette.basket;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.arc(x, y, r + 0.5, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  cross(ctx, x, y, r * 0.6 * k, f.palette.basket, 2.2);
}

function inPolygon(x: number, y: number, c: readonly [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = c.length - 1; i < c.length; j = i++) {
    const a = c[i];
    const b = c[j];
    if (a === undefined || b === undefined) continue;
    if (a[1] > y !== b[1] > y && x < ((b[0] - a[0]) * (y - a[1])) / (b[1] - a[1]) + a[0]) inside = !inside;
  }
  return inside;
}

/** What is under (x, y): a heatmap cell while the heatmap shows, else a node, else the nearest river within 8 px. */
export function hitTest(hits: Hits, x: number, y: number): Pick | null {
  // the last drawn cell is on top (towers are drawn far to near)
  const cell = hits.cells.findLast((c) => inPolygon(x, y, c.corners));
  if (cell !== undefined) return { kind: 'cell', id: cell.id };
  for (const n of hits.nodes) if (Math.hypot(x - n.x, y - n.y) <= n.r) return { kind: 'node', id: n.id };
  let best: Pick | null = null;
  let bestD = 0;
  for (const e of hits.edges) {
    const reach = Math.max(8, e.w / 2 + 3);
    for (let i = 0; i < e.pts.length - 1; i++) {
      const a = e.pts[i];
      const b = e.pts[i + 1];
      if (a === undefined || b === undefined) continue;
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const u = clamp01(((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy || 1));
      // inside a river counts by how deep inside: a thick one wins over a thin one crossing it
      const d = Math.hypot(x - a[0] - u * dx, y - a[1] - u * dy);
      const depth = reach - d;
      if (depth > bestD) {
        bestD = depth;
        best = { kind: 'edge', id: e.id };
      }
    }
  }
  return best;
}
