import { headId, type NodeId } from '@/core/ids';
import type { CellLayout, CellRect } from './cellLayout';
import { readPalette, rgbCss } from './colorScale';

export interface HeatmapMarks {
  selected: ReadonlySet<NodeId>;
  basket: ReadonlySet<NodeId>;
  focus: NodeId | null;
  /** Cells drawn larger than their slot (1 = flat), lifted above their neighbours: the hover and selection pop. */
  lift?: ReadonlyMap<NodeId, number>;
  /** The cell under the mouse: it gets a light edge. */
  hover?: NodeId | null;
}

/** Space between neighbouring tiles in px, so each head reads as its own square even when two match. */
export const TILE_GAP = 1;

function outline(ctx: CanvasRenderingContext2D, r: CellRect, color: string, width: number, inset: number): void {
  const i = Math.min(inset, r.w / 4, r.h / 4);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.strokeRect(r.x + i, r.y + i, Math.max(0, r.w - 2 * i), Math.max(0, r.h - 2 * i));
}

/** `r` scaled by `s` about its centre. */
export function grow(r: CellRect, s: number): CellRect {
  if (s === 1) return r;
  const w = r.w * s;
  const h = r.h * s;
  return { x: r.x - (w - r.w) / 2, y: r.y - (h - r.h) / 2, w, h };
}

/** `r` moved (not shrunk) so it stays within the box: a lifted edge cell is not cut off by the canvas. */
function keepInside(r: CellRect, x0: number, y0: number, x1: number, y1: number): CellRect {
  const fit = (at: number, size: number, lo: number, hi: number): number =>
    size >= hi - lo ? (lo + hi - size) / 2 : Math.min(Math.max(at, lo), hi - size);
  const x = fit(r.x, r.w, x0, x1);
  const y = fit(r.y, r.h, y0, y1);
  return { ...r, x, y };
}

/**
 * The tiled heatmap: one square per head, its colour the strength on the viridis ramp (dark violet weak, bright
 * yellow strong). Then the outlines: selected (red), basket (basket accent, inside it), the keyboard focus (light).
 * Lifted cells (hover, selection) come last, larger and with a shadow, so they sit on top of their neighbours.
 */
export function drawHeatmap(
  ctx: CanvasRenderingContext2D,
  layout: CellLayout,
  values: number[][],
  marks: HeatmapMarks,
): void {
  const palette = readPalette();
  const g = TILE_GAP / 2;
  const liftOf = (id: NodeId): number => marks.lift?.get(id) ?? 1;
  const flatMarked: { id: NodeId; rect: CellRect }[] = [];
  const lifted: { id: NodeId; rect: CellRect; s: number }[] = [];
  const rings = (id: NodeId, rect: CellRect): void => {
    if (marks.selected.has(id)) outline(ctx, rect, palette.focusRing, 2.5, 1.5);
    if (marks.basket.has(id)) outline(ctx, rect, palette.basket, 2, marks.selected.has(id) ? 4.5 : 1.5);
  };
  values.forEach((row, layer) => {
    row.forEach((_, head) => {
      const id = headId(layer, head);
      const rect = layout.cellRect(id);
      if (rect === null) return;
      const s = liftOf(id);
      if (s > 1.001) {
        lifted.push({ id, rect, s });
        return;
      }
      ctx.fillStyle = rgbCss(layout.cellColor(id));
      ctx.fillRect(rect.x + g, rect.y + g, Math.max(0, rect.w - TILE_GAP), Math.max(0, rect.h - TILE_GAP));
      if (marks.selected.has(id) || marks.basket.has(id) || id === marks.focus) flatMarked.push({ id, rect });
    });
  });
  for (const { id, rect } of flatMarked) rings(id, rect);
  // Smallest lift first, so the hovered cell (the largest) ends up on top.
  lifted.sort((a, b) => a.s - b.s);
  const first = layout.cellRect(headId(0, 0));
  const last = layout.cellRect(headId(layout.n_layers - 1, layout.n_heads - 1));
  const inside = (r: CellRect): CellRect =>
    first === null || last === null ? r : keepInside(r, first.x, first.y, last.x + last.w, last.y + last.h);
  for (const { id, rect, s } of lifted) {
    const r = inside(grow(rect, s));
    const t = Math.min(1, (s - 1) / 0.35);
    ctx.shadowColor = 'rgba(0, 0, 0, 0.55)';
    ctx.shadowBlur = 10 * t;
    ctx.shadowOffsetY = 2 * t;
    ctx.fillStyle = rgbCss(layout.cellColor(id));
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;
    if (id === marks.hover) outline(ctx, r, palette.focusMark, 1.5, 0.75);
    rings(id, r);
  }
  const focusRect = marks.focus === null ? null : layout.cellRect(marks.focus);
  if (marks.focus !== null && focusRect !== null) {
    const s = liftOf(marks.focus);
    outline(ctx, s > 1.001 ? inside(grow(focusRect, s)) : focusRect, palette.focusMark, 2, 0);
  }
}
