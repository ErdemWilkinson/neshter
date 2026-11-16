import { headId, type NodeId } from '@/core/ids';
import type { CellLayout, CellRect } from './cellLayout';
import { readPalette, rgbCss } from './colorScale';

export interface HeatmapMarks {
  selected: ReadonlySet<NodeId>;
  basket: ReadonlySet<NodeId>;
  focus: NodeId | null;
}

/** Largest dot radius in a cell: a hair under half its shorter side, so neighbours never touch. */
const maxRadius = (r: CellRect): number => Math.max(1, Math.min(r.w, r.h) / 2 - 0.75);

/** A 0..1 strength -> dot radius. Area grows with the value (sqrt); a floor keeps the weakest heads visible. */
export function dotRadius(r: CellRect, value: number): number {
  const v = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
  return Math.max(0.9, maxRadius(r) * (0.28 + 0.72 * Math.sqrt(v)));
}

function ring(ctx: CanvasRenderingContext2D, r: CellRect, color: string, width: number, grow: number): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.arc(r.x + r.w / 2, r.y + r.h / 2, maxRadius(r) + grow, 0, Math.PI * 2);
  ctx.stroke();
}

/**
 * The dotted heatmap: one dot per head, its size the strength and its colour the same strength on the viridis
 * ramp. Then the marks: selected (focus ring colour), basket (basket accent) as rings, the keyboard focus as a square.
 */
export function drawHeatmap(
  ctx: CanvasRenderingContext2D,
  layout: CellLayout,
  values: number[][],
  marks: HeatmapMarks,
): void {
  const palette = readPalette();
  const marked: { id: NodeId; rect: CellRect }[] = [];
  values.forEach((row, layer) => {
    row.forEach((value, head) => {
      const id = headId(layer, head);
      const rect = layout.cellRect(id);
      if (rect === null) return;
      ctx.fillStyle = rgbCss(layout.cellColor(id));
      ctx.beginPath();
      ctx.arc(rect.x + rect.w / 2, rect.y + rect.h / 2, dotRadius(rect, value), 0, Math.PI * 2);
      ctx.fill();
      if (marks.selected.has(id) || marks.basket.has(id) || id === marks.focus) marked.push({ id, rect });
    });
  });
  for (const { id, rect } of marked) {
    if (marks.selected.has(id)) ring(ctx, rect, palette.focusRing, 2, 0.5);
    if (marks.basket.has(id)) ring(ctx, rect, palette.basket, 2, marks.selected.has(id) ? 3 : 0.5);
  }
  for (const { id, rect } of marked) {
    if (id !== marks.focus) continue;
    ctx.strokeStyle = palette.focusMark;
    ctx.lineWidth = 2;
    ctx.strokeRect(rect.x + 1, rect.y + 1, Math.max(0, rect.w - 2), Math.max(0, rect.h - 2));
  }
}
