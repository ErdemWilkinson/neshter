import { headId, type NodeId } from '@/core/ids';
import type { CellLayout, CellRect } from './cellLayout';
import { readPalette, rgbCss } from './colorScale';

export interface HeatmapMarks {
  selected: ReadonlySet<NodeId>;
  basket: ReadonlySet<NodeId>;
  focus: NodeId | null;
}

function outline(ctx: CanvasRenderingContext2D, r: CellRect, color: string, width: number, inset: number): void {
  const i = Math.min(inset, r.w / 4, r.h / 4);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.strokeRect(r.x + i, r.y + i, Math.max(0, r.w - 2 * i), Math.max(0, r.h - 2 * i));
}

/** A selected cell is lightened (not just outlined), so it reads at a glance on dark and on bright cells alike. */
const lighten = ([r, g, b]: [number, number, number]): string =>
  rgbCss([Math.round(r + (255 - r) * 0.4), Math.round(g + (255 - g) * 0.4), Math.round(b + (255 - b) * 0.4)]);

/** One `fillRect` per cell, then the outlines: selected (focus ring), basket (basket accent), the keyboard focus. */
export function drawHeatmap(
  ctx: CanvasRenderingContext2D,
  layout: CellLayout,
  values: number[][],
  marks: HeatmapMarks,
): void {
  const palette = readPalette();
  const outlined: { id: NodeId; rect: CellRect }[] = [];
  values.forEach((row, layer) => {
    row.forEach((_, head) => {
      const id = headId(layer, head);
      const rect = layout.cellRect(id);
      if (rect === null) return;
      ctx.fillStyle = marks.selected.has(id) ? lighten(layout.cellColor(id)) : rgbCss(layout.cellColor(id));
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
      if (marks.selected.has(id) || marks.basket.has(id) || id === marks.focus) outlined.push({ id, rect });
    });
  });
  for (const { id, rect } of outlined) {
    if (marks.selected.has(id)) outline(ctx, rect, palette.focusRing, 3, 1.5);
    if (marks.basket.has(id)) outline(ctx, rect, palette.basket, 2, 3);
  }
  for (const { id, rect } of outlined) {
    if (id === marks.focus) outline(ctx, rect, palette.focusMark, 2, 0);
  }
}
