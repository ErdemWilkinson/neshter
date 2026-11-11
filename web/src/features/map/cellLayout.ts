import { headId, type NodeId } from '@/core/ids';

export interface CellRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Rows are layers (K0 at the top), columns are heads. Rects are viewport px; Plan 3's fly-in starts from them. */
export interface CellLayout {
  n_layers: number;
  n_heads: number;
  cellRect(id: NodeId): CellRect | null;
  cellColor(id: NodeId): [number, number, number];
}

const HEAD_ID = /^h:(0|[1-9]\d*):(0|[1-9]\d*)$/;

function cellOf(id: NodeId, n_layers: number, n_heads: number): { layer: number; head: number } | null {
  const m = HEAD_ID.exec(id);
  if (m === null) return null;
  const layer = Number(m[1]);
  const head = Number(m[2]);
  return layer < n_layers && head < n_heads ? { layer, head } : null;
}

export function makeCellLayout(
  dims: { n_layers: number; n_heads: number },
  box: Pick<DOMRect, 'left' | 'top' | 'width' | 'height'>,
  values: number[][],
  scale: (v: number) => [number, number, number],
): CellLayout {
  const { n_layers, n_heads } = dims;
  const w = n_heads > 0 ? box.width / n_heads : 0;
  const h = n_layers > 0 ? box.height / n_layers : 0;
  return {
    n_layers,
    n_heads,
    cellRect(id) {
      const cell = cellOf(id, n_layers, n_heads);
      return cell === null ? null : { x: box.left + cell.head * w, y: box.top + cell.layer * h, w, h };
    },
    cellColor(id) {
      const cell = cellOf(id, n_layers, n_heads);
      return scale(cell === null ? 0 : (values[cell.layer]?.[cell.head] ?? 0));
    },
  };
}

/** The head under a point in the layout's own coordinates (a pure function, so jsdom can test the pointer). */
export function hitTest(layout: CellLayout, x: number, y: number): NodeId | null {
  const origin = layout.cellRect(headId(0, 0));
  if (origin === null || origin.w <= 0 || origin.h <= 0) return null;
  const head = Math.floor((x - origin.x) / origin.w);
  const layer = Math.floor((y - origin.y) / origin.h);
  if (layer < 0 || head < 0 || layer >= layout.n_layers || head >= layout.n_heads) return null;
  return headId(layer, head);
}
