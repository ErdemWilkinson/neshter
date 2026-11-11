import { useCallback, type KeyboardEvent } from 'react';
import { useStore } from 'zustand';
import { headId, parseNodeId, type NodeId } from '@/core/ids';
import { selectionStore } from '@/core/stores/selection';
import { t } from '@/i18n';
import { announce } from '@/ui/LiveRegion';
import { formatPower } from './format';

export interface Dims {
  n_layers: number;
  n_heads: number;
}
export interface Cell {
  layer: number;
  head: number;
}

const clamp = (n: number, max: number) => Math.min(Math.max(n, 0), max);

/** The focused head, pulled into the grid (a stale or non-head focus id falls back to K0 H0); `null` for an empty grid. */
export function focusCell(focus: NodeId | null, dims: Dims): Cell | null {
  if (dims.n_layers < 1 || dims.n_heads < 1) return null;
  const parsed = focus === null ? null : parseNodeId(focus);
  const cell = parsed?.kind === 'head' ? { layer: parsed.layer, head: parsed.head } : { layer: 0, head: 0 };
  return { layer: clamp(cell.layer, dims.n_layers - 1), head: clamp(cell.head, dims.n_heads - 1) };
}

/** Arrows move one cell, Home/End jump to the row ends; every move stays inside the grid. */
export function nextCell(key: string, from: Cell, dims: Dims): Cell | null {
  const lastLayer = dims.n_layers - 1;
  const lastHead = dims.n_heads - 1;
  switch (key) {
    case 'ArrowUp':
      return { layer: clamp(from.layer - 1, lastLayer), head: from.head };
    case 'ArrowDown':
      return { layer: clamp(from.layer + 1, lastLayer), head: from.head };
    case 'ArrowLeft':
      return { layer: from.layer, head: clamp(from.head - 1, lastHead) };
    case 'ArrowRight':
      return { layer: from.layer, head: clamp(from.head + 1, lastHead) };
    case 'Home':
      return { layer: from.layer, head: 0 };
    case 'End':
      return { layer: from.layer, head: lastHead };
    default:
      return null;
  }
}

/** `K{L} · H{H} · güç {v} · seçili|seçili değil` */
export function describeCell(cell: Cell, values: number[][], selected: boolean): string {
  return t('map.announce', {
    layer: cell.layer,
    head: cell.head,
    value: formatPower(values[cell.layer]?.[cell.head] ?? 0),
    state: t(selected ? 'map.selected' : 'map.notSelected'),
  });
}

/** Keyboard model of the heatmap: one focused cell (kept in `selectionStore.focus`), a live-region readout per move. */
export function useHeatmapKeys(dims: Dims, values: number[][]) {
  const focus = useStore(selectionStore, (s) => s.focus);
  const { n_layers, n_heads } = dims;
  const current = focusCell(focus, { n_layers, n_heads });

  const say = useCallback(
    (cell: Cell) =>
      announce(describeCell(cell, values, selectionStore.getState().selected.has(headId(cell.layer, cell.head)))),
    [values],
  );

  const focusOn = useCallback(
    (cell: Cell) => {
      selectionStore.getState().setFocus(headId(cell.layer, cell.head));
      say(cell);
    },
    [say],
  );

  const toggle = useCallback(
    (cell: Cell) => {
      const store = selectionStore.getState();
      store.setFocus(headId(cell.layer, cell.head));
      store.toggle(headId(cell.layer, cell.head));
      say(cell);
    },
    [say],
  );

  const onKeyDown = useCallback(
    (event: KeyboardEvent<HTMLElement>) => {
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      const from = focusCell(selectionStore.getState().focus, { n_layers, n_heads });
      if (from === null) return;
      if (event.key === ' ' || event.key === 'Enter') {
        event.preventDefault();
        toggle(from);
        return;
      }
      const to = nextCell(event.key, from, { n_layers, n_heads });
      if (to === null) return;
      event.preventDefault();
      focusOn(to);
    },
    [n_layers, n_heads, toggle, focusOn],
  );

  /** On entering the grid the cell that holds the focus is read out (K0 H0 the first time). */
  const onFocus = useCallback(() => {
    const from = focusCell(selectionStore.getState().focus, { n_layers, n_heads });
    if (from !== null) focusOn(from);
  }, [n_layers, n_heads, focusOn]);

  return { current, onKeyDown, onFocus, focusOn, toggle };
}
