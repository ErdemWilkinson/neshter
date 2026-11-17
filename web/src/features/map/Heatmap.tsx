import { useCallback, useEffect, useImperativeHandle, useRef, useState, type Ref } from 'react';
import { useStore } from 'zustand';
import { headId, type NodeId } from '@/core/ids';
import { selectionStore } from '@/core/stores/selection';
import { useT } from '@/i18n';
import { makeCellLayout, hitTest, type CellLayout } from './cellLayout';
import { heatColor } from './colorScale';
import { drawHeatmap } from './drawHeatmap';
import { focusCell, useHeatmapKeys, type Dims } from './useHeatmapKeys';

export interface HeatmapHandle {
  /** Cell rects in viewport px, from the canvas's current bounding box. */
  getCellLayout(): CellLayout;
}

const ROW_PX = 22;
const MIN_HEIGHT = 120;
const MAX_HEIGHT = 860;
/** Rows grow with the width (up to this many px), so wide screens get chunky tiles instead of thin stripes. */
const MAX_ROW_PX = 40;
/** A tile is about this many times as wide as it is tall once the rows have room to grow. */
const TILE_ASPECT = 3;

export function Heatmap({
  dims,
  values,
  basket,
  describedBy,
  ref,
}: {
  dims: Dims;
  values: number[][];
  basket: ReadonlySet<NodeId>;
  /** Id of the legend element. */
  describedBy: string;
  ref?: Ref<HeatmapHandle>;
}) {
  const t = useT();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const selected = useStore(selectionStore, (s) => s.selected);
  const [hasFocus, setHasFocus] = useState(false);
  const [width, setWidth] = useState(0);
  const keys = useHeatmapKeys(dims, values);
  const focusId = keys.current === null ? null : headId(keys.current.layer, keys.current.head);
  const tileW = width > 0 ? width / Math.max(1, dims.n_heads) : 0;
  const rowPx = Math.min(MAX_ROW_PX, Math.max(ROW_PX, tileW / TILE_ASPECT));
  const height = Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, dims.n_layers * rowPx));

  const scale = heatColor;

  useImperativeHandle(
    ref,
    () => ({
      getCellLayout: () => {
        const box = canvasRef.current?.getBoundingClientRect() ?? { left: 0, top: 0, width: 0, height: 0 };
        return makeCellLayout(dims, box, values, scale);
      },
    }),
    [dims, values, scale],
  );

  // Draw. jsdom has no canvas, so a missing context is "nothing to paint", not an error.
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d') ?? null;
    if (canvas === null || ctx === null) return;
    const cssWidth = width || canvas.clientWidth || canvas.width;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(cssWidth * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const layout = makeCellLayout(dims, { left: 0, top: 0, width: cssWidth, height }, values, scale);
    drawHeatmap(ctx, layout, values, { selected, basket, focus: hasFocus ? focusId : null });
  }, [dims, values, scale, selected, basket, focusId, hasFocus, width, height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => setWidth(canvas.clientWidth));
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  const onClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (canvas === null) return;
      const box = canvas.getBoundingClientRect();
      const layout = makeCellLayout(dims, box, values, scale);
      const id = hitTest(layout, event.clientX, event.clientY);
      const cell = id === null ? null : focusCell(id, dims);
      if (cell !== null) keys.toggle(cell);
    },
    [dims, values, scale, keys],
  );

  // Axis labels and numbers are decoration for sighted users (the canvas has its own keyboard and spoken model).
  const headStep = dims.n_heads > 24 ? 4 : 1;
  const layerStep = height / Math.max(1, dims.n_layers) >= 12 ? 1 : 5;
  const heads = Array.from({ length: dims.n_heads }, (_, i) => i);
  const layers = Array.from({ length: dims.n_layers }, (_, i) => i);

  return (
    <div className="heatmap-frame">
      <span className="heatmap-axis-x" aria-hidden="true">
        {t('map.axis.head')} →
      </span>
      <div className="heatmap-xticks num" aria-hidden="true" style={{ gridTemplateColumns: `repeat(${dims.n_heads}, 1fr)` }}>
        {heads.map((i) => (
          <span key={i}>{i % headStep === 0 ? i : ''}</span>
        ))}
      </div>
      <span className="heatmap-axis-y" aria-hidden="true">
        {t('map.axis.layer')}
      </span>
      <div
        className="heatmap-yticks num"
        aria-hidden="true"
        style={{ height, gridTemplateRows: `repeat(${dims.n_layers}, 1fr)` }}
      >
        {layers.map((i) => (
          <span key={i}>{i % layerStep === 0 ? i : ''}</span>
        ))}
      </div>
      {/* The canvas is one composite widget with its own keyboard model (arrows, Home/End, Space/Enter). */}
      <canvas
        ref={canvasRef}
        className="heatmap-canvas"
        style={{ height }}
        tabIndex={0}
        // eslint-disable-next-line jsx-a11y/no-interactive-element-to-noninteractive-role -- the spec fixes role=group on this focusable canvas
        role="group"
        aria-roledescription={t('map.heatmap.role')}
        aria-label={t('map.heatmap.label')}
        aria-describedby={describedBy}
        onKeyDown={keys.onKeyDown}
        onFocus={() => {
          setHasFocus(true);
          keys.onFocus();
        }}
        onBlur={() => setHasFocus(false)}
        onClick={onClick}
      />
    </div>
  );
}
