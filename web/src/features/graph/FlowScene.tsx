import { useEffect, useMemo, useRef } from 'react';
import type { GraphEdge } from '@/api/schemas';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { approach, easeOut, tiltMix, PITCH_MAX, PITCH_MIN, VIEWS, ZOOM_MAX, ZOOM_MIN, type Camera } from './camera';
import { drawFlow, hitTest, readFlowPalette, type Hits, type Hover, type Pick, type SceneData, type SceneEdge } from './drawFlow';
import { COL, edgeCurve, extent, type Grid, type PlacedNode } from './layout';

/** `heat`: the heatmap from above; `towers`: the same heatmap tilted, each head a tower (no graph needed); `flow`: the graph. */
export type SceneMode = 'heat' | 'towers' | 'flow';

/** Where the camera goes for a mode; every mode but the flat heatmap can be turned and zoomed. */
const viewFor = (mode: SceneMode): Camera => (mode === 'heat' ? VIEWS.heat : VIEWS.tilt);
const turnable = (mode: SceneMode): boolean => mode !== 'heat';

export interface FlowSceneProps {
  grid: Grid;
  nodes: readonly PlacedNode[];
  /** Visible edges, strongest first. */
  edges: readonly GraphEdge[];
  /** Drawn weight per edge after cuts (see `liveWeights`). */
  live: ReadonlyMap<string, number>;
  maxWeight: number;
  trunk: readonly number[];
  heat: readonly (readonly number[])[] | null;
  mode: SceneMode;
  cut: ReadonlySet<string>;
  basket: ReadonlySet<string>;
  chosen: ReadonlySet<string>;
  selected: Pick | null;
  labels: ReadonlySet<string>;
  reducedMotion: boolean;
  name(id: string): string;
  onSelect(pick: Pick | null): void;
  onHover(hover: Hover | null): void;
}

interface Anim {
  cam: Camera;
  target: Camera;
  mix: number;
  /** 0 = flat … 1 = towers (see `FrameState.rise`). */
  rise: number;
  widths: Map<string, number>;
  cutAt: Map<string, number>;
  basketAt: Map<string, number>;
  hits: Hits;
  hover: Pick | null;
  /** Where the tooltip was last reported, so a still pointer does not re-render the panel. */
  hoverAt: { x: number; y: number } | null;
  drag: { x: number; y: number; moved: boolean } | null;
  /** Draw until this time even when nothing moves (props changed, a cut snapped, a head closed). */
  awakeUntil: number;
  /** False until the first cut/basket sync: what is already there when the scene opens does not animate. */
  primed: boolean;
}

const now = () => performance.now() / 1000;
/** How long a cut or basket animation runs (s); the scene keeps drawing that long after a change. */
const SETTLE = 1.6;
const settled = (a: number, b: number) => Math.abs(a - b) < 1e-4;
const samePick = (a: Pick | null, b: Pick | null) => a?.kind === b?.kind && a?.id === b?.id;

/** The ids lit by a pick: both ends of an edge, or the node / cell itself. */
function litBy(pick: Pick | null, edges: readonly GraphEdge[], out: Set<string>, rivers: Set<string>) {
  if (pick === null) return;
  if (pick.kind === 'edge') {
    const e = edges.find((x) => x.id === pick.id);
    rivers.add(pick.id);
    if (e !== undefined) out.add(e.src).add(e.dst);
    return;
  }
  out.add(pick.id);
  for (const e of edges) if (e.src === pick.id || e.dst === pick.id) rivers.add(e.id);
}

/**
 * The heatmap and the connection graph as one canvas scene (no WebGL): seen from above the floor is the heatmap;
 * switching to the graph tilts the camera, the cells of graph heads shrink into their rings and the rivers grow
 * from the words toward the answer. Cuts snap, heads put in the basket close in on themselves.
 */
export default function FlowScene(props: FlowSceneProps) {
  const t = useT();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const latest = useRef(props);
  const anim = useRef<Anim | null>(null);
  const { grid, nodes, edges, trunk, heat, maxWeight, mode, cut, basket, reducedMotion } = props;

  const scene = useMemo((): SceneData => {
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const depth = Math.max(1, grid.nLayers);
    const sceneEdges: SceneEdge[] = [];
    edges.forEach((e, i) => {
      const a = byId.get(e.src);
      const b = byId.get(e.dst);
      if (a === undefined || b === undefined) return;
      sceneEdges.push({ ...e, pts: edgeCurve(a.pos, b.pos, e.weight / maxWeight, i), from: a.kind === 'token' ? 0 : (a.layer + 1) / depth });
    });
    sceneEdges.reverse(); // weakest first, so strong rivers lie on top
    return { grid, nodes, edges: sceneEdges, trunk, heat, maxWeight };
  }, [grid, nodes, edges, trunk, heat, maxWeight]);
  // What the graph view must fit: the floor, widened to the token row and the answer when they reach further.
  const worlds = useMemo(() => {
    const floor = extent(grid);
    let half = floor.w / 2;
    for (const n of nodes) half = Math.max(half, Math.abs(n.pos[0]) + COL * 0.6);
    return { heat: floor, flow: { w: 2 * half, d: floor.d } };
  }, [grid, nodes]);
  const latestWorlds = useRef(worlds);

  const latestScene = useRef(scene);
  const latestT = useRef(t);
  useEffect(() => {
    latest.current = props;
    latestT.current = t;
    latestScene.current = scene;
    latestWorlds.current = worlds;
    if (anim.current !== null) anim.current.awakeUntil = now() + SETTLE;
  });

  // Camera target follows the mode; animation clocks start when a cut or a basket head appears.
  useEffect(() => {
    const a = (anim.current ??= {
      cam: { ...viewFor(mode) },
      target: { ...viewFor(mode) },
      mix: mode === 'flow' ? 1 : 0,
      rise: mode === 'towers' ? 1 : 0,
      widths: new Map(),
      cutAt: new Map(),
      basketAt: new Map(),
      hits: { cells: [], nodes: [], edges: [] },
      hover: null,
      hoverAt: null,
      drag: null,
      awakeUntil: 0,
      primed: false,
    });
    a.awakeUntil = now() + SETTLE;
    a.target = { ...viewFor(mode) };
  }, [mode]);
  useEffect(() => {
    const a = anim.current;
    if (a === null) return;
    const stamp = reducedMotion || !a.primed ? -9 : now();
    a.primed = true;
    for (const id of cut) if (!a.cutAt.has(id)) a.cutAt.set(id, stamp);
    for (const id of [...a.cutAt.keys()]) if (!cut.has(id)) a.cutAt.delete(id);
    for (const id of basket) if (!a.basketAt.has(id)) a.basketAt.set(id, stamp);
    for (const id of [...a.basketAt.keys()]) if (!basket.has(id)) a.basketAt.delete(id);
    a.awakeUntil = now() + SETTLE;
  }, [cut, basket, reducedMotion]);

  // The frame loop. jsdom has no canvas context: then nothing is drawn and nothing can be hit.
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d') ?? null;
    if (canvas === null || ctx === null) return;
    const palette = readFlowPalette();
    let frame = 0;
    const size = { w: 0, h: 0, dpr: 1 };
    let onScreen = true;
    const measure = () => {
      if (anim.current !== null) anim.current.awakeUntil = now() + SETTLE;
      size.w = canvas.clientWidth;
      size.h = canvas.clientHeight;
      size.dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.max(1, Math.round(size.w * size.dpr));
      canvas.height = Math.max(1, Math.round(size.h * size.dpr));
    };
    measure();
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(measure);
    observer?.observe(canvas);
    const seen =
      typeof IntersectionObserver === 'undefined'
        ? null
        : new IntersectionObserver(([entry]) => {
            onScreen = entry?.isIntersecting ?? true;
            if (onScreen && anim.current !== null) anim.current.awakeUntil = now() + SETTLE;
          });
    seen?.observe(canvas);

    const tick = () => {
      const a = anim.current;
      const p = latest.current;
      // Off screen, or a still picture (heatmap, or reduced motion) that has settled: skip the frame.
      const still =
        a !== null &&
        (p.mode !== 'flow' || p.reducedMotion) &&
        now() > a.awakeUntil &&
        settled(a.cam.yaw, a.target.yaw) &&
        settled(a.cam.pitch, a.target.pitch) &&
        settled(a.cam.zoom, a.target.zoom) &&
        settled(a.mix, p.mode === 'flow' ? 1 : 0) &&
        settled(a.rise, p.mode === 'towers' ? 1 : 0);
      if (a !== null && size.w > 0 && onScreen && !still) {
        const rate = p.reducedMotion ? 1 : 0.085;
        a.cam = { yaw: approach(a.cam.yaw, a.target.yaw, rate), pitch: approach(a.cam.pitch, a.target.pitch, rate), zoom: approach(a.cam.zoom, a.target.zoom, rate) };
        a.mix = approach(a.mix, p.mode === 'flow' ? 1 : 0, p.reducedMotion ? 1 : 0.05);
        a.rise = approach(a.rise, p.mode === 'towers' ? 1 : 0, p.reducedMotion ? 1 : 0.06);
        for (const e of p.edges) {
          const target = Math.abs(p.live.get(e.id) ?? e.weight);
          a.widths.set(e.id, approach(a.widths.get(e.id) ?? Math.abs(e.weight), target, p.reducedMotion ? 1 : 0.05));
        }
        const focus = new Set<string>();
        const highlight = new Set<string>();
        litBy(p.selected, p.edges, focus, highlight);
        litBy(a.hover, p.edges, focus, highlight);
        ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
        a.hits = drawFlow(ctx, latestScene.current, {
          t: now(),
          mix: a.mix * tiltMix(a.cam.pitch),
          rise: a.rise,
          cam: a.cam,
          vp: {
            w: size.w,
            h: size.h,
            world: (() => {
              const { heat: a0, flow: b0 } = latestWorlds.current;
              const m = easeOut(a.mix);
              return { w: a0.w + (b0.w - a0.w) * m, d: a0.d + (b0.d - a0.d) * m };
            })(),
          },
          widths: a.widths,
          cut: p.cut,
          cutAt: a.cutAt,
          basket: p.basket,
          basketAt: a.basketAt,
          chosen: p.chosen,
          focus,
          highlight,
          labels: p.labels,
          reduced: p.reducedMotion,
          palette,
          name: p.name,
          layerName: (l) => latestT.current('layer.id', { layer: l }),
        });
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);

    const onWheel = (event: WheelEvent) => {
      const a = anim.current;
      if (a === null || !turnable(latest.current.mode) || event.deltaY === 0) return;
      event.preventDefault();
      a.target = { ...a.target, zoom: Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, a.target.zoom * (event.deltaY < 0 ? 1.1 : 0.91))) };
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      cancelAnimationFrame(frame);
      observer?.disconnect();
      seen?.disconnect();
      canvas.removeEventListener('wheel', onWheel);
    };
  }, []);

  const local = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - box.left, y: event.clientY - box.top };
  };
  const setView = (view: Camera) => {
    if (anim.current !== null) anim.current.target = { ...view };
  };
  const zoomBy = (factor: number) => {
    const a = anim.current;
    if (a !== null) a.target = { ...a.target, zoom: Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, a.target.zoom * factor)) };
  };

  return (
    <>
      <canvas
        ref={canvasRef}
        className={`flow-canvas${turnable(mode) ? ' is-flow' : ''}`}
        aria-hidden="true"
        onPointerDown={(event) => {
          const a = anim.current;
          if (a === null) return;
          a.drag = { x: event.clientX, y: event.clientY, moved: false };
          // Keep the drag when the pointer runs over the buttons or off the canvas.
          event.currentTarget.setPointerCapture?.(event.pointerId);
        }}
        onPointerMove={(event) => {
          const a = anim.current;
          if (a === null) return;
          if (a.drag !== null) {
            const dx = event.clientX - a.drag.x;
            const dy = event.clientY - a.drag.y;
            if (a.drag.moved || Math.hypot(dx, dy) > 5) {
              a.drag = { x: event.clientX, y: event.clientY, moved: true };
              // The heatmap is seen flat from above: a drag there only stops being a click.
              if (turnable(latest.current.mode))
                a.target = { ...a.target, yaw: a.target.yaw + dx * 0.006, pitch: Math.min(PITCH_MAX, Math.max(PITCH_MIN, a.target.pitch + dy * 0.005)) };
              if (a.hover !== null) {
                a.hover = null;
                latest.current.onHover(null);
              }
              return;
            }
          }
          const busy = a.mix > 0.02 && a.mix < 0.98;
          const { x, y } = local(event);
          const pick = busy ? null : hitTest(a.hits, x, y);
          const far = a.hoverAt === null || Math.hypot(x - a.hoverAt.x, y - a.hoverAt.y) > 3;
          if (!samePick(pick, a.hover) || (pick !== null && far)) {
            latest.current.onHover(pick === null ? null : { pick, x, y });
            a.hoverAt = pick === null ? null : { x, y };
            a.awakeUntil = now() + SETTLE;
          }
          a.hover = pick;
        }}
        onPointerUp={(event) => {
          const a = anim.current;
          if (a === null) return;
          const drag = a.drag;
          a.drag = null;
          // Only a press that started here and did not move is a click; nothing is hit while the scene turns.
          if (drag === null || drag.moved || (a.mix > 0.02 && a.mix < 0.98)) return;
          const { x, y } = local(event);
          latest.current.onSelect(hitTest(a.hits, x, y));
        }}
        onPointerCancel={() => {
          if (anim.current !== null) anim.current.drag = null;
        }}
        onPointerLeave={() => {
          const a = anim.current;
          if (a === null) return;
          if (a.hover !== null) {
            a.hover = null;
            a.hoverAt = null;
            latest.current.onHover(null);
          }
        }}
      />
      {turnable(mode) && (
        <div className="flow-hud">
          <Button onClick={() => setView(VIEWS.tilt)}>{t('graph.view.tilt')}</Button>
          <Button onClick={() => setView(VIEWS.low)}>{t('graph.view.low')}</Button>
          <Button aria-label={t('graph.view.zoomIn')} onClick={() => zoomBy(1.15)}>
            +
          </Button>
          <Button aria-label={t('graph.view.zoomOut')} onClick={() => zoomBy(1 / 1.15)}>
            −
          </Button>
        </div>
      )}
    </>
  );
}
