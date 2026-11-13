import type { GraphEdge, GraphNode, GraphResult } from '@/api/schemas';

/**
 * One world for both views. The floor is the heatmap: heads sit on their heatmap column (x) and layer row (z),
 * layer 0 farthest away. Seen from straight above it *is* the heatmap; tilted, it is the connection graph. Tokens
 * lie beyond the first row, a layer's MLP just after its attention row, the answer in front of the last row.
 */
export const COL = 120;
export const ROW = 100;
export const CURVE_STEPS = 32;

export type Vec3 = readonly [number, number, number];

export interface PlacedNode extends GraphNode {
  pos: Vec3;
}

export interface Grid {
  nLayers: number;
  nHeads: number;
}

export const headX = (head: number, nHeads: number): number => (head - (nHeads - 1) / 2) * COL;
export const layerZ = (layer: number, nLayers: number): number => ((nLayers - 1) / 2 - layer) * ROW;

/** World size of the floor (heatmap area plus the token and answer rows). */
export function extent({ nLayers, nHeads }: Grid): { w: number; d: number } {
  return { w: Math.max(1, nHeads) * COL, d: (Math.max(1, nLayers) + 2.6) * ROW };
}

export function placeNodes(graph: Pick<GraphResult, 'nodes' | 'n_layers' | 'n_heads' | 'tokens'>): PlacedNode[] {
  const { n_layers: L, n_heads: H } = graph;
  const T = graph.tokens.length;
  const neuronsByLayer = new Map<number, GraphNode[]>();
  for (const n of graph.nodes) {
    if (n.kind !== 'neuron') continue;
    const list = neuronsByLayer.get(n.layer) ?? [];
    list.push(n);
    neuronsByLayer.set(n.layer, list);
  }
  for (const list of neuronsByLayer.values()) list.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
  const width = Math.max(1, H - 1) * COL;
  const tokenSpan = Math.max(width, 1.3 * COL * (T - 1));
  return graph.nodes.map((n): PlacedNode => {
    switch (n.kind) {
      case 'token': {
        const x = T <= 1 ? 0 : -tokenSpan / 2 + (tokenSpan * n.index) / (T - 1);
        return { ...n, pos: [x, 0, layerZ(0, L) + 1.3 * ROW] };
      }
      case 'head':
        return { ...n, pos: [headX(n.index, H), 0, layerZ(n.layer, L)] };
      case 'neuron': {
        const list = neuronsByLayer.get(n.layer) ?? [];
        const rank = list.indexOf(n);
        const spread = Math.min(0.7 * COL, width / Math.max(1, list.length));
        return { ...n, pos: [(rank - (list.length - 1) / 2) * spread, 0, layerZ(n.layer, L) - ROW / 2] };
      }
      default:
        return { ...n, pos: [0, 0, layerZ(L - 1, L) - 1.3 * ROW] };
    }
  });
}

/** A river: a cubic arc lifted off the floor (stronger edges arch higher), sampled at CURVE_STEPS + 1 points. The
 * side it bows to alternates by edge so parallel rivers do not lie on top of each other. */
export function edgeCurve(a: Vec3, b: Vec3, weight: number, side: number): Vec3[] {
  const dz = b[2] - a[2];
  const wob = (side % 2 === 0 ? -1 : 1) * (20 + ((side * 37) % 50));
  const lift = 18 + Math.min(1, Math.abs(weight)) * 60;
  const c1: Vec3 = [a[0] + wob * 0.4, lift, a[2] + dz * 0.35];
  const c2: Vec3 = [b[0] - wob * 0.6, lift, a[2] + dz * 0.7];
  return Array.from({ length: CURVE_STEPS + 1 }, (_, i): Vec3 => {
    const t = i / CURVE_STEPS;
    const u = 1 - t;
    const f = (k: 0 | 1 | 2) => u * u * u * a[k] + 3 * u * u * t * c1[k] + 3 * u * t * t * c2[k] + t * t * t * b[k];
    return [f(0), f(1), f(2)];
  });
}

/** Order of computation: tokens, then per layer its attention then its MLP, the answer last. */
const step = (n: GraphNode): number =>
  n.kind === 'token' ? -1 : n.kind === 'head' ? 2 * n.layer : n.kind === 'neuron' ? 2 * n.layer + 1 : Infinity;

/**
 * What still flows, as drawn (an estimate from the graph, not a measurement): a node keeps the share of its
 * incoming |e| that is not cut, and passes that share on to every edge it sends. Returns each edge's drawn weight.
 */
export function liveWeights(nodes: readonly GraphNode[], edges: readonly GraphEdge[], cut: ReadonlySet<string>): Map<string, number> {
  const incoming = new Map<string, GraphEdge[]>();
  for (const e of edges) {
    const list = incoming.get(e.dst) ?? [];
    list.push(e);
    incoming.set(e.dst, list);
  }
  const ratio = new Map<string, number>();
  const live = new Map<string, number>();
  for (const n of [...nodes].sort((a, b) => step(a) - step(b))) {
    let all = 0;
    let kept = 0;
    for (const e of incoming.get(n.id) ?? []) {
      const v = cut.has(e.id) ? 0 : e.weight * (ratio.get(e.src) ?? 1);
      live.set(e.id, v);
      all += Math.abs(e.weight);
      kept += Math.abs(v);
    }
    ratio.set(n.id, all > 0 ? kept / all : 1);
  }
  for (const e of edges) if (!live.has(e.id)) live.set(e.id, cut.has(e.id) ? 0 : e.weight);
  return live;
}

/** Layer-boundary width 0..1 of the residual stream toward the answer, scaled so the largest positive is 1. */
export function trunkWidths(trunk: readonly number[]): number[] {
  const top = Math.max(0, ...trunk);
  return trunk.map((v) => (top > 0 ? Math.max(0, v) / top : 0));
}

/** The strongest `limit` edges, plus every cut edge (a cut never disappears from view). */
export function visibleEdges(edges: readonly GraphEdge[], limit: number, cut: ReadonlySet<string>): GraphEdge[] {
  return edges.filter((e, i) => i < limit || cut.has(e.id));
}

/** How much of what reaches `edge.dst` (by |e| over the graph's incoming edges in the same channel) this edge carries. */
export function incomingShare(edges: readonly GraphEdge[], edge: GraphEdge): number {
  let total = 0;
  for (const e of edges) if (e.dst === edge.dst && e.channel === edge.channel) total += Math.abs(e.weight);
  return total > 0 ? Math.abs(edge.weight) / total : 0;
}
