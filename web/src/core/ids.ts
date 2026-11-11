// Canonical node and edge ids shared with the 3D view: h:{layer}:{head}, n:{layer}:{index}, t:{pos},
// tid:{token_id}, out, e:{edge_id}. A heatmap cell IS node h:L:H. Numbers are canonical: no sign, no leading zeros.

export type NodeId = string & { readonly __brand: 'NodeId' };
export type EdgeId = string & { readonly __brand: 'EdgeId' };

export type ParsedNode =
  | { kind: 'head'; layer: number; head: number }
  | { kind: 'neuron'; layer: number; index: number }
  | { kind: 'tpos'; pos: number }
  | { kind: 'tid'; id: number }
  | { kind: 'out' };

export const OUT_ID = 'out' as NodeId;

function index(name: string, value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`${name} must be a non-negative integer, got ${value}`);
  return value;
}

export const headId = (layer: number, head: number): NodeId =>
  `h:${index('layer', layer)}:${index('head', head)}` as NodeId;
export const neuronId = (layer: number, idx: number): NodeId =>
  `n:${index('layer', layer)}:${index('index', idx)}` as NodeId;
export const tokenPosId = (pos: number): NodeId => `t:${index('pos', pos)}` as NodeId;
export const tokenIdId = (id: number): NodeId => `tid:${index('id', id)}` as NodeId;
export const edgeId = (id: string): EdgeId => `e:${id}` as EdgeId;

const UINT = /^(?:0|[1-9]\d*)$/;
function uint(s: string | undefined): number | null {
  if (s === undefined || !UINT.test(s)) return null;
  const n = Number(s);
  return Number.isSafeInteger(n) ? n : null;
}

/** Parses a node id; anything that `headId` & co. would not have produced (`h:01:2`, `h:-1:2`, `h:1`) is `null`. */
export function parseNodeId(s: string): ParsedNode | null {
  if (typeof s !== 'string') return null;
  if (s === OUT_ID) return { kind: 'out' };
  const [tag, a, b, ...rest] = s.split(':');
  if (rest.length > 0) return null;
  if ((tag === 'h' || tag === 'n') && b !== undefined) {
    const layer = uint(a);
    const second = uint(b);
    if (layer === null || second === null) return null;
    return tag === 'h' ? { kind: 'head', layer, head: second } : { kind: 'neuron', layer, index: second };
  }
  if ((tag === 't' || tag === 'tid') && b === undefined) {
    const value = uint(a);
    if (value === null) return null;
    return tag === 't' ? { kind: 'tpos', pos: value } : { kind: 'tid', id: value };
  }
  return null;
}
