// Ops: HeadOp and DirectionOp mirror neshter/recipe.py exactly (strict). Other kinds go through registerOpKind;
// anything this version cannot validate survives as an opaque op (kept verbatim, never sent to the server).
import { z } from './zod';

export type Semantics = 'zero' | 'direction' | 'path' | 'read' | 'steer' | 'unknown';

const MAX_FLOAT32 = 3.4028234663852886e38;
const BASE64 = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;

/** Strict base64 (alphabet and padding only, like Python's `b64decode(validate=True)`); `null` when invalid. */
function decodeBase64(s: string): Uint8Array | null {
  if (!BASE64.test(s)) return null;
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

/** recipe.py `DirectionOp._check_vector`: 4 * d_model bytes, finite float32 values, finite non-zero float32 norm. */
function directionVectorOk(op: { d_model: number; vector_b64: string }): boolean {
  const bytes = decodeBase64(op.vector_b64);
  if (bytes === null || bytes.length !== 4 * op.d_model) return false;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let sumSquares = 0;
  for (let i = 0; i < op.d_model; i += 1) {
    const x = view.getFloat32(4 * i, true);
    if (!Number.isFinite(x)) return false;
    sumSquares += Math.fround(x * x); // float32 squares, so underflow and overflow behave like torch's float32 norm
  }
  return sumSquares > 0 && sumSquares <= MAX_FLOAT32;
}

export const HeadOpSchema = z.strictObject({
  type: z.literal('head'),
  layer: z.number().int().nonnegative(),
  head: z.number().int().nonnegative(),
});

export const DirectionOpSchema = z
  .strictObject({
    type: z.literal('direction'),
    d_model: z.number().int().positive(),
    vector_b64: z.string(),
  })
  .refine(directionVectorOk, { message: 'vector_b64 must be 4 * d_model bytes of finite, non-zero float32' });

export type HeadOp = z.infer<typeof HeadOpSchema>;
export type DirectionOp = z.infer<typeof DirectionOpSchema>;
/** An op kind added with `registerOpKind` (Plan 3's explore ops); its shape is whatever its schema accepts. */
export interface ExtensionOp {
  readonly type: string;
  readonly [key: string]: unknown;
}
export type KnownOp = HeadOp | DirectionOp | ExtensionOp;
export type OpaqueOp = { type: string; opaque: true; raw: Record<string, unknown> };
export type AnyOp = KnownOp | OpaqueOp;

export interface OpSpec {
  schema: z.ZodType;
  /** Can go into a saved model / recipe. Explore-only kinds are sendable (compare) but not savable. */
  savable: boolean;
  semantics: Semantics;
}

const registry = new Map<string, OpSpec>([
  ['head', { schema: HeadOpSchema, savable: true, semantics: 'zero' }],
  ['direction', { schema: DirectionOpSchema, savable: true, semantics: 'direction' }],
]);

export function registerOpKind(type: string, spec: OpSpec): void {
  if (type === 'head' || type === 'direction') throw new Error(`The built-in op kind '${type}' cannot be replaced.`);
  registry.set(type, spec);
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);

/** A registered kind that validates becomes a known op; everything else is kept verbatim as an opaque op. */
export function parseOp(raw: unknown): AnyOp {
  const record = isRecord(raw) ? raw : {};
  const type = typeof record['type'] === 'string' ? record['type'] : 'unknown';
  const spec = registry.get(type);
  if (spec && isRecord(raw) && !('opaque' in raw)) {
    const parsed = spec.schema.safeParse(raw);
    if (parsed.success && isRecord(parsed.data) && typeof parsed.data['type'] === 'string') {
      return parsed.data as unknown as KnownOp;
    }
  }
  return { type, opaque: true, raw: record };
}

export const isOpaque = (op: AnyOp): op is OpaqueOp => (op as { opaque?: unknown }).opaque === true;
export const isKnown = (op: AnyOp): op is KnownOp => !isOpaque(op);
export const isSavable = (op: AnyOp): op is KnownOp => isKnown(op) && registry.get(op.type)?.savable === true;
export const isSendable = (op: AnyOp): op is KnownOp => isKnown(op);
export const isHeadOp = (op: AnyOp): op is HeadOp => isKnown(op) && op.type === 'head';
export const isDirectionOp = (op: AnyOp): op is DirectionOp => isKnown(op) && op.type === 'direction';

export function opSemantics(op: AnyOp): Semantics {
  return isKnown(op) ? (registry.get(op.type)?.semantics ?? 'unknown') : 'unknown';
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (typeof value === 'object' && value !== null) {
    const record = value as Record<string, unknown>;
    const body = Object.keys(record)
      .sort()
      .filter((key) => record[key] !== undefined)
      .map((key) => `${JSON.stringify(key)}:${canonical(record[key])}`);
    return `{${body.join(',')}}`;
  }
  return JSON.stringify(value) ?? 'null';
}

/** Canonical JSON with sorted keys; an opaque op is keyed by its verbatim `raw`. */
export const opKey = (op: AnyOp): string => canonical(isOpaque(op) ? op.raw : op);

/** What goes back on the wire: the op itself, or an opaque op's original object. */
export const opToJson = (op: AnyOp): Record<string, unknown> => (isOpaque(op) ? op.raw : { ...op });
