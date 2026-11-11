import { newBasketItem, type BasketItem, type Origin } from '../../basket';
import { parseOp } from '../../ops';

export const NOW = '2026-11-11T10:00:00.000Z';
export const SHA = 'a'.repeat(64);

/** float32 little-endian base64, the wire format of a direction. */
export function vectorB64(values: number[]): string {
  const bytes = new Uint8Array(new Float32Array(values).buffer);
  return btoa(String.fromCharCode(...bytes));
}

export const headOp = (layer: number, head: number) => ({ type: 'head', layer, head });
export const directionOp = (values: number[]) => ({
  type: 'direction',
  d_model: values.length,
  vector_b64: vectorB64(values),
});

export function item(
  raw: unknown,
  extra: { origin?: Origin; concept?: string | null; pendingDiscard?: boolean } = {},
): BasketItem {
  const made = newBasketItem(parseOp(raw), {
    origin: extra.origin ?? 'heatmap',
    concept: extra.concept ?? null,
    addedAt: NOW,
  });
  return extra.pendingDiscard ? { ...made, pendingDiscard: true } : made;
}

export const head = (layer: number, h: number, extra?: Parameters<typeof item>[1]) => item(headOp(layer, h), extra);

/** Narrowing for tests: fails loudly instead of using a non-null assertion. */
export function must<T>(value: T | null | undefined): T {
  if (value === null || value === undefined) throw new Error('expected a value');
  return value;
}
