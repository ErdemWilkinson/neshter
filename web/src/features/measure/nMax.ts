export interface NMaxBounds {
  min: 19;
  max: 199 | 999;
  default: 39;
}

/** The random-control draw budget: 19..999, and the server caps it at 199 on CPU (422 otherwise). */
export function nMaxBounds(device: string): NMaxBounds {
  return { min: 19, max: device === 'cpu' ? 199 : 999, default: 39 };
}

/** Whole number inside the bounds; text that is not a number falls back to the default. */
export function clampNMax(text: string, bounds: NMaxBounds): number {
  const n = Math.round(Number(text));
  if (text.trim() === '' || !Number.isFinite(n)) return bounds.default;
  return Math.min(bounds.max, Math.max(bounds.min, n));
}

/** The smallest p-value a permutation test with `nMax` draws can reach: 1 / (nMax + 1). */
export const smallestP = (nMax: number): number => 1 / (nMax + 1);
