// The random-control statistics of a "real effect" measurement, ported from neshter/metrics.py for the phone,
// where the model runs on the device and there is no Python. Same rules, same order of draws; the random numbers
// come from a small seeded generator instead of Python's `random`, so a seed gives the same controls on every
// phone, not the same controls as the desktop.

export type Head = readonly [layer: number, head: number];

/** mulberry32: a tiny, fast seeded generator; good enough to pick random heads. Returns 0 <= x < 1. */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** k distinct items of `pool`, in random order (Python's `random.sample`). */
export function sample<T>(pool: readonly T[], k: number, rnd: () => number): T[] {
  if (k > pool.length) throw new Error(`sample: ${k} > ${pool.length}`);
  const copy = [...pool];
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(rnd() * (copy.length - i));
    [copy[i], copy[j]] = [copy[j] as T, copy[i] as T];
  }
  return copy.slice(0, k);
}

/**
 * For every layer, as many unselected heads of that same layer as were selected there. A layer whose pool runs out
 * is topped up from the nearest layers (ties: the lower layer first) and `matched` is false.
 */
export function layerMatchedHeads(
  heads: readonly Head[],
  nLayers: number,
  nHeads: number,
  rnd: () => number,
): { picked: Head[]; matched: boolean } {
  const chosen = new Set(heads.map(([l, h]) => `${l}:${h}`));
  const pools = new Map<number, Head[]>();
  for (let l = 0; l < nLayers; l++) {
    pools.set(
      l,
      Array.from({ length: nHeads }, (_, h): Head => [l, h]).filter(([, h]) => !chosen.has(`${l}:${h}`)),
    );
  }
  const total = [...pools.values()].reduce((s, p) => s + p.length, 0);
  if (total < heads.length) throw new Error('Rastgele kontrol için yeterli seçilmemiş head yok.');
  const counts = new Map<number, number>();
  for (const [l] of heads) counts.set(l, (counts.get(l) ?? 0) + 1);

  const picked: Head[] = [];
  const deficits = new Map<number, number>();
  for (const layer of [...counts.keys()].sort((a, b) => a - b)) {
    const pool = pools.get(layer) ?? [];
    const want = counts.get(layer) ?? 0;
    const take = Math.min(want, pool.length);
    const got = sample(pool, take, rnd);
    pools.set(layer, pool.filter((x) => !got.includes(x)));
    picked.push(...got);
    if (take < want) deficits.set(layer, want - take);
  }
  for (const layer of [...deficits.keys()].sort((a, b) => a - b)) {
    let need = deficits.get(layer) ?? 0;
    const others = Array.from({ length: nLayers }, (_, i) => i).sort(
      (a, b) => Math.abs(a - layer) - Math.abs(b - layer) || a - b,
    );
    for (const other of others) {
      if (need === 0) break;
      const pool = pools.get(other) ?? [];
      if (other === layer || pool.length === 0) continue;
      const got = sample(pool, Math.min(need, pool.length), rnd);
      pools.set(other, pool.filter((x) => !got.includes(x)));
      picked.push(...got);
      need -= got.length;
    }
  }
  return { picked, matched: deficits.size === 0 };
}

/** The largest k with (1 + k)/(1 + nMax) <= tau (-1 if none). */
export const futilityK = (nMax: number, tau = 0.05): number => Math.floor(tau * (nMax + 1) + 1e-9) - 1;

export interface PermutationOutcome {
  nDrawn: number;
  kExceed: number;
  pValue: number;
  pKind: 'full' | 'besag_clifford';
  significant: boolean;
  stoppedEarly: boolean;
  seen: number[];
}

/**
 * Draws one control at a time and stops as soon as the verdict can no longer change (futility). The verdict is the
 * one all nMax draws would give; ties count as exceeding (conservative).
 */
export async function sequentialPermutation(
  target: number,
  draw: (i: number) => Promise<number>,
  nMax: number,
  tau = 0.05,
): Promise<PermutationOutcome> {
  const kMax = futilityK(nMax, tau);
  if (kMax < 0) throw new Error(`${nMax} çekişle p en az ${(1 / (nMax + 1)).toPrecision(4)} olur; τ = ${tau} hiç sağlanamaz.`);
  const seen: number[] = [];
  let k = 0;
  while (seen.length < nMax) {
    const value = await draw(seen.length);
    seen.push(value);
    if (value >= target) k += 1;
    if (k > kMax && seen.length < nMax) {
      return { nDrawn: seen.length, kExceed: k, pValue: k / seen.length, pKind: 'besag_clifford', significant: false, stoppedEarly: true, seen };
    }
  }
  return { nDrawn: nMax, kExceed: k, pValue: (1 + k) / (1 + nMax), pKind: 'full', significant: k <= kMax, stoppedEarly: false, seen };
}

/** numpy.percentile with the default linear interpolation. */
export function percentile(values: readonly number[], q: number): number {
  const s = [...values].sort((a, b) => a - b);
  if (s.length === 0) return Number.NaN;
  const pos = (q / 100) * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(s.length - 1, lo + 1);
  return (s[lo] as number) + ((s[hi] as number) - (s[lo] as number)) * (pos - lo);
}
