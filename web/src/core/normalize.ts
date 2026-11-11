// Same min-max rules as neshter/activity.py (normalize_per_layer / normalize_global); a constant row or matrix is zeros.

/** `power_raw[layer][head][token]` -> `[layer][head]` at token `t` (negative `t` counts from the end). */
export function sliceToken(powerRaw: number[][][], t: number): number[][] {
  return powerRaw.map((layer, l) =>
    layer.map((head, h) => {
      const value = head[t < 0 ? head.length + t : t];
      if (value === undefined) throw new RangeError(`token index ${t} is out of range for head ${l}:${h}`);
      return value;
    }),
  );
}

export function normalizePerLayer(m: number[][]): number[][] {
  return m.map((row) => {
    const lo = Math.min(...row);
    const span = Math.max(...row) - lo;
    return row.map((x) => (span > 0 ? (x - lo) / span : 0));
  });
}

export function normalizeGlobal(m: number[][]): number[][] {
  const flat = m.flat();
  const lo = Math.min(...flat);
  const hi = Math.max(...flat);
  return m.map((row) => row.map((x) => (hi > lo ? (x - lo) / (hi - lo) : 0)));
}
