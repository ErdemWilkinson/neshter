export type Rgb = [number, number, number];

export interface Palette {
  low: Rgb;
  high: Rgb;
  focusRing: string;
  basket: string;
  focusMark: string;
}

const HEX = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i;

/** `#rrggbb` -> channels; anything else (an unset token, `rgb(...)`) is `null`. */
export function parseHex(value: string): Rgb | null {
  const m = HEX.exec(value.trim());
  if (m === null) return null;
  const [, r = '0', g = '0', b = '0'] = m;
  return [parseInt(r, 16), parseInt(g, 16), parseInt(b, 16)];
}

function token(style: CSSStyleDeclaration | null, name: string, fallback: string): string {
  const value = style?.getPropertyValue(name).trim();
  return value ? value : fallback;
}

/** Reads the look tokens from `:root` once; fallbacks are the 3D §8 values (jsdom has no stylesheet). */
export function readPalette(): Palette {
  const style = typeof document === 'undefined' ? null : getComputedStyle(document.documentElement);
  return {
    low: parseHex(token(style, '--surface-1', '#14132a')) ?? [20, 19, 42],
    high: parseHex(token(style, '--accent-increase', '#5ec962')) ?? [94, 201, 98],
    focusRing: token(style, '--select-ring', '#ff3b30'),
    basket: token(style, '--accent-basket', '#ff7a45'),
    focusMark: token(style, '--text-1', '#e6eaf2'),
  };
}

/** A 0..1 value (clamped; NaN is 0) -> a colour between the panel colour and the increase accent. */
export function makeColorScale(low: Rgb, high: Rgb): (value: number) => Rgb {
  return (value) => {
    const v = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
    const mix = (a: number, b: number) => Math.round(a + (b - a) * v);
    return [mix(low[0], high[0]), mix(low[1], high[1]), mix(low[2], high[2])];
  };
}

export const rgbCss = ([r, g, b]: Rgb): string => `rgb(${r}, ${g}, ${b})`;

/** Viridis, five stops: perceptually even, and readable as dots on the dark panel. */
export const VIRIDIS: readonly Rgb[] = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];

/** A 0..1 value (clamped; NaN is 0) -> a colour along several stops, linear between neighbours. */
export function makeRampScale(stops: readonly Rgb[]): (value: number) => Rgb {
  return (value) => {
    const v = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
    const x = v * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(x));
    const f = x - i;
    const a = stops[i] ?? [0, 0, 0];
    const b = stops[i + 1] ?? a;
    return [Math.round(a[0] + (b[0] - a[0]) * f), Math.round(a[1] + (b[1] - a[1]) * f), Math.round(a[2] + (b[2] - a[2]) * f)];
  };
}

/** The heat colour used everywhere a head's strength is drawn (2D dots, the 3D floor, the head rings). */
export const heatColor = makeRampScale(VIRIDIS);
