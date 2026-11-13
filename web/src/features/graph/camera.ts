import type { Vec3 } from './layout';

export interface Camera {
  yaw: number;
  pitch: number;
  zoom: number;
}

/** Straight above is the heatmap; `tilt` is the default graph view, `low` looks along the floor. */
export const VIEWS = {
  heat: { yaw: 0, pitch: Math.PI / 2, zoom: 1 },
  tilt: { yaw: -0.18, pitch: 0.95, zoom: 0.86 },
  low: { yaw: -0.4, pitch: 0.42, zoom: 0.95 },
} as const satisfies Record<string, Camera>;

export const PITCH_MIN = 0.25;
export const PITCH_MAX = Math.PI / 2;
export const ZOOM_MIN = 0.4;
export const ZOOM_MAX = 3;

export interface Viewport {
  w: number;
  h: number;
  /** World size of what must fit, from `extent()`. */
  world: { w: number; d: number };
}

/** px per world unit that fits the floor into the viewport when seen from straight above. */
export const fitScale = (vp: Viewport): number => Math.max(1e-6, Math.min((vp.w * 0.88) / vp.world.w, (vp.h * 0.86) / vp.world.d));

/**
 * World point → [screen x, screen y, depth factor]. The depth factor is 1 at the floor's centre seen from above
 * and grows toward the viewer, so marks drawn in px stay readable at any model size.
 */
export function project(cam: Camera, vp: Viewport, [x, y, z]: Vec3): [number, number, number] {
  const cy = Math.cos(cam.yaw);
  const sy = Math.sin(cam.yaw);
  const cp = Math.cos(cam.pitch);
  const sp = Math.sin(cam.pitch);
  const xr = x * cy - z * sy;
  const zr = x * sy + z * cy;
  const dist = 1.6 * Math.max(vp.world.w, vp.world.d);
  const depth = Math.max(dist * 0.2, dist + zr * cp - y * sp);
  const k = dist / depth;
  const s = fitScale(vp) * cam.zoom * k;
  return [vp.w / 2 + xr * s, vp.h / 2 - (zr * sp + y * cp) * s, k];
}

export const clamp01 = (x: number): number => Math.max(0, Math.min(1, x));
export const easeOut = (x: number): number => 1 - (1 - clamp01(x)) ** 3;

/** One frame of the critically-damped-ish approach used for every animated number in the scene. */
export const approach = (from: number, to: number, rate: number): number => {
  const next = from + (to - from) * rate;
  return Math.abs(next - to) < 1e-3 ? to : next;
};
