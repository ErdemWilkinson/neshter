import { createStore } from 'zustand/vanilla';
import { subscribeWithSelector } from 'zustand/middleware';
import type { Measurement } from '../measurement';

export interface MeasureState {
  /** Latest measurement per op-set fingerprint. */
  byFp: Record<string, Measurement>;
  put(m: Measurement): void;
  get(fp: string): Measurement | undefined;
}

export const createMeasureStore = () =>
  createStore<MeasureState>()(
    subscribeWithSelector((set, get) => ({
      byFp: {},
      put: (m) => set((s) => ({ byFp: { ...s.byFp, [m.ops_fp]: m } })),
      get: (fp) => (Object.hasOwn(get().byFp, fp) ? get().byFp[fp] : undefined),
    })),
  );

export const measureStore = createMeasureStore();
