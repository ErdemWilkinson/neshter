import { createStore } from 'zustand/vanilla';
import { subscribeWithSelector } from 'zustand/middleware';
import type { CompareResponse, SuiteResult } from '@/api/schemas';

export type ComparePair = CompareResponse;

export interface CompareState {
  lastPrompt: string | null;
  lastPair: ComparePair | null;
  /** `fp` is the `opsFingerprint` the suite ran with, so a stale result is recognisable. */
  lastSuite: { fp: string; result: SuiteResult } | null;
  /** How many cuts the last save-flow refresh dropped (shown once as "Sayılar yenilendi"). */
  refreshedDropped: number | null;
  setLastPrompt(prompt: string | null): void;
  setLastPair(pair: ComparePair | null): void;
  setLastSuite(suite: { fp: string; result: SuiteResult } | null): void;
  setRefreshedDropped(n: number | null): void;
}

export const createCompareStore = () =>
  createStore<CompareState>()(
    subscribeWithSelector((set) => ({
      lastPrompt: null,
      lastPair: null,
      lastSuite: null,
      refreshedDropped: null,
      setLastPrompt: (lastPrompt) => set({ lastPrompt }),
      setLastPair: (lastPair) => set({ lastPair }),
      setLastSuite: (lastSuite) => set({ lastSuite }),
      setRefreshedDropped: (refreshedDropped) => set({ refreshedDropped }),
    })),
  );

export const compareStore = createCompareStore();
