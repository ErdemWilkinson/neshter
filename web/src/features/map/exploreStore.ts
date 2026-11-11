import { createStore } from 'zustand/vanilla';
import { subscribeWithSelector } from 'zustand/middleware';

/** What one `POST /api/activity` returned, with the prompt that produced it (the MLP list asks about the same text). */
export interface ActivityResult {
  prompt: string;
  tokens: string[];
  /** `[layer][head][token]` */
  powerRaw: number[][][];
}

export type Scale = 'layer' | 'global';
export type Unit = 'head' | 'mlp';

export interface ExploreState {
  prompt: string;
  result: ActivityResult | null;
  /** Position in `result.tokens`; `null` is the last token. */
  tokenIndex: number | null;
  scale: Scale;
  unit: Unit;
  setPrompt(prompt: string): void;
  setResult(result: ActivityResult | null): void;
  setTokenIndex(index: number | null): void;
  setScale(scale: Scale): void;
  setUnit(unit: Unit): void;
  /** A different model has different heads and tokens: the whole Explore view starts over. */
  reset(): void;
}

const initial = { prompt: '', result: null, tokenIndex: null, scale: 'layer', unit: 'head' } as const;

export const createExploreStore = () =>
  createStore<ExploreState>()(
    subscribeWithSelector((set) => ({
      ...initial,
      setPrompt: (prompt) => set({ prompt }),
      setResult: (result) => set({ result, tokenIndex: null }),
      setTokenIndex: (tokenIndex) => set({ tokenIndex }),
      setScale: (scale) => set({ scale }),
      setUnit: (unit) => set({ unit }),
      reset: () => set({ ...initial }),
    })),
  );

export const exploreStore = createExploreStore();
