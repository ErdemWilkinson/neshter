import { createStore } from 'zustand/vanilla';
import type { Recipe } from '@/core/recipe';

export interface PendingImport {
  /** A parsed recipe waiting for a loaded model; memory only, never persisted. */
  recipe: Recipe | null;
  set(recipe: Recipe | null): void;
}

export const pendingImportStore = createStore<PendingImport>()((set) => ({
  recipe: null,
  set: (recipe) => set({ recipe }),
}));
