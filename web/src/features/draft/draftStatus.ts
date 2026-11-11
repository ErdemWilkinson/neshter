import { createStore } from 'zustand/vanilla';

export interface DraftStatus {
  /** A server draft with unsaved cuts exists (written by autosave, or found by the restore prompt). */
  hasDraft: boolean;
  /** The last autosave was refused as too large (413); cleared by the next successful write. */
  tooLarge: boolean;
  setHasDraft(value: boolean): void;
  setTooLarge(value: boolean): void;
}

export const draftStatusStore = createStore<DraftStatus>()((set) => ({
  hasDraft: false,
  tooLarge: false,
  setHasDraft: (hasDraft) => set({ hasDraft }),
  setTooLarge: (tooLarge) => set({ tooLarge }),
}));
