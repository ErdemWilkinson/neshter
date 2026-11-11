import { createStore } from 'zustand/vanilla';

export interface DraftStatus {
  /** A server draft with unsaved cuts exists (written by autosave, or found by the restore prompt). */
  hasDraft: boolean;
  /** The last autosave was refused as too large (413); cleared by the next successful write. */
  tooLarge: boolean;
  /**
   * The model sha whose earlier-session draft is still unresolved (offered by the restore prompt, not yet restored
   * or deleted). While it equals the loaded model's sha, autosave writes nothing for that model: a write would
   * replace the earlier cuts (I1).
   */
  pendingSha: string | null;
  setHasDraft(value: boolean): void;
  setPendingSha(sha: string | null): void;
  setTooLarge(value: boolean): void;
}

export const draftStatusStore = createStore<DraftStatus>()((set) => ({
  hasDraft: false,
  tooLarge: false,
  pendingSha: null,
  setHasDraft: (hasDraft) => set({ hasDraft }),
  setPendingSha: (pendingSha) => set({ pendingSha }),
  setTooLarge: (tooLarge) => set({ tooLarge }),
}));
