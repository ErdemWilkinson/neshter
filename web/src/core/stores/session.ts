import { createStore } from 'zustand/vanilla';
import { subscribeWithSelector } from 'zustand/middleware';
import type { LoadedSession, SystemInfo } from '@/api/schemas';

export type Stale = 'none' | 'restarted' | 'down';

export interface SessionState {
  system: SystemInfo | null;
  model: LoadedSession | null;
  /** `restarted`: the token no longer fits (403). `down`: the server does not answer. */
  stale: Stale;
  /** A job or request holds the model (409). */
  busy: boolean;
  setSystem(system: SystemInfo | null): void;
  setModel(model: LoadedSession | null): void;
  setStale(stale: Stale): void;
  setBusy(busy: boolean): void;
}

export const createSessionStore = () =>
  createStore<SessionState>()(
    subscribeWithSelector((set) => ({
      system: null,
      model: null,
      stale: 'none',
      busy: false,
      setSystem: (system) => set({ system }),
      setModel: (model) => set({ model }),
      setStale: (stale) => set({ stale }),
      setBusy: (busy) => set({ busy }),
    })),
  );

export const sessionStore = createSessionStore();
