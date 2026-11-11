import { createStore } from 'zustand/vanilla';
import { subscribeWithSelector } from 'zustand/middleware';
import type { NodeId } from '../ids';

export interface SelectionState {
  selected: ReadonlySet<NodeId>;
  focus: NodeId | null;
  toggle(id: NodeId): void;
  set(ids: Iterable<NodeId>): void;
  clear(): void;
  setFocus(id: NodeId | null): void;
}

export const createSelectionStore = () =>
  createStore<SelectionState>()(
    subscribeWithSelector((set) => ({
      selected: new Set<NodeId>(),
      focus: null,
      toggle: (id) =>
        set((s) => {
          const next = new Set(s.selected);
          if (!next.delete(id)) next.add(id);
          return { selected: next };
        }),
      set: (ids) => set({ selected: new Set(ids) }),
      clear: () => set({ selected: new Set<NodeId>() }),
      setFocus: (focus) => set({ focus }),
    })),
  );

export const selectionStore = createSelectionStore();
