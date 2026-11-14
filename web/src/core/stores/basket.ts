import { createStore } from 'zustand/vanilla';
import { subscribeWithSelector } from 'zustand/middleware';
import { MAX_ITEMS, type BasketItem } from '../basket';
import { isSavable, opKey } from '../ops';

export interface BasketState {
  items: BasketItem[];
  shelf: BasketItem[];
  /** `opKey`s captured by the last save. */
  savedOps: string[];
  lastRemoved: BasketItem | null;
  /** The last `add` that landed: how many, and a running number (restores and undo do not count). */
  lastAdded: { n: number; seq: number } | null;
  /** Adds items whose `opKey` is not in the basket yet, up to `MAX_ITEMS`; returns how many were added. */
  add(items: readonly BasketItem[]): number;
  remove(id: string): void;
  undoRemove(): void;
  markPendingDiscard(ids: readonly string[]): void;
  clearPendingDiscard(): void;
  moveToShelf(ids: readonly string[]): void;
  /** Moves a shelf item back; `true` when it now sits in the basket (added, or already there). */
  restoreFromShelf(id: string): boolean;
  /**
   * Records the `opKey`s of the savable ops that were saved. `keys` are the ops the export actually sent; without
   * them every savable item in the basket counts as saved.
   */
  markSaved(keys?: readonly string[]): void;
  replaceAll(state: { items: readonly BasketItem[]; shelf: readonly BasketItem[]; savedOps: readonly string[] }): void;
}

const withoutPending = (item: BasketItem): BasketItem => {
  const copy = { ...item };
  delete copy.pendingDiscard;
  return copy;
};

export const createBasketStore = () =>
  createStore<BasketState>()(
    subscribeWithSelector((set, get) => {
      const add = (incoming: readonly BasketItem[]): number => {
        const items = [...get().items];
        const keys = new Set(items.map((item) => opKey(item.op)));
        let added = 0;
        for (const item of incoming) {
          if (items.length >= MAX_ITEMS) break;
          const key = opKey(item.op);
          if (keys.has(key)) continue;
          keys.add(key);
          items.push(item);
          added += 1;
        }
        if (added > 0) set((s) => ({ items, lastAdded: { n: added, seq: (s.lastAdded?.seq ?? 0) + 1 } }));
        return added;
      };

      return {
        items: [],
        shelf: [],
        savedOps: [],
        lastRemoved: null,
        lastAdded: null,
        add,
        remove: (id) =>
          set((s) => {
            const removed = s.items.find((item) => item.id === id);
            return removed ? { items: s.items.filter((item) => item.id !== id), lastRemoved: removed } : s;
          }),
        undoRemove: () => {
          const { lastRemoved } = get();
          if (lastRemoved === null) return;
          add([lastRemoved]);
          set({ lastRemoved: null });
        },
        markPendingDiscard: (ids) =>
          set((s) => ({ items: s.items.map((item) => (ids.includes(item.id) ? { ...item, pendingDiscard: true } : item)) })),
        clearPendingDiscard: () => set((s) => ({ items: s.items.map(withoutPending) })),
        moveToShelf: (ids) =>
          set((s) => {
            const moved = s.items.filter((item) => ids.includes(item.id)).map(withoutPending);
            if (moved.length === 0) return s;
            return {
              items: s.items.filter((item) => !ids.includes(item.id)),
              shelf: [...s.shelf, ...moved].slice(-MAX_ITEMS),
            };
          }),
        restoreFromShelf: (id) => {
          const item = get().shelf.find((candidate) => candidate.id === id);
          if (item === undefined) return false;
          const present = get().items.some((candidate) => opKey(candidate.op) === opKey(item.op));
          if (!present && get().items.length >= MAX_ITEMS) return false;
          set((s) => ({ shelf: s.shelf.filter((candidate) => candidate.id !== id) }));
          add([item]);
          return true;
        },
        markSaved: (keys) =>
          set((s) => {
            const sent = keys === undefined ? null : new Set(keys);
            const saved = s.items.filter((item) => isSavable(item.op)).map((item) => opKey(item.op));
            return { savedOps: sent === null ? saved : saved.filter((key) => sent.has(key)) };
          }),
        replaceAll: ({ items, shelf, savedOps }) =>
          set({
            items: items.slice(-MAX_ITEMS),
            shelf: shelf.slice(-MAX_ITEMS),
            savedOps: [...savedOps],
            lastRemoved: null,
          }),
      };
    }),
  );

export const basketStore = createBasketStore();
