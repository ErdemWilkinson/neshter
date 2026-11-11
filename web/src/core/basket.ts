import type { Measurement } from './measurement';
import { isKnown, isSavable, opKey, opSemantics, type AnyOp, type KnownOp, type Semantics } from './ops';

/** Server draft lists hold at most this many entries (`DraftIn`); the basket store enforces it. */
export const MAX_ITEMS = 1000;

export type Origin = 'heatmap' | 'concept' | 'graph' | 'import';

export interface BasketItem {
  id: string;
  op: AnyOp;
  semantics: Semantics;
  measurement?: Measurement;
  origin: Origin;
  concept: string | null;
  addedAt: string;
  /** Explore-only item that the save flow is about to drop; never sent to compare/suite/export. */
  pendingDiscard?: boolean;
}

let counter = 0;
export function makeItemId(): string {
  const uuid = globalThis.crypto?.randomUUID?.bind(globalThis.crypto);
  if (uuid) return uuid();
  counter += 1;
  return `i${Date.now().toString(36)}${counter.toString(36)}`;
}

export function newBasketItem(
  op: AnyOp,
  fields: { origin: Origin; concept: string | null; addedAt: string },
): BasketItem {
  return { id: makeItemId(), op, semantics: opSemantics(op), ...fields };
}

/** An item is unsaved when its `opKey` is not in `savedOps` (what the last save captured). */
export function unsavedCount(items: readonly BasketItem[], savedOps: readonly string[]): number {
  const saved = new Set(savedOps);
  return items.reduce((n, item) => (saved.has(opKey(item.op)) ? n : n + 1), 0);
}

export const isDirty = (items: readonly BasketItem[], savedOps: readonly string[]): boolean =>
  unsavedCount(items, savedOps) > 0;

export const countSavable = (items: readonly BasketItem[]): number =>
  items.filter((item) => isSavable(item.op)).length;

/** Not savable and not already marked for discard (opaque ops count: they cannot be saved either). */
export const countExploreOnly = (items: readonly BasketItem[]): number =>
  items.filter((item) => !isSavable(item.op) && !item.pendingDiscard).length;

export const savableOps = (items: readonly BasketItem[]): KnownOp[] =>
  items.flatMap((item) => (isSavable(item.op) ? [item.op] : []));

/** What may be sent to compare/suite/export: known ops that are not pending discard. */
export const sendableOps = (items: readonly BasketItem[]): KnownOp[] =>
  items.flatMap((item) => (isKnown(item.op) && !item.pendingDiscard ? [item.op] : []));
