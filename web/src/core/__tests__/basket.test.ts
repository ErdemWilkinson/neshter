import {
  MAX_ITEMS,
  countExploreOnly,
  countSavable,
  isDirty,
  savableOps,
  sendableOps,
  unsavedCount,
} from '../basket';
import { basketConcept } from '../basketConcept';
import { opKey } from '../ops';
import { createBasketStore } from '../stores/basket';
import { registerFakeExploreOp } from './fixtures/fakeExploreOp';
import { head, item } from './fixtures/items';
import type { Concept } from '@/api/schemas';

beforeAll(registerFakeExploreOp);

const explore = (src = 'h:0:1', extra?: Parameters<typeof item>[1]) => item({ type: 'test_explore', src }, extra);
const opaque = () => item({ type: 'edge_read_v9', x: 1 });

test('unsavedCount and isDirty follow savedOps', () => {
  const store = createBasketStore();
  store.getState().add([head(0, 1), head(0, 2)]);
  expect(unsavedCount(store.getState().items, store.getState().savedOps)).toBe(2);
  store.getState().markSaved();
  const { items, savedOps } = store.getState();
  expect(unsavedCount(items, savedOps)).toBe(0);
  expect(isDirty(items, savedOps)).toBe(false);
  store.getState().add([head(1, 1)]);
  expect(unsavedCount(store.getState().items, store.getState().savedOps)).toBe(1);
  expect(isDirty(store.getState().items, store.getState().savedOps)).toBe(true);
});

test('markSaved records savable items only, so explore-only items stay unsaved', () => {
  const store = createBasketStore();
  store.getState().add([head(0, 1), explore(), opaque()]);
  store.getState().markSaved();
  expect(store.getState().savedOps).toEqual([opKey(head(0, 1).op)]);
  expect(unsavedCount(store.getState().items, store.getState().savedOps)).toBe(2);
});

test('markSaved(keys) marks only the exported ops, so a cut added during the export stays unsaved', () => {
  const store = createBasketStore();
  store.getState().add([head(0, 1), head(0, 2)]);
  const exported = store.getState().items.map((i) => opKey(i.op));
  store.getState().add([head(5, 5)]); // cut while the export runs
  store.getState().markSaved(exported);
  const { items, savedOps } = store.getState();
  expect(savedOps).toEqual(exported);
  expect(unsavedCount(items, savedOps)).toBe(1);
  expect(isDirty(items, savedOps)).toBe(true);
});

test('selectors split savable, explore-only, sendable', () => {
  const items = [head(0, 1), explore('a'), explore('b', { pendingDiscard: true }), opaque()];
  expect(countSavable(items)).toBe(1);
  expect(countExploreOnly(items)).toBe(2); // the fake explore op without pending, and the opaque one
  expect(savableOps(items)).toHaveLength(1);
  expect(sendableOps(items).map((op) => op.type)).toEqual(['head', 'test_explore']); // no pending, no opaque
});

describe('basket store', () => {
  test('add dedupes by opKey, also inside one batch, and returns the added count', () => {
    const store = createBasketStore();
    expect(store.getState().add([head(0, 1), head(0, 1), head(0, 2)])).toBe(2);
    expect(store.getState().add([head(0, 2), head(3, 3)])).toBe(1);
    expect(store.getState().items).toHaveLength(3);
  });

  test('add stops at 1000 items and returns the added count', () => {
    const store = createBasketStore();
    const many = Array.from({ length: 1005 }, (_, i) => head(Math.floor(i / 50), i % 50));
    expect(store.getState().add(many)).toBe(MAX_ITEMS);
    expect(store.getState().items).toHaveLength(MAX_ITEMS);
    expect(store.getState().add([head(99, 99)])).toBe(0);
  });

  test('remove + undoRemove', () => {
    const store = createBasketStore();
    const a = head(0, 1);
    store.getState().add([a, head(0, 2)]);
    store.getState().remove(a.id);
    expect(store.getState().items).toHaveLength(1);
    expect(store.getState().lastRemoved).toBe(a);
    store.getState().undoRemove();
    expect(store.getState().items.map((i) => i.id)).toContain(a.id);
    expect(store.getState().lastRemoved).toBeNull();
    store.getState().remove('does-not-exist');
    expect(store.getState().items).toHaveLength(2);
  });

  test('moveToShelf + restoreFromShelf', () => {
    const store = createBasketStore();
    const a = head(0, 1);
    store.getState().add([a, head(0, 2)]);
    store.getState().markPendingDiscard([a.id]);
    store.getState().moveToShelf([a.id]);
    expect(store.getState().items).toHaveLength(1);
    expect(store.getState().shelf).toHaveLength(1);
    expect(store.getState().shelf[0]?.pendingDiscard).toBeUndefined();
    expect(store.getState().restoreFromShelf(a.id)).toBe(true);
    expect(store.getState().shelf).toHaveLength(0);
    expect(store.getState().items.map((i) => i.id)).toContain(a.id);
    expect(store.getState().restoreFromShelf('nope')).toBe(false);
  });

  test('shelf keeps the newest 1000', () => {
    const store = createBasketStore();
    const many = Array.from({ length: 600 }, (_, i) => head(1, i));
    store.getState().add(many);
    store.getState().moveToShelf(many.map((i) => i.id));
    const more = Array.from({ length: 600 }, (_, i) => head(2, i));
    store.getState().add(more);
    store.getState().moveToShelf(more.map((i) => i.id));
    expect(store.getState().shelf).toHaveLength(MAX_ITEMS);
    expect(store.getState().shelf.at(-1)?.id).toBe(more.at(-1)?.id);
    expect(store.getState().shelf[0]?.id).toBe(many[200]?.id);
  });

  test('restoreFromShelf keeps the item on the shelf when the basket is full', () => {
    const store = createBasketStore();
    const parked = head(5, 5);
    store.getState().replaceAll({ items: [], shelf: [parked], savedOps: [] });
    store.getState().add(Array.from({ length: MAX_ITEMS }, (_, i) => head(1, i)));
    expect(store.getState().restoreFromShelf(parked.id)).toBe(false);
    expect(store.getState().shelf).toHaveLength(1);
  });

  test('markPendingDiscard and clearPendingDiscard', () => {
    const store = createBasketStore();
    const a = head(0, 1);
    store.getState().add([a, head(0, 2)]);
    store.getState().markPendingDiscard([a.id]);
    expect(store.getState().items.filter((i) => i.pendingDiscard)).toHaveLength(1);
    store.getState().clearPendingDiscard();
    expect(store.getState().items.some((i) => i.pendingDiscard)).toBe(false);
  });

  test('replaceAll swaps everything and clamps to 1000', () => {
    const store = createBasketStore();
    const a = head(0, 1);
    store.getState().add([a]);
    store.getState().remove(a.id);
    store.getState().replaceAll({
      items: Array.from({ length: 1001 }, (_, i) => head(1, i)),
      shelf: [head(9, 9)],
      savedOps: ['k'],
    });
    expect(store.getState().items).toHaveLength(MAX_ITEMS);
    expect(store.getState().shelf).toHaveLength(1);
    expect(store.getState().savedOps).toEqual(['k']);
    expect(store.getState().lastRemoved).toBeNull();
  });

  test('subscribers are notified with selectors', () => {
    const store = createBasketStore();
    const seen: number[] = [];
    store.subscribe((s) => s.items.length, (n) => seen.push(n));
    store.getState().add([head(0, 1)]);
    store.getState().markSaved();
    expect(seen).toEqual([1]);
  });
});

describe('basketConcept', () => {
  const concept = (id: string, suite_target: string | null): Concept => ({
    id,
    name_tr: id,
    name_en: id,
    safety_ablation: id === 'refusal',
    suite_target,
    positive: [],
    negative: [],
  });
  const concepts = [concept('code', 'code'), concept('refusal', 'harmful_refusal'), concept('tone', null)];

  test('refusal wins, single shared concept, mixed -> null, tone -> suiteTarget null', () => {
    expect(basketConcept([head(0, 1, { concept: 'refusal' }), head(0, 2, { concept: 'code' })], concepts)).toEqual({
      conceptId: 'refusal',
      suiteTarget: 'harmful_refusal',
    });
    expect(basketConcept([head(0, 1, { concept: 'code' }), head(0, 2, { concept: 'code' })], concepts)).toEqual({
      conceptId: 'code',
      suiteTarget: 'code',
    });
    expect(basketConcept([head(0, 1, { concept: 'code' }), head(0, 2, { concept: 'tone' })], concepts)).toEqual({
      conceptId: null,
      suiteTarget: null,
    });
    expect(basketConcept([head(0, 1, { concept: 'tone' })], concepts)).toEqual({ conceptId: 'tone', suiteTarget: null });
  });

  test('no concept, empty basket and a manual pick next to a concept item all give null', () => {
    expect(basketConcept([], concepts)).toEqual({ conceptId: null, suiteTarget: null });
    expect(basketConcept([head(0, 1)], concepts)).toEqual({ conceptId: null, suiteTarget: null });
    expect(basketConcept([head(0, 1, { concept: 'code' }), head(0, 2)], concepts)).toEqual({
      conceptId: null,
      suiteTarget: null,
    });
  });

  test('an unknown shared concept id has no suite target; refusal matches case-insensitively', () => {
    expect(basketConcept([head(0, 1, { concept: 'custom' })], concepts)).toEqual({ conceptId: 'custom', suiteTarget: null });
    expect(basketConcept([head(0, 1, { concept: 'Refusal' })], concepts).conceptId).toBe('refusal');
  });
});
