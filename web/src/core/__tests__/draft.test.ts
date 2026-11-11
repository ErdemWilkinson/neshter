import type { BasketItem } from '../basket';
import type { Measurement } from '../measurement';
import { decodeDraft, encodeDraft, type DraftInput } from '../draft';
import { isOpaque, opKey } from '../ops';
import { registerFakeExploreOp } from './fixtures/fakeExploreOp';
import { NOW, SHA, directionOp, head, item, must } from './fixtures/items';

beforeAll(registerFakeExploreOp);

const model = { name: 'tiny-model', sha256: SHA };
const input = (over: Partial<DraftInput> = {}): DraftInput => ({
  items: [],
  shelf: [],
  savedOps: [],
  model,
  prompts: [],
  ...over,
});

function draftWith(ops: unknown[], over: Record<string, unknown> = {}) {
  return {
    schema_version: 1,
    sha256: SHA,
    model_name: 'tiny-model',
    unsaved_count: ops.length,
    basket: ops.map((op, i) => ({ id: `i${i}`, op, semantics: 'zero', origin: 'heatmap', concept: null, addedAt: NOW })),
    shelf: [],
    saved_ops: [],
    prompts: [],
    ...over,
  };
}

test('decodeDraft keeps unknown ops opaque', () => {
  const r = decodeDraft(draftWith([{ type: 'edge_read_v9' }]));
  expect(r.ok && r.unsupported === 1 && r.items.length === 1).toBe(true);
  if (r.ok) expect(isOpaque(must(r.items[0]).op)).toBe(true);
});

test('decodeDraft keeps known ops known and counts only unsupported ones', () => {
  const r = decodeDraft(draftWith([{ type: 'head', layer: 0, head: 1 }, { type: 'test_explore', src: 'x' }, { type: 'nope' }]));
  if (!r.ok) throw new Error('expected ok');
  expect(r.items.map((i) => isOpaque(i.op))).toEqual([false, false, true]);
  expect(r.items.map((i) => i.semantics)).toEqual(['zero', 'path', 'unknown']);
  expect(r.unsupported).toBe(1);
});

test('decodeDraft rejects other schema versions', () => {
  expect(decodeDraft(draftWith([], { schema_version: 2 }))).toEqual({ ok: false, reason: 'schema_version' });
  expect(decodeDraft(draftWith([], { schema_version: 0 }))).toEqual({ ok: false, reason: 'schema_version' });
});

test('decodeDraft rejects malformed drafts', () => {
  for (const bad of [null, 'x', [], {}, draftWith([], { schema_version: '1' }), draftWith([], { basket: 'no' }), draftWith([], { prompts: [1] })]) {
    expect(decodeDraft(bad)).toEqual({ ok: false, reason: 'invalid' });
  }
  const noOp = draftWith([{ type: 'head', layer: 0, head: 1 }]);
  (noOp.basket[0] as Record<string, unknown>)['op'] = 'head';
  expect(decodeDraft(noOp)).toEqual({ ok: false, reason: 'invalid' });
  const noId = draftWith([{ type: 'head', layer: 0, head: 1 }]);
  delete (noId.basket[0] as Record<string, unknown>)['id'];
  expect(decodeDraft(noId)).toEqual({ ok: false, reason: 'invalid' });
});

test('decodeDraft tolerates unknown item fields and unknown origins (a newer writer)', () => {
  const d = draftWith([{ type: 'head', layer: 0, head: 1 }]);
  Object.assign(d.basket[0] as object, { origin: 'viewport3d', colour: 'red' });
  const r = decodeDraft(d);
  expect(r.ok && r.items[0]?.origin === 'import').toBe(true);
});

test('encodeDraft round-trips through decodeDraft and drops pendingDiscard', () => {
  const items = [head(0, 1, { concept: 'code' }), item(directionOp([1, 2, 3]), { origin: 'import' }), item({ type: 'test_explore', src: 'h:0:1' }, { pendingDiscard: true }), item({ type: 'edge_read_v9', x: { b: 1, a: 2 } })];
  const shelf = [head(4, 4)];
  const payload = encodeDraft(input({ items, shelf, savedOps: [opKey(must(items[0]).op)], prompts: ['hello', 'merhaba'] }));
  expect(payload).toMatchObject({ schema_version: 1, sha256: SHA, model_name: 'tiny-model', unsaved_count: 3, saved_ops: [opKey(must(items[0]).op)], prompts: ['hello', 'merhaba'] });
  expect(JSON.stringify(payload)).not.toContain('pendingDiscard');
  const back = decodeDraft(JSON.parse(JSON.stringify(payload)));
  if (!back.ok) throw new Error('expected ok');
  expect(back.items.map((i) => ({ ...i, pendingDiscard: undefined }))).toEqual(items.map((i) => ({ ...i, pendingDiscard: undefined })));
  expect(back.shelf).toEqual(shelf);
  expect(back.savedOps).toEqual(payload.saved_ops);
  expect(back.prompts).toEqual(['hello', 'merhaba']);
  expect(back.unsupported).toBe(1);
  expect(must(back.items[3]).op).toMatchObject({ opaque: true, raw: { type: 'edge_read_v9', x: { b: 1, a: 2 } } }); // written back verbatim
});

test('encodeDraft caps prompts at the newest 20 x 8000 chars', () => {
  const prompts = Array.from({ length: 25 }, (_, i) => `${i}`.padEnd(9000, 'x'));
  const payload = encodeDraft(input({ prompts }));
  expect(payload.prompts).toHaveLength(20);
  expect(payload.prompts.every((p) => p.length === 8000)).toBe(true);
  expect(must(payload.prompts[0]).startsWith('5x')).toBe(true);
  expect(must(payload.prompts[19]).startsWith('24x')).toBe(true);
});

test('encodeDraft never cuts a surrogate pair in half', () => {
  const emoji = '\u{1F600}';
  const payload = encodeDraft(input({ prompts: [`${'a'.repeat(7999)}${emoji}`] }));
  expect(payload.prompts[0]).toBe('a'.repeat(7999));
  expect(() => encodeURIComponent(must(payload.prompts[0]))).not.toThrow();
});

test('encodeDraft keeps the newest 1000 items, shelf entries and saved ops, and a model name within 256', () => {
  const many = Array.from({ length: 1003 }, (_, i) => head(Math.floor(i / 50), i % 50));
  const payload = encodeDraft(input({ items: many, shelf: many, savedOps: many.map((i) => opKey(i.op)), model: { name: 'm'.repeat(300), sha256: SHA } }));
  expect(payload.basket).toHaveLength(1000);
  expect(payload.shelf).toHaveLength(1000);
  expect(payload.saved_ops).toHaveLength(1000);
  expect(payload.basket.at(-1)?.['id']).toBe(many.at(-1)?.id);
  expect(payload.model_name).toHaveLength(256);
});

test('encodeDraft carries only ops, prompts and model name/SHA: no source_path, no model output, no measurements', () => {
  const smuggled: BasketItem = {
    ...head(2, 3),
    source_path: 'C:\\Users\\x\\secret-model',
    output: 'MODEL OUTPUT TEXT',
    measurement: { kind: 'set_effect', value: 1, note: 'MEASURED TEXT' } as unknown as Measurement,
  } as BasketItem;
  const loadedSession = { name: 'tiny-model', sha256: SHA, source_path: 'C:\\Users\\x\\secret-model', source_sha256: SHA };
  const payload = encodeDraft(
    input({ items: [smuggled], shelf: [smuggled], model: loadedSession, prompts: ['a prompt'] }),
  );
  const text = JSON.stringify(payload);
  for (const leak of ['source_path', 'secret-model', 'MODEL OUTPUT TEXT', 'MEASURED TEXT', 'measurement', 'source_sha256']) {
    expect(text).not.toContain(leak);
  }
  expect(Object.keys(payload).sort()).toEqual(
    ['basket', 'model_name', 'prompts', 'saved_ops', 'schema_version', 'sha256', 'shelf', 'unsaved_count'],
  );
  expect(Object.keys(must(payload.basket[0])).sort()).toEqual(['addedAt', 'concept', 'id', 'op', 'origin', 'semantics']);
  expect(must(payload.basket[0])['op']).toEqual({ type: 'head', layer: 2, head: 3 });
});
