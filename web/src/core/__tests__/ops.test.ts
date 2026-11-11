import { opsFingerprint } from '../fingerprint';
import { isKnown, isSavable, isSendable, opKey, parseOp, registerOpKind } from '../ops';
import { z } from '../zod';
import { registerFakeExploreOp } from './fixtures/fakeExploreOp';
import { directionOp, headOp, vectorB64 } from './fixtures/items';

beforeAll(registerFakeExploreOp);

test('parseOp: head/direction known, fake explore known but not savable, unknown opaque', () => {
  expect(isSavable(parseOp({ type: 'head', layer: 0, head: 1 }))).toBe(true);
  const fake = parseOp({ type: 'test_explore', src: 'h:0:1' });
  expect(isSendable(fake) && !isSavable(fake)).toBe(true);
  const unk = parseOp({ type: 'edge_read_v9', x: 1 });
  expect(unk).toMatchObject({ opaque: true });
  expect(isSendable(unk)).toBe(false);
  expect(parseOp({ type: 'head', layer: 0, head: 1, extra: 1 })).toMatchObject({ opaque: true }); // strict
});

test('invalid head ops are opaque and keep their raw object', () => {
  const raw = { type: 'head', layer: -1, head: 0 };
  expect(parseOp(raw)).toEqual({ type: 'head', opaque: true, raw });
  expect(parseOp({ type: 'head', layer: 1.5, head: 0 })).toMatchObject({ opaque: true });
  expect(parseOp(null)).toMatchObject({ opaque: true, raw: {} });
  expect(parseOp('head')).toMatchObject({ opaque: true });
});

test('a registered kind that fails its schema is opaque', () => {
  expect(parseOp({ type: 'test_explore', src: 3 })).toMatchObject({ opaque: true });
});

test('a raw object that claims to be opaque stays opaque even for a registered kind', () => {
  expect(isKnown(parseOp({ type: 'test_explore', src: 'x', opaque: true }))).toBe(false);
});

test('direction op validates vector length', () => {
  const ok = directionOp([1, 2, 3, 4]);
  expect(isSavable(parseOp(ok))).toBe(true);
  expect(parseOp({ ...ok, d_model: 5 })).toMatchObject({ opaque: true });
  expect(parseOp({ ...ok, d_model: 3 })).toMatchObject({ opaque: true });
  expect(parseOp({ ...ok, vector_b64: 'not base64!' })).toMatchObject({ opaque: true });
  expect(parseOp({ ...ok, vector_b64: `${ok.vector_b64} ` })).toMatchObject({ opaque: true });
  expect(parseOp({ ...ok, vector_b64: '' })).toMatchObject({ opaque: true });
  expect(parseOp({ ...ok, extra: 1 })).toMatchObject({ opaque: true });
});

test('direction op with NaN or an all-zero vector is opaque', () => {
  expect(parseOp(directionOp([1, Number.NaN, 3]))).toMatchObject({ opaque: true });
  expect(parseOp(directionOp([1, Number.POSITIVE_INFINITY, 3]))).toMatchObject({ opaque: true });
  expect(parseOp(directionOp([0, 0, 0]))).toMatchObject({ opaque: true });
  expect(parseOp(directionOp([0, 0, 1e-30]))).toMatchObject({ opaque: true }); // float32 norm underflows to 0
  expect(parseOp(directionOp([1e20, 1e20]))).toMatchObject({ opaque: true }); // float32 norm overflows
  expect(isSavable(parseOp(directionOp([0, 0, 1])))).toBe(true);
});

test('vectorB64 helper encodes little-endian float32', () => {
  expect(vectorB64([1])).toBe('AACAPw==');
});

test('built-in kinds cannot be re-registered', () => {
  const spec = { schema: z.unknown(), savable: true, semantics: 'zero' as const };
  expect(() => registerOpKind('head', spec)).toThrow();
});

test('opKey is canonical: key order does not matter, opaque ops key on raw', () => {
  expect(opKey(parseOp({ layer: 1, type: 'head', head: 2 }))).toBe(opKey(parseOp(headOp(1, 2))));
  expect(opKey(parseOp(headOp(1, 2)))).toBe('{"head":2,"layer":1,"type":"head"}');
  const raw = { z: 1, type: 'edge_read_v9', a: { y: [1, { b: 2, a: 1 }] } };
  expect(opKey(parseOp(raw))).toBe('{"a":{"y":[1,{"a":1,"b":2}]},"type":"edge_read_v9","z":1}');
});

test('opsFingerprint ignores order and joins sorted keys with |', () => {
  const a = parseOp(headOp(1, 2));
  const b = parseOp(headOp(0, 3));
  if (!isKnown(a) || !isKnown(b)) throw new Error('expected known ops');
  expect(opsFingerprint([a, b])).toBe(opsFingerprint([b, a]));
  expect(opsFingerprint([a, b])).toBe(`${opKey(a)}|${opKey(b)}`);
  expect(opsFingerprint([])).toBe('');
});
