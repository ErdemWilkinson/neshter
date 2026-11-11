import { OUT_ID, edgeId, headId, neuronId, parseNodeId, tokenIdId, tokenPosId } from '../ids';

test('headId round-trips and rejects non-canonical ids', () => {
  expect(parseNodeId(headId(3, 5))).toEqual({ kind: 'head', layer: 3, head: 5 });
  for (const bad of ['h:01:2', 'h:-1:2', 'h:1', 'x:1:2', 'h:1:2:3']) expect(parseNodeId(bad)).toBeNull();
});

test('every id kind round-trips', () => {
  expect(parseNodeId(neuronId(2, 0))).toEqual({ kind: 'neuron', layer: 2, index: 0 });
  expect(parseNodeId(tokenPosId(7))).toEqual({ kind: 'tpos', pos: 7 });
  expect(parseNodeId(tokenIdId(151643))).toEqual({ kind: 'tid', id: 151643 });
  expect(parseNodeId(OUT_ID)).toEqual({ kind: 'out' });
  expect(headId(0, 0)).toBe('h:0:0');
  expect(edgeId('a-b')).toBe('e:a-b');
});

test('malformed ids of the other kinds are null', () => {
  for (const bad of ['', 'out:1', 'n:1', 't:', 't:01', 't:1:2', 'tid:-3', 'tid:1.5', 'h:1:2e3', 'h: 1:2', 'e:1', 'h:99999999999999999999:1']) {
    expect(parseNodeId(bad)).toBeNull();
  }
});

test('constructors refuse values that would not parse back', () => {
  expect(() => headId(-1, 0)).toThrow(RangeError);
  expect(() => headId(1.5, 0)).toThrow(RangeError);
  expect(() => tokenPosId(Number.NaN)).toThrow(RangeError);
});
