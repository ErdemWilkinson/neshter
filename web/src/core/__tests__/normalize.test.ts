import { normalizeGlobal, normalizePerLayer, sliceToken } from '../normalize';

test('normalize mirrors python: constant row -> zeros', () => {
  expect(normalizePerLayer([[2, 2], [1, 3]])).toEqual([[0, 0], [0, 1]]);
});

test('normalizePerLayer scales each row on its own', () => {
  expect(normalizePerLayer([[1, 2, 3], [10, 30, 20]])).toEqual([[0, 0.5, 1], [0, 1, 0.5]]);
  expect(normalizePerLayer([[5]])).toEqual([[0]]); // 1x1 model
});

test('normalizeGlobal uses one min-max; a constant matrix is zeros', () => {
  expect(normalizeGlobal([[1, 2], [3, 5]])).toEqual([[0, 0.25], [0.5, 1]]);
  expect(normalizeGlobal([[4, 4], [4, 4]])).toEqual([[0, 0], [0, 0]]);
  expect(normalizeGlobal([])).toEqual([]);
});

test('sliceToken picks a token, negative from the end', () => {
  const raw = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]; // [layer][head][token]
  expect(sliceToken(raw, 0)).toEqual([[1, 4], [7, 10]]);
  expect(sliceToken(raw, -1)).toEqual([[3, 6], [9, 12]]);
  expect(sliceToken(raw, -3)).toEqual([[1, 4], [7, 10]]);
});

test('sliceToken throws on an index outside the prompt instead of returning holes', () => {
  const raw = [[[1, 2, 3]]];
  expect(() => sliceToken(raw, 3)).toThrow(RangeError);
  expect(() => sliceToken(raw, -4)).toThrow(RangeError);
  expect(() => sliceToken(raw, 0.5)).toThrow(RangeError);
});
