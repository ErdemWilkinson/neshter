import { futilityK, layerMatchedHeads, percentile, sample, seededRandom, sequentialPermutation } from '../stats';

test('the seeded generator repeats for a seed and stays in [0, 1)', () => {
  const a = seededRandom(7);
  const b = seededRandom(7);
  const xs = Array.from({ length: 50 }, () => a());
  expect(Array.from({ length: 50 }, () => b())).toEqual(xs);
  expect(xs.every((x) => x >= 0 && x < 1)).toBe(true);
  expect(seededRandom(8)()).not.toBe(xs[0]);
});

test('sample draws distinct items and refuses more than the pool', () => {
  const got = sample([1, 2, 3, 4, 5], 3, seededRandom(1));
  expect(new Set(got).size).toBe(3);
  expect(() => sample([1], 2, seededRandom(1))).toThrow();
});

test('controls come from the same layers, never the selected heads', () => {
  const heads = [[0, 1], [0, 2], [5, 3]] as const;
  for (let seed = 0; seed < 20; seed++) {
    const { picked, matched } = layerMatchedHeads(heads, 6, 4, seededRandom(seed));
    expect(matched).toBe(true);
    expect(picked.map(([l]) => l).sort()).toEqual([0, 0, 5]);
    for (const [l, h] of picked) expect(heads.some(([a, b]) => a === l && b === h)).toBe(false);
  }
});

test('a layer without enough free heads borrows from the nearest layer, lower first, and says so', () => {
  const { picked, matched } = layerMatchedHeads([[1, 0], [1, 1]], 3, 2, seededRandom(3));
  expect(matched).toBe(false);
  expect(picked.map(([l]) => l)).toEqual([0, 0]);
  expect(() => layerMatchedHeads([[0, 0], [0, 1]], 1, 2, seededRandom(0))).toThrow();
});

test('futility k matches the desktop formula', () => {
  expect(futilityK(19)).toBe(0);
  expect(futilityK(39)).toBe(1);
  expect(futilityK(10)).toBe(-1);
});

test('the permutation stops early once significance is out of reach, else reports the full p', async () => {
  const always = await sequentialPermutation(1, async () => 2, 39);
  expect(always).toMatchObject({ nDrawn: 2, kExceed: 2, pKind: 'besag_clifford', significant: false, stoppedEarly: true });
  const never = await sequentialPermutation(1, async () => 0, 19);
  expect(never).toMatchObject({ nDrawn: 19, kExceed: 0, pValue: 1 / 20, pKind: 'full', significant: true });
  await expect(sequentialPermutation(1, async () => 0, 10)).rejects.toThrow();
});

test('percentile interpolates linearly like numpy', () => {
  expect(percentile([1, 2, 3, 4], 50)).toBeCloseTo(2.5);
  expect(percentile([10, 0], 95)).toBeCloseTo(9.5);
  expect(percentile([3], 5)).toBe(3);
});
