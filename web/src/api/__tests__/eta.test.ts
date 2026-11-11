import { etaSeconds } from '../eta';

test('etaSeconds is null below 5% progress', () => {
  expect(etaSeconds(0.04, 9000)).toBeNull();
  expect(etaSeconds(0.5, 10_000)).toBe(10);
});

test('etaSeconds at the 5% boundary and at completion', () => {
  expect(etaSeconds(0.05, 1000)).toBeCloseTo(19);
  expect(etaSeconds(1, 5000)).toBe(0);
  expect(etaSeconds(Number.NaN, 5000)).toBeNull();
});
