import { formatFixed } from '@/features/map/format';
import { t } from '@/i18n';

/** A proportion (0..1) as a whole percentage in the current language. */
export const pct = (rate: number): string => formatFixed(rate * 100, 0);

/** `%12 (GA %3–%31)` for a rate and its Wilson interval. */
export const rateWithCi = (k: number, n: number, ci: readonly [number, number]): string =>
  t('compare.stat.rate', { p: n === 0 ? '–' : pct(k / n), lo: pct(ci[0]), hi: pct(ci[1]) });

/** A signed difference in percentage points (`+7`, `−25`, `0`). */
export function points(diff: number): string {
  const value = Math.round(diff * 100);
  if (value === 0) return '0';
  return `${value > 0 ? '+' : '−'}${Math.abs(value)}`;
}
