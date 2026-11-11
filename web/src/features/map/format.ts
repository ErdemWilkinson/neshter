import { uiStore } from '@/core/stores/ui';

const formats = new Map<string, Intl.NumberFormat>();

/** A number with a fixed count of decimals in the current language (`0,25` in tr). */
export function formatFixed(value: number, digits: number): string {
  const lang = uiStore.getState().lang;
  const key = `${lang}:${digits}`;
  let format = formats.get(key);
  if (format === undefined) {
    format = new Intl.NumberFormat(lang, { minimumFractionDigits: digits, maximumFractionDigits: digits });
    formats.set(key, format);
  }
  return format.format(value);
}

export const formatPower = (value: number): string => formatFixed(value, 2);
