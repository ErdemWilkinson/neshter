import { uiStore } from '@/core/stores/ui';

const formats = new Map<string, Intl.NumberFormat>();

/** Three significant digits in the current language (`0,0021`), so small KL values and p-values stay readable. */
export function formatStat(value: number): string {
  const lang = uiStore.getState().lang;
  let format = formats.get(lang);
  if (format === undefined) {
    format = new Intl.NumberFormat(lang, { maximumSignificantDigits: 3 });
    formats.set(lang, format);
  }
  return format.format(value);
}
