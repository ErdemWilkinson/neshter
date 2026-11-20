import { useStore } from 'zustand';
import { uiStore, type Lang } from '@/core/stores/ui';
import en from './en.json';
import tr from './tr.json';

export type MessageKey = keyof typeof tr;
export type Params = Record<string, string | number>;

// Compile-time parity: en has exactly the keys of tr (check-i18n.mjs also checks the files and placeholders).
const catalogs: Record<Lang, Record<MessageKey, string>> = { tr, en };

const formatters = new Map<Lang, Intl.NumberFormat>();
function numberFormat(lang: Lang): Intl.NumberFormat {
  let format = formatters.get(lang);
  if (format === undefined) {
    format = new Intl.NumberFormat(lang);
    formatters.set(lang, format);
  }
  return format;
}

/** `{name}` placeholders; numbers are formatted for the current language (tr: `0,025`), strings pass through. */
export function t(key: MessageKey, params?: Params): string {
  const lang = uiStore.getState().lang;
  const message = catalogs[lang][key];
  if (params === undefined) return message;
  return message.replace(/\{(\w+)\}/g, (placeholder, name: string) => {
    const value = Object.hasOwn(params, name) ? params[name] : undefined;
    if (value === undefined) return placeholder;
    return typeof value === 'number' ? numberFormat(lang).format(value) : value;
  });
}

/** Picks `<key>.<category>` (Intl.PluralRules: `one`, `other`, ...) and falls back to `<key>`. */
export function plural(key: MessageKey, n: number, params?: Params): string {
  const lang = uiStore.getState().lang;
  const category = new Intl.PluralRules(lang).select(n);
  const specific = `${key}.${category}`;
  const chosen = Object.hasOwn(catalogs[lang], specific) ? (specific as MessageKey) : key;
  return t(chosen, { n, ...params });
}

/** The saved choice, else English: the app starts in English whatever the system language is. */
export function resolveLang(prefLang: Lang | null): Lang {
  return prefLang ?? 'en';
}

export function setLang(lang: Lang): void {
  uiStore.getState().setLang(lang);
  document.documentElement.lang = lang;
}

/** Subscribes the component to the language and returns `t`, so a language switch re-renders it. */
export function useT(): typeof t {
  useStore(uiStore, (s) => s.lang);
  return t;
}
