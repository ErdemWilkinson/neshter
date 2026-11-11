import type { Api } from '@/api/client';
import type { Prefs } from '@/api/schemas';
import { uiStore, type Lang, type ReducedMotion } from '@/core/stores/ui';
import { setLang } from '@/i18n';

export const DEFAULT_PREFS: Prefs = { lang: null, quality: 'balanced', reduced_motion: 'system', refusal_card_seen: false };

type MatchMedia = ((query: string) => { matches: boolean }) | undefined;

/** `window.matchMedia`, or `undefined` where there is none (jsdom). */
export const systemMatchMedia = (): MatchMedia =>
  typeof window.matchMedia === 'function' ? window.matchMedia.bind(window) : undefined;

// The server PUT replaces the whole object, so the UI always sends the last known prefs plus its change.
let current: Prefs = DEFAULT_PREFS;

/** GET the prefs; on any failure the defaults stand (preferences must never block the app). */
export async function loadPrefs(api: Api): Promise<Prefs> {
  try {
    current = await api.prefs();
  } catch {
    current = DEFAULT_PREFS;
  }
  return current;
}

/** The only way the UI writes prefs: PUT of `{...current, ...patch}`; resolves with what the server stored. */
export async function savePrefs(api: Api, patch: Partial<Prefs>): Promise<Prefs> {
  const next = await api.putPrefs({ ...current, ...patch });
  current = next;
  return next;
}

/** Whether animations are off right now: the explicit choice, else the operating system's setting. */
export function isMotionReduced(pref: ReducedMotion, matchMedia: MatchMedia): boolean {
  if (pref === 'on') return true;
  if (pref === 'off') return false;
  return matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
}

/** Sets `html[data-motion]` to `reduce` or `full` (base.css turns transitions off for `reduce`). */
export function applyMotion(pref: ReducedMotion, matchMedia: MatchMedia): void {
  document.documentElement.dataset['motion'] = isMotionReduced(pref, matchMedia) ? 'reduce' : 'full';
}

/** Takes the stored prefs into the stores and the document. */
export function applyPrefs(prefs: Prefs, lang: Lang, matchMedia: MatchMedia): void {
  const ui = uiStore.getState();
  setLang(lang);
  ui.setReducedMotion(prefs.reduced_motion);
  ui.setQuality(prefs.quality);
  applyMotion(prefs.reduced_motion, matchMedia);
}

function persist(api: Api, patch: Partial<Prefs>): void {
  // A failed save only loses the persistence of a preference; the change already took effect.
  void savePrefs(api, patch).catch(() => undefined);
}

export function chooseLang(api: Api, lang: Lang): void {
  setLang(lang);
  persist(api, { lang });
}

export function chooseReducedMotion(api: Api, reduce: boolean, matchMedia: MatchMedia): void {
  const pref: ReducedMotion = reduce ? 'on' : 'off';
  uiStore.getState().setReducedMotion(pref);
  applyMotion(pref, matchMedia);
  persist(api, { reduced_motion: pref });
}
