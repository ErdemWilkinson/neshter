// The only module that touches window.sessionStorage (ESLint bans it elsewhere). Nothing is persisted beyond
// the tab: localStorage is banned everywhere in product code.

const TOKEN_KEY = 'neshter.token';
const ACTIVE_JOB_KEY = 'neshter.activeJob';
const EXPLORE_PROMPT_KEY = 'neshter.explorePrompt';

export function getSessionStorage(): Storage {
  return window.sessionStorage;
}

// Storage can throw (blocked site data, quota); the app then simply works without the convenience.
function read(ss: Storage, key: string): string | null {
  try {
    return ss.getItem(key);
  } catch {
    return null;
  }
}

function write(ss: Storage, key: string, value: string | null): void {
  try {
    if (value === null) ss.removeItem(key);
    else ss.setItem(key, value);
  } catch {
    // ignore
  }
}

export const getToken = (ss: Storage = getSessionStorage()): string | null => read(ss, TOKEN_KEY);
export const setToken = (token: string, ss: Storage = getSessionStorage()): void => write(ss, TOKEN_KEY, token);

export const getActiveJob = (ss: Storage = getSessionStorage()): string | null => read(ss, ACTIVE_JOB_KEY);
export const setActiveJob = (id: string, ss: Storage = getSessionStorage()): void => write(ss, ACTIVE_JOB_KEY, id);
export const clearActiveJob = (ss: Storage = getSessionStorage()): void => write(ss, ACTIVE_JOB_KEY, null);

/** The Explore prompt survives a reload of this tab only (never beyond the tab, never on disk). */
export const getExplorePrompt = (ss: Storage = getSessionStorage()): string | null => read(ss, EXPLORE_PROMPT_KEY);
export const setExplorePrompt = (prompt: string, ss: Storage = getSessionStorage()): void =>
  write(ss, EXPLORE_PROMPT_KEY, prompt === '' ? null : prompt);
