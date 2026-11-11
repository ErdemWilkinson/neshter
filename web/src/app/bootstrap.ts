import { getToken, setToken } from './sessionStorage';

const TOKEN_IN_HASH = /^#token=([^&]+)/;

/**
 * Where the API token comes from. `#token=X` (the address the server prints) is moved into sessionStorage and the
 * hash is replaced so the token does not stay in the address bar or history. A reload reuses the stored token.
 * In dev with none, `''` (the Vite proxy injects it); otherwise `null` (show "open the address printed in the terminal").
 */
export function bootstrapToken(
  loc: Pick<Location, 'hash' | 'pathname'>,
  ss: Storage,
  hist: Pick<History, 'replaceState'>,
  dev: boolean,
): string | null {
  const match = TOKEN_IN_HASH.exec(loc.hash);
  if (match?.[1]) {
    let token: string;
    try {
      token = decodeURIComponent(match[1]);
    } catch {
      token = match[1];
    }
    setToken(token, ss);
    hist.replaceState(null, '', `${loc.pathname}#/`);
    return token;
  }
  return getToken(ss) ?? (dev ? '' : null);
}
