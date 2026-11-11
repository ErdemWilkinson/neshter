import { uiStore, type Route } from '@/core/stores/ui';

export const ROUTES = ['home', 'models', 'explore', 'concept', 'compare', 'save'] as const;

// Compile-time check: ROUTES lists every member of the Route union (and nothing else).
type MissingRoutes = Exclude<Route, (typeof ROUTES)[number]>;
const routesAreComplete: [MissingRoutes] extends [never] ? true : never = true;
void routesAreComplete;

const HASH_ROUTE = /^#\/([a-z]+)(?:[/?].*)?$/;

/** `#/models` -> `models`; anything else (no hash, unknown name, `#token=...`) -> `home`. */
export function parseHash(hash: string): Route {
  const name = HASH_ROUTE.exec(hash)?.[1];
  return ROUTES.find((route) => route === name) ?? 'home';
}

export function navigate(route: Route): void {
  window.location.hash = `#/${route}`;
  uiStore.getState().setRoute(route); // hashchange arrives later; the screen should not wait for it
}

/** Mirrors the address hash into `uiStore.route` and follows `hashchange`. The token is bootstrap's business. */
export function startRouter(): () => void {
  const sync = () => uiStore.getState().setRoute(parseHash(window.location.hash));
  sync();
  window.addEventListener('hashchange', sync);
  return () => window.removeEventListener('hashchange', sync);
}
