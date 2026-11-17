import { isDeviceApp } from '@/api/device/bridge';
import { uiStore, type Route } from '@/core/stores/ui';

export const ROUTES = ['home', 'models', 'explore', 'concept', 'compare', 'save', 'chat'] as const;

// Compile-time check: ROUTES lists every member of the Route union (and nothing else).
type MissingRoutes = Exclude<Route, (typeof ROUTES)[number]>;
const routesAreComplete: [MissingRoutes] extends [never] ? true : never = true;
void routesAreComplete;

const HASH_ROUTE = /^#\/([a-z]+)(?:[/?].*)?$/;

/** Screens the phone app does not have: behaviour search needs gradients, saving writes model files. */
const NOT_ON_PHONE: readonly Route[] = ['concept', 'save'];
/** Screens only the phone app has: chatting runs on the phone's own llama.cpp. */
const NOT_ON_DESKTOP: readonly Route[] = ['chat'];

/**
 * `#/models` -> `models`; anything else (no hash, unknown name, `#token=...`, a desktop-only screen in the phone
 * app) -> `home`.
 */
export function parseHash(hash: string, phone: boolean = isDeviceApp()): Route {
  const name = HASH_ROUTE.exec(hash)?.[1];
  const route = ROUTES.find((r) => r === name) ?? 'home';
  return (phone ? NOT_ON_PHONE : NOT_ON_DESKTOP).includes(route) ? 'home' : route;
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
