import type { Api } from './client';

// The app creates one `Api` after the token is known (it needs `onStale`/`onBusy` wired to the session store) and
// registers it here, so hooks such as `useJob` reach it without a prop chain or a React context.
let current: Api | null = null;

export function setApi(api: Api | null): void {
  current = api;
}

export function getApi(): Api {
  if (current === null) throw new Error('The API client is not initialised (setApi was not called).');
  return current;
}
