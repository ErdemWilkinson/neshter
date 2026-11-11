const LOCK_NAME = 'neshter-ui';

// One answer per lock manager. React StrictMode runs the boot effect twice; a second `ifAvailable` request would
// find the lock the first one holds and wrongly report another tab.
const attempts = new WeakMap<LockManager, Promise<'owner' | 'other'>>();

/**
 * Takes the page-wide Web Lock `neshter-ui` and keeps it for the life of the page (the callback returns a promise
 * that never settles). `'other'`: another tab holds it. No Web Locks API (old browsers, jsdom): `'owner'`.
 */
export function acquireTabLock(locks: LockManager | undefined = globalThis.navigator?.locks): Promise<'owner' | 'other'> {
  if (locks === undefined) return Promise.resolve('owner');
  const known = attempts.get(locks);
  if (known !== undefined) return known;
  const attempt = new Promise<'owner' | 'other'>((resolve) => {
    locks
      .request(LOCK_NAME, { ifAvailable: true }, (lock) => {
        if (lock === null) {
          resolve('other');
          return undefined;
        }
        resolve('owner');
        return new Promise<never>(() => undefined);
      })
      .catch(() => resolve('owner')); // a broken lock manager must not lock the person out
  });
  attempts.set(locks, attempt);
  return attempt;
}
