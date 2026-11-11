import { acquireTabLock } from '../singleTab';

type Callback = (lock: object | null) => unknown;

/** A lock manager that grants the lock once, then answers `null` (the lock is held for the life of the page). */
function fakeLocks(held: boolean) {
  let taken = held;
  const request = vi.fn(async (_name: string, _opts: unknown, cb: Callback) => {
    if (taken) return cb(null);
    taken = true;
    return cb({ name: 'neshter-ui' });
  });
  return { locks: { request } as unknown as LockManager, request };
}

test('the first tab owns the lock and holds it with a promise that never settles', async () => {
  const { locks, request } = fakeLocks(false);
  expect(await acquireTabLock(locks)).toBe('owner');
  expect(request).toHaveBeenCalledWith('neshter-ui', { ifAvailable: true }, expect.any(Function));
  const callback = request.mock.calls[0]?.[2] as Callback;
  const held = callback({}) as Promise<unknown>;
  const settled = vi.fn();
  void held.then(settled, settled);
  await Promise.resolve();
  expect(settled).not.toHaveBeenCalled();
});

test('second tab shows the multi-tab screen', async () => {
  expect(await acquireTabLock(fakeLocks(true).locks)).toBe('other');
});

test('a missing Web Locks API means owner (old browsers, jsdom)', async () => {
  expect(await acquireTabLock(undefined)).toBe('owner');
});

test('asking twice on one lock manager gives the same answer (StrictMode runs the boot effect twice)', async () => {
  const { locks, request } = fakeLocks(false);
  const [first, second] = await Promise.all([acquireTabLock(locks), acquireTabLock(locks)]);
  expect([first, second]).toEqual(['owner', 'owner']);
  expect(request).toHaveBeenCalledTimes(1);
});

test('a lock manager that throws does not lock the person out', async () => {
  const locks = { request: vi.fn(() => Promise.reject(new Error('SecurityError'))) } as unknown as LockManager;
  expect(await acquireTabLock(locks)).toBe('owner');
});
