import { act, renderHook } from '@testing-library/react';
import { ApiError, BusyError } from '@/api/errors';
import { sessionStore } from '@/core/stores/session';
import { BUSY_RETRIES, BUSY_RETRY_MS, useInsight } from '../useInsight';

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

test('a busy model is asked again and the panel keeps loading until the answer comes', async () => {
  const load = vi
    .fn<() => Promise<string>>()
    .mockRejectedValueOnce(new BusyError())
    .mockRejectedValueOnce(new BusyError())
    .mockResolvedValue('lens');
  const { result } = renderHook(() => useInsight(true, 'k', load));
  await act(async () => {});
  expect(result.current.status).toBe('loading');
  await act(async () => vi.advanceTimersByTimeAsync(BUSY_RETRY_MS * 2));
  expect(load).toHaveBeenCalledTimes(3);
  expect(result.current).toEqual({ status: 'ready', data: 'lens' });
});

test('a model that stays busy hands over to the banner and goes idle', async () => {
  sessionStore.getState().setBusy(false);
  const load = vi.fn<() => Promise<string>>().mockRejectedValue(new BusyError());
  const { result } = renderHook(() => useInsight(true, 'k', load));
  await act(async () => vi.advanceTimersByTimeAsync(BUSY_RETRY_MS * (BUSY_RETRIES + 1)));
  expect(load).toHaveBeenCalledTimes(BUSY_RETRIES + 1);
  expect(result.current.status).toBe('idle');
  expect(sessionStore.getState().busy).toBe(true);
});

test('other errors are not retried', async () => {
  const load = vi.fn<() => Promise<string>>().mockRejectedValue(new ApiError(500, 'internal', 'x'));
  const { result } = renderHook(() => useInsight(true, 'k', load));
  await act(async () => vi.advanceTimersByTimeAsync(BUSY_RETRY_MS * 3));
  expect(load).toHaveBeenCalledTimes(1);
  expect(result.current).toMatchObject({ status: 'failed', code: 'internal' });
});

test('closing the panel stops the retries', async () => {
  const load = vi.fn<() => Promise<string>>().mockRejectedValue(new BusyError());
  const { rerender } = renderHook(({ open }) => useInsight(open, 'k', load), { initialProps: { open: true } });
  await act(async () => {});
  rerender({ open: false });
  await act(async () => vi.advanceTimersByTimeAsync(BUSY_RETRY_MS * 5));
  expect(load).toHaveBeenCalledTimes(1);
});
