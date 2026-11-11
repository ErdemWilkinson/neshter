import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { basketStore } from '@/core/stores/basket';
import { sessionStore } from '@/core/stores/session';
import { uiStore } from '@/core/stores/ui';
import { sessionLoaded } from '@/app/__tests__/fakeServer';
import { draftStatusStore } from '@/features/draft/draftStatus';
import type { LoadedSession } from '@/api/schemas';
import { head } from '@/core/__tests__/fixtures/items';
import { setLang, t } from '@/i18n';
import { useLeaveGuard } from '../useLeaveGuard';

function Harness({ proceed }: { proceed: () => void }) {
  const guard = useLeaveGuard();
  return (
    <div>
      <button type="button" onClick={() => guard.request('close', proceed)}>
        çık
      </button>
      {guard.dialog}
    </div>
  );
}

beforeEach(() => {
  setLang('tr');
  window.location.hash = '#/';
});
afterEach(() => {
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
  uiStore.setState({ route: 'home' });
  sessionStore.setState({ model: null });
  draftStatusStore.setState({ hasDraft: false, tooLarge: false, pendingSha: null });
  vi.restoreAllMocks();
});

const dirty = () =>
  act(() => {
    basketStore.getState().add([head(0, 1)]);
  });

test('exact leave copy', () => {
  setLang('tr');
  expect(t('leave.title')).toBe('Ayrılıyorsunuz, kesimlerinizi daha kaydetmediniz');
  expect(t('leave.save')).toBe('Kaydet');
  expect(t('leave.leave')).toBe('Kaydetmeden çık');
  expect(t('leave.cancel')).toBe('Vazgeç');
  setLang('en');
  expect(t('leave.title')).toBe('You are leaving and your cuts are not saved yet');
  expect(t('leave.leave')).toBe('Leave without saving');
});

test('leave guard proceeds at once when clean', async () => {
  const proceed = vi.fn();
  render(<Harness proceed={proceed} />);
  await userEvent.click(screen.getByRole('button', { name: 'çık' }));
  expect(proceed).toHaveBeenCalledTimes(1);
  expect(screen.queryByRole('dialog')).toBeNull();
});

test('leave guard shows the three exact buttons when dirty and Vazgeç keeps state', async () => {
  const proceed = vi.fn();
  render(<Harness proceed={proceed} />);
  dirty();
  await userEvent.click(screen.getByRole('button', { name: 'çık' }));
  const dialog = screen.getByRole('dialog', { name: 'Ayrılıyorsunuz, kesimlerinizi daha kaydetmediniz' });
  const names = Array.from(dialog.querySelectorAll('button')).map((b) => b.textContent);
  expect(names).toEqual(['Kaydet', 'Kaydetmeden çık', 'Vazgeç']);
  await userEvent.click(screen.getByRole('button', { name: 'Vazgeç' }));
  expect(screen.queryByRole('dialog')).toBeNull();
  expect(proceed).not.toHaveBeenCalled();
  expect(basketStore.getState().items).toHaveLength(1);
});

test('Kaydetmeden çık proceeds and keeps the basket; Kaydet opens the save screen without proceeding', async () => {
  const proceed = vi.fn();
  render(<Harness proceed={proceed} />);
  dirty();
  await userEvent.click(screen.getByRole('button', { name: 'çık' }));
  await userEvent.click(screen.getByRole('button', { name: 'Kaydetmeden çık' }));
  expect(proceed).toHaveBeenCalledTimes(1);
  expect(basketStore.getState().items).toHaveLength(1);

  await userEvent.click(screen.getByRole('button', { name: 'çık' }));
  await userEvent.click(screen.getByRole('button', { name: 'Kaydet' }));
  expect(proceed).toHaveBeenCalledTimes(1);
  expect(uiStore.getState().route).toBe('save');
});

test('beforeunload listener exists only while dirty', () => {
  const add = vi.spyOn(window, 'addEventListener');
  const remove = vi.spyOn(window, 'removeEventListener');
  const count = (spy: typeof add) => spy.mock.calls.filter(([type]) => type === 'beforeunload').length;
  render(<Harness proceed={() => undefined} />);
  expect(count(add)).toBe(0);
  dirty();
  expect(count(add)).toBe(1);
  const handler = add.mock.calls.find(([type]) => type === 'beforeunload')?.[1] as (e: BeforeUnloadEvent) => void;
  const event = { preventDefault: vi.fn(), returnValue: 'x' } as unknown as BeforeUnloadEvent;
  handler(event);
  expect(event.preventDefault).toHaveBeenCalled();
  expect(event.returnValue).toBe('');
  expect(count(remove)).toBe(0);
  act(() => {
    basketStore.getState().markSaved();
  });
  expect(count(remove)).toBe(1);
});

test('the leave body promises the draft only while autosave is not held back by an unresolved earlier draft', async () => {
  const model = sessionLoaded as unknown as LoadedSession;
  sessionStore.setState({ model });
  render(<Harness proceed={vi.fn()} />);
  dirty();
  await userEvent.click(screen.getByRole('button', { name: 'çık' }));
  expect(screen.getByText(/kesimleriniz taslakta kalır/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Vazgeç' }));

  act(() => draftStatusStore.getState().setPendingSha(model.source_sha256));
  await userEvent.click(screen.getByRole('button', { name: 'çık' }));
  expect(screen.queryByText(/kesimleriniz taslakta kalır/)).toBeNull();
  expect(screen.getByText(/yeni kesimler taslağa yazılmaz/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Vazgeç' }));

  act(() => setLang('en'));
  await userEvent.click(screen.getByRole('button', { name: 'çık' }));
  expect(screen.getByText(/New cuts are not written to the draft/)).toBeInTheDocument();
});
