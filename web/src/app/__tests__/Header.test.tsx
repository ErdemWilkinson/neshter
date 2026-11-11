import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createApi } from '@/api/client';
import { setApi } from '@/api/currentApi';
import type { SystemInfo } from '@/api/schemas';
import { newBasketItem } from '@/core/basket';
import { parseOp } from '@/core/ops';
import { basketStore } from '@/core/stores/basket';
import { jobsStore } from '@/core/stores/jobs';
import { sessionStore } from '@/core/stores/session';
import { uiStore } from '@/core/stores/ui';
import { setLang } from '@/i18n';
import { ProgressCard } from '@/ui/ProgressCard';
import { Header } from '../Header';
import { BusyBanner } from '../BusyBanner';
import { DeviceBanner } from '../DeviceBanner';
import { StaleScreen } from '../StaleScreen';
import { Stepper } from '../Stepper';
import { fakeServer, jsonResponse, sessionLoaded, system } from './fakeServer';

const GIB = 1024 ** 3;
const sys = (patch: Partial<SystemInfo>): SystemInfo => ({ ...system, ...patch });

function useServer(server = fakeServer()) {
  setApi(createApi({ token: 't', fetchFn: server.fetchFn as unknown as typeof fetch, onStale: () => undefined }));
  return server;
}

const cut = (head: number) =>
  newBasketItem(parseOp({ type: 'head', layer: 0, head }), { origin: 'heatmap', concept: null, addedAt: '2026-11-11T10:00:00.000Z' });

beforeEach(() => setLang('tr'));
afterEach(() => {
  setApi(null);
  vi.useRealTimers();
  sessionStore.setState({ system: null, model: null, stale: 'none', busy: false });
  uiStore.setState({ route: 'home', lang: 'tr', reducedMotion: 'system' });
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
  document.documentElement.removeAttribute('data-motion');
});

test('header shows Yalnızca CPU when gpu_name is null and hides unsaved chip at 0', () => {
  useServer();
  sessionStore.getState().setSystem(sys({ gpu_name: null, vram_bytes: 0, ram_bytes: 31.74 * GIB }));
  render(<Header onClose={() => undefined} />);
  expect(screen.getByText('Yalnızca CPU')).toBeInTheDocument();
  expect(screen.queryByText(/VRAM/)).toBeNull();
  expect(screen.getByText('RAM 31,7 GB')).toBeInTheDocument();
  expect(screen.queryByText(/Kaydedilmemiş/)).toBeNull();
});

test('header shows GPU and VRAM chips when there is a GPU, and the unsaved chip follows the basket', () => {
  useServer();
  sessionStore.getState().setSystem(sys({ gpu_name: 'RTX <b>4090</b>', vram_bytes: 24 * GIB, ram_bytes: 64 * GIB }));
  render(<Header onClose={() => undefined} />);
  expect(screen.getByText('GPU RTX <b>4090</b>')).toBeInTheDocument();
  expect(document.querySelector('b')).toBeNull();
  expect(screen.getByText('VRAM 24 GB')).toBeInTheDocument();
  expect(screen.queryByText('Yalnızca CPU')).toBeNull();

  expect(screen.queryByText(/Kaydedilmemiş/)).toBeNull();
  act(() => void basketStore.getState().add([cut(1)]));
  expect(screen.getByText('Kaydedilmemiş 1 kesim')).toBeInTheDocument();
  act(() => void basketStore.getState().add([cut(2)]));
  expect(screen.getByText('Kaydedilmemiş 2 kesim')).toBeInTheDocument();
  act(() => basketStore.getState().markSaved());
  expect(screen.queryByText(/Kaydedilmemiş/)).toBeNull();
});

test('the unsaved chip reads in English and uses the singular for one cut', () => {
  useServer();
  setLang('en');
  basketStore.getState().add([cut(1)]);
  const { rerender } = render(<Header onClose={() => undefined} />);
  expect(screen.getByText('1 unsaved cut')).toBeInTheDocument();
  basketStore.getState().add([cut(2)]);
  rerender(<Header onClose={() => undefined} />);
  expect(screen.getByText('2 unsaved cuts')).toBeInTheDocument();
});

test('the model chip is name · dtype · device, rendered as text', () => {
  useServer();
  sessionStore.getState().setModel({ ...sessionLoaded, name: '<img src=x onerror=alert(1)>', dtype: 'bfloat16', device: 'cuda' } as never);
  render(<Header onClose={() => undefined} />);
  expect(screen.getByText('<img src=x onerror=alert(1)> · bfloat16 · cuda')).toBeInTheDocument();
  expect(document.querySelector('img')).toBeNull();
});

test('Kapat calls onClose', async () => {
  useServer();
  const onClose = vi.fn();
  render(<Header onClose={onClose} />);
  await userEvent.click(screen.getByRole('button', { name: 'Kapat' }));
  expect(onClose).toHaveBeenCalledTimes(1);
});

test('the language select switches the whole header to English', async () => {
  useServer();
  render(<Header onClose={() => undefined} />);
  await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Dil' }), 'en');
  expect(screen.getByRole('combobox', { name: 'Language' })).toHaveValue('en');
  expect(screen.getByRole('button', { name: 'Close' })).toBeInTheDocument();
  expect(screen.getByRole('checkbox', { name: 'Reduce motion' })).toBeInTheDocument();
});

test('Stepper is hidden on home and lists the four steps elsewhere', () => {
  const { container } = render(<Stepper />);
  expect(container).toBeEmptyDOMElement();
  act(() => uiStore.getState().setRoute('save'));
  const nav = screen.getByRole('navigation', { name: 'Adımlar' });
  const steps = within(nav).getAllByRole('listitem');
  expect(steps).toHaveLength(4);
  expect(steps.map((li) => li.textContent)).toEqual(['Model', 'Davranışı bulAğ haritası', 'Karşılaştır', 'Kaydet']);
  expect(within(nav).getByRole('link', { name: 'Kaydet' })).toHaveAttribute('aria-current', 'step');
  expect(within(nav).getByRole('link', { name: 'Model' })).toHaveAttribute('href', '#/models');
});

test('StaleScreen: the pinned titles, and the draft hint only when there is a draft', () => {
  const { rerender } = render(<StaleScreen kind="restarted" hasDraft={false} />);
  expect(screen.getByRole('heading')).toHaveTextContent('Oturum bitti, sunucu yeniden başlatılmış');
  expect(screen.queryByText(/geri yüklenebilir/)).toBeNull();
  rerender(<StaleScreen kind="restarted" hasDraft />);
  expect(screen.getByText('Kesimleriniz yeni oturumda geri yüklenebilir.')).toBeInTheDocument();
  rerender(<StaleScreen kind="down" hasDraft />);
  expect(screen.getByRole('heading')).toHaveTextContent('Sunucu kapandı; kesimlerin taslakta');
  rerender(<StaleScreen kind="down" hasDraft={false} />);
  expect(screen.getByRole('heading')).toHaveTextContent('Sunucu kapandı');
  expect(screen.getByRole('heading')).not.toHaveTextContent('taslakta');
});

test('DeviceBanner appears only when the model has a device note', () => {
  const { container } = render(<DeviceBanner />);
  expect(container).toBeEmptyDOMElement();
  act(() => sessionStore.getState().setModel({ ...sessionLoaded, device_note: 'GPU belleği yetmedi <i>x</i>' } as never));
  expect(screen.getByRole('complementary', { name: 'Cihaz notu' })).toHaveTextContent('GPU belleği yetmedi <i>x</i>');
  expect(document.querySelector('i')).toBeNull();
});

test('BusyBanner shows for a held model and clears a 409 flag once the server has no running job', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  let running = true;
  useServer(
    fakeServer({
      'GET /api/jobs': () => jsonResponse(200, running ? [{ id: 'j', kind: 'hunt', state: 'running', progress: 0.1, message: '', error: null, error_code: null, started_at: null, created_at: 'x', finished_at: null }] : []),
    }),
  );
  const { container } = render(<BusyBanner />);
  expect(container).toBeEmptyDOMElement();

  act(() => sessionStore.getState().setBusy(true));
  expect(await screen.findByRole('status')).toHaveTextContent('Model şu an çalışıyor');
  await vi.advanceTimersByTimeAsync(3100);
  expect(sessionStore.getState().busy).toBe(true); // a job still runs
  running = false;
  await vi.advanceTimersByTimeAsync(3100);
  expect(sessionStore.getState().busy).toBe(false);
  expect(container).toBeEmptyDOMElement();
});

test('BusyBanner stays hidden for our own job while a progress card shows it, and floats over the page', () => {
  const job = { id: 'j1', kind: 'hunt', state: 'running', progress: 0.2, message: '', error: null, error_code: null, started_at: null, created_at: 'x', finished_at: null };
  act(() => jobsStore.getState().setActive(job as never));
  render(<BusyBanner />);
  expect(screen.getByRole('status')).toHaveClass('busy-banner'); // own job, no card: the notice explains the disabled buttons
  const card = render(<ProgressCard job={job as never} onCancel={() => undefined} />);
  expect(screen.queryByRole('status')).toBeNull();
  card.unmount();
  expect(screen.getByRole('status')).toBeInTheDocument();
  act(() => jobsStore.getState().clearActive());
});
