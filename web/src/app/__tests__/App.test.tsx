import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useJob } from '@/api/useJob';
import { setApi, getApi } from '@/api/currentApi';
import { newBasketItem } from '@/core/basket';
import { parseOp } from '@/core/ops';
import { basketStore } from '@/core/stores/basket';
import { jobsStore } from '@/core/stores/jobs';
import { sessionStore } from '@/core/stores/session';
import { uiStore } from '@/core/stores/ui';
import { setLang } from '@/i18n';
import { App } from '../App';
import { getSessionStorage } from '../sessionStorage';
import { SCREENS } from '../Shell';
import { bodyOf, fakeServer, headersOf, jsonResponse, prefs, sessionLoaded } from './fakeServer';

function mount(server = fakeServer()) {
  vi.stubGlobal('fetch', server.fetchFn);
  return { ...render(<App />), server };
}

beforeEach(() => {
  getSessionStorage().clear();
  window.history.replaceState(null, '', '/#/');
  vi.spyOn(navigator, 'language', 'get').mockReturnValue('tr-TR');
  setLang('tr');
});

afterEach(() => {
  setApi(null);
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.useRealTimers();
  sessionStore.setState({ system: null, model: null, stale: 'none', busy: false });
  uiStore.setState({ route: 'home', lang: 'tr', reducedMotion: 'system' });
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
  jobsStore.setState({ active: null });
  document.documentElement.removeAttribute('data-motion');
});

test('App renders the shell with a mocked api and placeholder screens', async () => {
  mount();
  expect(screen.getByText('Yükleniyor…')).toBeInTheDocument();
  expect(await screen.findByRole('heading', { level: 1, name: 'Karşılama' })).toBeInTheDocument();

  expect(screen.getByRole('banner')).toBeInTheDocument();
  expect(screen.getByText('Yalnızca CPU')).toBeInTheDocument();
  expect(screen.getByText('abc123 · float32 · cpu')).toBeInTheDocument();
  expect(screen.queryByRole('navigation')).toBeNull(); // the stepper is hidden on home
  expect(document.documentElement).toHaveAttribute('lang', 'tr');
  expect(document.documentElement).toHaveAttribute('data-motion', 'full');
});

test('boot order: prefs, then system and session, with the token on every call', async () => {
  window.history.replaceState(null, '', '/#token=secret%20t');
  const { server } = mount();
  await screen.findByRole('heading', { name: 'Karşılama' });
  expect(server.calls.map((call) => call.key)).toEqual(['GET /api/prefs', 'GET /api/system', 'GET /api/session']);
  for (const call of server.calls) expect(headersOf(call.init)['X-Neshter-Token']).toBe('secret t');
  expect(window.location.hash).toBe('#/'); // the token is out of the address bar
});

test('the router follows the hash and the stepper marks the current step', async () => {
  mount();
  await screen.findByRole('heading', { name: 'Karşılama' });
  act(() => {
    window.location.hash = '#/concept';
  });
  expect(await screen.findByRole('heading', { level: 1, name: 'Davranışı bul' })).toBeInTheDocument();
  const nav = screen.getByRole('navigation', { name: 'Adımlar' });
  expect(nav.querySelector('[aria-current="step"]')).toHaveTextContent('Davranışı bul');
  expect(nav).toHaveTextContent('Ağ haritası');
  expect(screen.getByRole('main')).toHaveFocus(); // a route change moves focus to the new screen...
  expect(screen.getByRole('status')).toHaveTextContent('Davranışı bul'); // ...and announces it
  act(() => {
    window.location.hash = '#/no-such-screen';
  });
  expect(await screen.findByRole('heading', { level: 1, name: 'Karşılama' })).toBeInTheDocument();
});

test('prefs.lang wins over the browser language and sets <html lang>', async () => {
  mount(fakeServer({ 'GET /api/prefs': () => jsonResponse(200, { ...prefs, lang: 'en' }) }));
  expect(await screen.findByRole('heading', { level: 1, name: 'Welcome' })).toBeInTheDocument();
  expect(screen.getByText('CPU only')).toBeInTheDocument();
  expect(document.documentElement).toHaveAttribute('lang', 'en');
});

test('a failing prefs read falls back to defaults and the app still starts', async () => {
  mount(fakeServer({ 'GET /api/prefs': () => jsonResponse(500, { detail: 'x', error_code: 'internal' }) }));
  expect(await screen.findByRole('heading', { name: 'Karşılama' })).toBeInTheDocument();
});

test('an unloaded session shows no model chip', async () => {
  mount(fakeServer({ 'GET /api/session': () => jsonResponse(200, { loaded: false }) }));
  await screen.findByRole('heading', { name: 'Karşılama' });
  expect(sessionStore.getState().model).toBeNull();
  expect(screen.queryByText(/float32/)).toBeNull();
});

test('a 403 shows the one stale screen instead of the app', async () => {
  mount(fakeServer({ 'GET /api/session': () => jsonResponse(403, { detail: 'x', error_code: 'forbidden' }) }));
  expect(await screen.findByRole('heading', { name: 'Oturum bitti, sunucu yeniden başlatılmış' })).toBeInTheDocument();
  expect(screen.queryByRole('banner')).toBeNull();
  expect(screen.queryByText(/Teknik ayrıntı/)).toBeNull();
  expect(sessionStore.getState().stale).toBe('restarted');
});

test('another boot failure shows an error card with the detail only in the disclosure', async () => {
  mount(fakeServer({ 'GET /api/system': () => jsonResponse(500, { detail: 'boom <b>x</b>', error_code: 'internal' }) }));
  expect(await screen.findByText('Neşter başlatılamadı')).toBeInTheDocument();
  expect(screen.getByText('Beklenmeyen bir hata oldu.')).toBeInTheDocument();
  expect(screen.getByText('boom <b>x</b>').closest('details')).not.toBeNull();
});

test('without a token outside dev the app asks for the printed address', async () => {
  vi.stubEnv('DEV', false);
  const server = fakeServer();
  mount(server);
  expect(await screen.findByRole('heading', { name: "Neşter'i terminalde yazdırılan adresten açın." })).toBeInTheDocument();
  expect(server.fetchFn).not.toHaveBeenCalled();
});

test('the heartbeat reports a dead server as stale "down" and an unsaved cut changes the wording', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  basketStore.getState().add([
    newBasketItem(parseOp({ type: 'head', layer: 0, head: 1 }), { origin: 'heatmap', concept: null, addedAt: '2026-11-11T10:00:00.000Z' }),
  ]);
  let alive = true;
  mount(
    fakeServer({
      'GET /api/session': () => {
        if (!alive) throw new TypeError('Failed to fetch');
        return jsonResponse(200, sessionLoaded);
      },
    }),
  );
  await screen.findByRole('heading', { name: 'Karşılama' });
  expect(screen.getByText('Kaydedilmemiş 1 kesim')).toBeInTheDocument();
  alive = false;
  await act(() => vi.advanceTimersByTimeAsync(10_500));
  expect(await screen.findByRole('heading', { name: 'Sunucu kapandı; kesimlerin taslakta' })).toBeInTheDocument();
  expect(sessionStore.getState().stale).toBe('down');
});

test('Kapat shuts the server down and shows the closed screen', async () => {
  const { server } = mount();
  await screen.findByRole('heading', { name: 'Karşılama' });
  await userEvent.click(screen.getByRole('button', { name: 'Kapat' }));
  expect(await screen.findByRole('heading', { name: 'Sunucu kapandı' })).toBeInTheDocument();
  expect(server.count('POST /api/shutdown')).toBe(1);
});

test('a render error inside a screen shows the error screen, not a blank page', async () => {
  const original = SCREENS.home;
  const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
  SCREENS.home = function Broken() {
    throw new Error('screen exploded');
  };
  try {
    mount();
    expect(await screen.findByText('Bir şey ters gitti')).toBeInTheDocument();
    expect(screen.getByText('screen exploded').closest('details')).not.toBeNull();
    expect(screen.getByRole('banner')).toBeInTheDocument(); // the shell around the screen survives
  } finally {
    SCREENS.home = original;
    spy.mockRestore();
  }
});

test('useJob works inside the mounted App (the api is registered after bootstrap)', async () => {
  const original = SCREENS.home;
  const done = {
    id: 'j1',
    kind: 'sample',
    state: 'done',
    progress: 1,
    message: 'bitti',
    result: { path: 'C:\\x' },
    error: null,
    error_code: null,
    started_at: null,
  };
  SCREENS.home = function JobScreen() {
    const { job, start } = useJob<{ path: string }>();
    return (
      <div>
        <button type="button" onClick={() => void start(() => getApi().sample())}>
          başlat
        </button>
        <output>{job?.state ?? 'yok'}</output>
      </div>
    );
  };
  try {
    mount(
      fakeServer({
        'POST /api/models/sample': () => jsonResponse(200, { job_id: 'j1' }),
        'GET /api/jobs/j1': () => jsonResponse(200, done),
      }),
    );
    await userEvent.click(await screen.findByRole('button', { name: 'başlat' }));
    await waitFor(() => expect(document.querySelector('output')).toHaveTextContent('done'));
    expect(jobsStore.getState().active).toBeNull();
  } finally {
    SCREENS.home = original;
  }
});

test('the language select writes the whole prefs object, not just the change', async () => {
  const { server } = mount();
  await screen.findByRole('heading', { name: 'Karşılama' });
  await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Dil' }), 'en');
  expect(await screen.findByRole('heading', { name: 'Welcome' })).toBeInTheDocument();
  expect(document.documentElement).toHaveAttribute('lang', 'en');
  await waitFor(() => expect(server.count('PUT /api/prefs')).toBe(1));
  const put = server.calls.find((call) => call.key === 'PUT /api/prefs');
  expect(bodyOf(put?.init)).toEqual({ ...prefs, lang: 'en' });
});

test('the motion toggle sets data-motion and saves reduced_motion', async () => {
  const { server } = mount();
  await screen.findByRole('heading', { name: 'Karşılama' });
  const toggle = screen.getByRole('checkbox', { name: 'Animasyonları azalt' });
  expect(toggle).not.toBeChecked();
  await userEvent.click(toggle);
  expect(document.documentElement).toHaveAttribute('data-motion', 'reduce');
  expect(toggle).toBeChecked();
  await waitFor(() => expect(server.count('PUT /api/prefs')).toBe(1));
  expect(bodyOf(server.calls.find((call) => call.key === 'PUT /api/prefs')?.init)).toEqual({ ...prefs, reduced_motion: 'on' });
  await userEvent.click(toggle);
  expect(document.documentElement).toHaveAttribute('data-motion', 'full');
});
