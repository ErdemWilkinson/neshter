import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import conceptFixture from '@/api/__fixtures__/concepts.json';
import { createApi, type Api } from '@/api/client';
import { setApi } from '@/api/currentApi';
import type { Concept as ConceptInfo, HuntResult, JobState, LoadedSession } from '@/api/schemas';
import { bodyOf, fakeServer, jsonResponse, prefs, sessionLoaded, type Handler } from '@/app/__tests__/fakeServer';
import { basketStore } from '@/core/stores/basket';
import { jobsStore } from '@/core/stores/jobs';
import { sessionStore } from '@/core/stores/session';
import { uiStore } from '@/core/stores/ui';
import { vectorB64 } from '@/core/__tests__/fixtures/items';
import { setLang } from '@/i18n';
import { Concept } from '../Concept';

const lines = (n: number, prefix = 'istek') => Array.from({ length: n }, (_, i) => `${prefix} ${i}`);
const concepts: ConceptInfo[] = (conceptFixture as unknown as ConceptInfo[]).map((c) => ({
  ...c,
  positive: lines(32, 'evet'),
  negative: lines(32, 'hayır'),
}));

const huntResult = (patch: Partial<HuntResult> = {}): HuntResult => ({
  method: 'direction',
  layer_scores: [0, 1],
  heads: [{ layer: 0, head: 1, score: 0.8, measured: null }],
  direction: vectorB64([1, 2, 3, 4]),
  direction_shape: [4],
  direction_layer: 1,
  ref_layer: 1,
  val_accuracy: 0.9,
  val_ci: [0.7, 0.97],
  val_n: 16,
  ...patch,
});

const doneJob = (result: HuntResult): JobState<HuntResult> => ({
  id: 'h1',
  kind: 'hunt',
  state: 'done',
  progress: 1,
  message: '',
  result,
  error: null,
  error_code: null,
  started_at: null,
});

function mount(overrides: Record<string, Handler> = {}, list: ConceptInfo[] = concepts) {
  const server = fakeServer({
    'GET /api/concepts': () => jsonResponse(200, list),
    'POST /api/hunt': () => jsonResponse(200, { job_id: 'h1' }),
    'GET /api/jobs/h1': () => jsonResponse(200, doneJob(huntResult())),
    ...overrides,
  });
  setApi(createApi({ token: 't', fetchFn: server.fetchFn as unknown as typeof fetch, onStale: () => undefined }));
  render(<Concept />);
  return server;
}

const chip = (name: string | RegExp) => screen.findByRole('radio', { name });
const search = () => screen.getByRole('button', { name: 'Ara' });

beforeEach(() => {
  setLang('tr');
  window.location.hash = '#/';
  uiStore.setState({ route: 'concept' });
  sessionStore.setState({ model: sessionLoaded as unknown as LoadedSession });
});
afterEach(() => {
  cleanup();
  setApi(null);
  sessionStore.setState({ model: null, busy: false });
  jobsStore.setState({ active: null });
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
  vi.restoreAllMocks();
});

test('presets keep API order with code selected', async () => {
  mount();
  await chip('Kod yazma');
  const names = screen.getAllByRole('radio').map((radio) => radio.closest('label')?.textContent);
  expect(names.slice(0, 3)).toEqual(['Kod yazma', 'Reddetme', '+ Kendi kavramın']);
  expect(screen.getByRole('radio', { name: 'Kod yazma' })).toBeChecked();
  expect(screen.getByRole('heading', { level: 1, name: 'Davranışı bul' })).toBeInTheDocument();
});

test('a concept name with markup renders as text', async () => {
  const evil = '<img src=x onerror=alert(1)>';
  mount({}, [{ ...concepts[0], name_tr: evil } as ConceptInfo]);
  await chip(evil);
  expect(document.querySelector('img')).toBeNull();
});

test('refusal card shows once and stores the flag server-side', async () => {
  const user = userEvent.setup();
  const server = mount();
  await user.click(await chip('Reddetme'));
  expect(await screen.findByText(/safety-ablated olarak etiketlenir/)).toBeInTheDocument();
  expect(screen.getByText(/best-effort/)).toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'Anladım' }));
  expect(screen.queryByText(/safety-ablated olarak etiketlenir/)).toBeNull();
  await waitFor(() => expect(server.count('PUT /api/prefs')).toBe(1));
  expect(bodyOf(server.calls.find((call) => call.key === 'PUT /api/prefs')?.init)).toMatchObject({
    refusal_card_seen: true,
  });
  await user.click(await chip('Kod yazma'));
  await user.click(await chip('Reddetme'));
  expect(screen.queryByText(/safety-ablated olarak etiketlenir/)).toBeNull();
  expect(server.count('PUT /api/prefs')).toBe(1);
});

test('a card already seen on the server does not show again', async () => {
  const user = userEvent.setup();
  const server = mount({ 'GET /api/prefs': () => jsonResponse(200, { ...prefs, refusal_card_seen: true }) });
  await waitFor(() => expect(server.count('GET /api/prefs')).toBe(1));
  await user.click(await chip('Reddetme'));
  await waitFor(() => expect(screen.getByRole('radio', { name: 'Reddetme' })).toBeChecked());
  expect(screen.queryByText(/safety-ablated olarak etiketlenir/)).toBeNull();
});

test('refusal card PUT keeps the other prefs', async () => {
  const user = userEvent.setup();
  const server = mount({
    'GET /api/prefs': () => jsonResponse(200, { ...prefs, lang: 'en', quality: 'low' }),
  });
  await user.click(await chip('Reddetme'));
  await user.click(await screen.findByRole('button', { name: 'Anladım' }));
  await waitFor(() => expect(server.count('PUT /api/prefs')).toBe(1));
  expect(bodyOf(server.calls.find((call) => call.key === 'PUT /api/prefs')?.init)).toEqual({
    lang: 'en',
    quality: 'low',
    reduced_motion: 'system',
    refusal_card_seen: true,
  });
});

test('Ara stays disabled below 32 examples per side', async () => {
  const user = userEvent.setup();
  mount();
  await user.click(await chip('+ Kendi kavramın'));
  expect(search()).toBeDisabled();
  const [pos, neg] = screen.getAllByRole('textbox').filter((el) => el.tagName === 'TEXTAREA');
  fireEvent.change(pos as HTMLElement, { target: { value: lines(32).join('\n') } });
  expect(screen.getByText('32 / en az 32')).toBeInTheDocument();
  expect(search()).toBeDisabled(); // the other side is still empty
  fireEvent.change(neg as HTMLElement, { target: { value: lines(31).join('\n') + '\n\n  \n' } });
  expect(screen.getByText('31 / en az 32')).toBeInTheDocument();
  expect(search()).toBeDisabled();
  fireEvent.change(neg as HTMLElement, { target: { value: lines(32).join('\n') } });
  expect(search()).toBeEnabled();
});

test('a preset with 32 examples per side can be searched, one with fewer cannot', async () => {
  mount({}, [{ ...concepts[0], positive: lines(5) } as ConceptInfo, concepts[1] as ConceptInfo]);
  await chip('Kod yazma');
  expect(search()).toBeDisabled();
  await userEvent.click(await chip('Reddetme'));
  expect(search()).toBeEnabled();
});

test('method default follows session.default_method', async () => {
  sessionStore.setState({
    model: { ...(sessionLoaded as unknown as LoadedSession), default_method: 'direction' },
  });
  mount();
  await chip('Kod yazma');
  expect(screen.getByRole('radio', { name: /Davranış yönü/ })).toBeChecked();
  expect(document.body.textContent).not.toMatch(/\d+\s*(dk|sn)/);
});

test('a preset search posts concept_id and the chosen method, then shows the result', async () => {
  const user = userEvent.setup();
  const server = mount();
  await chip('Kod yazma');
  await user.click(screen.getByRole('radio', { name: /Davranış yönü/ }));
  await user.click(search());
  expect(await screen.findByText('doğrulamada doğruluk %90 (GA %70–%97, n=16)')).toBeInTheDocument();
  expect(bodyOf(server.calls.find((call) => call.key === 'POST /api/hunt')?.init)).toEqual({
    concept_id: 'code',
    method: 'direction',
  });
  expect(screen.getAllByText('gömme')).toHaveLength(1);
  expect(screen.getByRole('button', { name: 'Haritada göster ve ölç' })).toBeInTheDocument();
});

test('a custom concept is sent with a slug id, equal names and its examples', async () => {
  const user = userEvent.setup();
  const server = mount();
  await user.click(await chip('+ Kendi kavramın'));
  await user.type(screen.getByLabelText('Kavram adı'), 'Şiir yazma');
  const [pos, neg] = screen.getAllByRole('textbox').filter((el) => el.tagName === 'TEXTAREA');
  fireEvent.change(pos as HTMLElement, { target: { value: lines(33, 'a').join('\n') } });
  fireEvent.change(neg as HTMLElement, { target: { value: lines(32, 'b').join('\n') } });
  await user.click(search());
  await screen.findByText(/doğrulamada doğruluk/);
  expect(bodyOf(server.calls.find((call) => call.key === 'POST /api/hunt')?.init)).toEqual({
    concept: {
      id: 'Siir-yazma',
      name_tr: 'Şiir yazma',
      name_en: 'Şiir yazma',
      positive: lines(33, 'a'),
      negative: lines(32, 'b'),
    },
    method: 'activation_diff', // the loaded session's default
  });
});

test('safety concept direction carries concept refusal', async () => {
  const user = userEvent.setup();
  mount();
  await user.click(await chip('Reddetme'));
  await user.click(await screen.findByRole('button', { name: 'Anladım' }));
  await user.click(search());
  await user.click(await screen.findByRole('button', { name: 'Sepete ekle (geçici)' }));
  expect(basketStore.getState().items[0]).toMatchObject({ origin: 'concept', concept: 'refusal', semantics: 'direction' });
  const basket = screen.getByRole('region', { name: 'Sepet' });
  expect(within(basket).getByText('Yön · K0 sonrası')).toBeInTheDocument();
  expect(within(basket).getByText('kaydedilebilir')).toBeInTheDocument();
});

test('a plain preset direction carries its own concept id', async () => {
  const user = userEvent.setup();
  mount();
  await chip('Kod yazma');
  await user.click(search());
  await user.click(await screen.findByRole('button', { name: 'Sepete ekle (geçici)' }));
  expect(basketStore.getState().items[0]).toMatchObject({ concept: 'code' });
});

test('a failed hunt shows its error code message, not the server text', async () => {
  const user = userEvent.setup();
  mount({
    'GET /api/jobs/h1': () =>
      jsonResponse(200, { ...doneJob(huntResult()), state: 'error', result: null, error: 'sunucu ayrıntısı', error_code: 'internal' }),
  });
  await chip('Kod yazma');
  await user.click(search());
  expect(await screen.findByText('Arama tamamlanamadı')).toBeInTheDocument();
  expect(screen.getByRole('alert')).toBeInTheDocument();
  expect(screen.getByText('sunucu ayrıntısı').closest('details')).not.toBeNull(); // only inside the disclosure
});

test('a busy model (409) disables Ara through the banner state and shows no error card', async () => {
  const user = userEvent.setup();
  mount({
    'POST /api/hunt': () => jsonResponse(409, { detail: 'meşgul', error_code: 'busy' }),
  });
  await chip('Kod yazma');
  await user.click(search());
  await waitFor(() => expect(sessionStore.getState().busy).toBe(true));
  expect(screen.queryByRole('alert')).toBeNull();
});

test('the running hunt shows a progress card with İptal that cancels the job', async () => {
  const user = userEvent.setup();
  const running = { ...doneJob(huntResult()), state: 'running', progress: 0.3, result: null, message: 'çalışıyor' } as JobState;
  const api = {
    concepts: vi.fn(async () => concepts),
    prefs: vi.fn(async () => prefs),
    putPrefs: vi.fn(async (body: unknown) => body),
    hunt: vi.fn(async () => ({ job_id: 'h1' })),
    job: vi.fn(async () => running),
    streamJob: vi.fn(
      (_id: string, opts: { signal: AbortSignal }) =>
        new Promise<'ended' | 'aborted'>((resolve) => {
          opts.signal.addEventListener('abort', () => resolve('aborted'), { once: true });
        }),
    ),
    cancelJob: vi.fn(async () => ({ ...running, state: 'cancelled' })),
  };
  setApi(api as unknown as Api);
  render(<Concept />);
  await chip('Kod yazma');
  await user.click(search());
  expect(await screen.findByText('çalışıyor')).toBeInTheDocument();
  expect(search()).toBeDisabled(); // the job holds the model
  await user.click(screen.getByRole('button', { name: 'İptal' }));
  expect(api.cancelJob).toHaveBeenCalledWith('h1');
});
