import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import conceptFixture from '@/api/__fixtures__/concepts.json';
import { createApi } from '@/api/client';
import { setApi } from '@/api/currentApi';
import type { CompareResponse, JobState, LoadedSession, SuiteResult } from '@/api/schemas';
import { bodyOf, fakeServer, jsonResponse, sessionLoaded, type Handler } from '@/app/__tests__/fakeServer';
import { BusyBanner } from '@/app/BusyBanner';
import { IDLE } from '@/core/saveflow';
import { opsFingerprint } from '@/core/fingerprint';
import { basketStore } from '@/core/stores/basket';
import { compareStore } from '@/core/stores/compare';
import { jobsStore } from '@/core/stores/jobs';
import { sessionStore } from '@/core/stores/session';
import { uiStore } from '@/core/stores/ui';
import { registerFakeExploreOp } from '@/core/__tests__/fixtures/fakeExploreOp';
import { head, headOp, item, must } from '@/core/__tests__/fixtures/items';
import { saveFlowStore } from '@/features/saveflow/saveFlowStore';
import { setLang } from '@/i18n';
import { Compare } from '@/screens/Compare';
import { suiteNoTarget } from './fixtures/suiteNoTarget';
import { suiteWithTarget } from './fixtures/suiteWithTarget';

beforeAll(registerFakeExploreOp);

const pair = (patch: Partial<CompareResponse> = {}): CompareResponse => ({
  original: 'orijinal cevap',
  modified: 'kesilmiş cevap',
  mode: 'batch',
  target_hit: null,
  ...patch,
});

const suiteJob = (result: SuiteResult): JobState<SuiteResult> => ({
  id: 's1',
  kind: 'suite',
  state: 'done',
  progress: 1,
  message: '',
  result,
  error: null,
  error_code: null,
  started_at: null,
});

function mount(overrides: Record<string, Handler> = {}) {
  const server = fakeServer({
    'GET /api/concepts': () => jsonResponse(200, conceptFixture),
    'POST /api/compare': () => jsonResponse(200, pair()),
    'POST /api/suite': () => jsonResponse(200, { job_id: 's1' }),
    'GET /api/jobs/s1': () => jsonResponse(200, suiteJob(suiteWithTarget)),
    ...overrides,
  });
  setApi(createApi({ token: 't', fetchFn: server.fetchFn as unknown as typeof fetch, onStale: () => undefined }));
  render(
    <>
      <BusyBanner />
      <Compare />
    </>,
  );
  return server;
}

const callBody = (server: ReturnType<typeof fakeServer>, key: string) =>
  bodyOf(must(server.calls.find((call) => call.key === key)).init);

beforeEach(() => {
  setLang('tr');
  window.location.hash = '#/';
  uiStore.setState({ route: 'compare' });
  sessionStore.setState({ model: sessionLoaded as unknown as LoadedSession });
  basketStore.getState().replaceAll({ items: [head(0, 1, { concept: 'code' })], shelf: [], savedOps: [] });
});
afterEach(() => {
  cleanup();
  setApi(null);
  sessionStore.setState({ model: null, busy: false });
  jobsStore.setState({ active: null });
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
  compareStore.setState({ lastPrompt: null, lastPair: null, lastSuite: null, refreshedDropped: null });
  saveFlowStore.setState({ state: IDLE, error: null });
  vi.restoreAllMocks();
});

const ask = async (user: ReturnType<typeof userEvent.setup>, prompt = 'Merhaba') => {
  await user.type(screen.getByLabelText('Kendin dene'), prompt);
  await user.click(screen.getByRole('button', { name: 'İkisine de sor' }));
};

test('compare body excludes opaque and pending-discard ops', async () => {
  basketStore.getState().replaceAll({
    items: [
      head(0, 1, { concept: 'code' }),
      item({ type: 'test_explore', src: 'h:0:1' }, { pendingDiscard: true }),
      item({ type: 'from_the_future', x: 1 }),
    ],
    shelf: [],
    savedOps: [],
  });
  const user = userEvent.setup();
  const server = mount();
  await ask(user);
  await screen.findByText('orijinal cevap');
  expect(callBody(server, 'POST /api/compare')).toEqual({
    prompt: 'Merhaba',
    recipe_ops: [headOp(0, 1)],
    concept_id: 'code',
    max_new_tokens: 200,
  });
  expect(compareStore.getState().lastPrompt).toBe('Merhaba');
});

test('pair view renders model output as text', async () => {
  const evil = '<img src=x onerror=alert(1)>';
  const user = userEvent.setup();
  mount({ 'POST /api/compare': () => jsonResponse(200, pair({ original: evil })) });
  await ask(user);
  expect(await screen.findByText(evil)).toBeInTheDocument();
  expect(document.querySelector('img')).toBeNull();
  expect(screen.getByText(/sıcaklık 0 · seed 42 · batch/)).toBeInTheDocument();
});

test('compare shows busy banner on 409', async () => {
  const user = userEvent.setup();
  mount({ 'POST /api/compare': () => jsonResponse(409, { detail: 'meşgul', error_code: 'busy' }) });
  await ask(user);
  expect(await screen.findByText('Model şu an çalışıyor; iş bitince devam edebilirsiniz.')).toBeInTheDocument();
  expect(screen.queryByRole('alert')).toBeNull();
});

test('target_hit badges per side', async () => {
  const user = userEvent.setup();
  mount({
    'POST /api/compare': () => jsonResponse(200, pair({ target_hit: { original: true, modified: false } })),
  });
  await ask(user);
  const original = (await screen.findByRole('heading', { name: 'Orijinal' })).closest('section') as HTMLElement;
  const cut = screen.getByRole('heading', { name: 'Kesilmiş' }).closest('section') as HTMLElement;
  expect(within(original).getByText('hedef davranış var')).toBeInTheDocument();
  expect(within(cut).getByText('hedef davranış yok')).toBeInTheDocument();
});

test('pair view and suite are disabled with zero sendable ops', () => {
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
  mount();
  expect(screen.getByRole('button', { name: 'İkisine de sor' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Test setini koş' })).toBeDisabled();
  expect(screen.getByText('Sepette karşılaştırılacak kesim yok.')).toBeInTheDocument();
});

test('suite runs quick on cpu with the basket concept target', async () => {
  const user = userEvent.setup();
  const server = mount();
  expect(screen.getByRole('radio', { name: 'Hızlı' })).toBeChecked();
  await user.click(screen.getByRole('button', { name: 'Test setini koş' }));
  await screen.findByText('Hedef davranış belirgin şekilde azaldı.');
  expect(callBody(server, 'POST /api/suite')).toEqual({
    recipe_ops: [headOp(0, 1)],
    target: 'code',
    quick: true,
  });
  expect(compareStore.getState().lastSuite).toEqual({
    fp: opsFingerprint([must(basketStore.getState().items[0]).op as never]),
    result: suiteWithTarget,
  });
  expect(screen.queryByText('Sepet değişti; sayılar eski. Yeniden koş.')).toBeNull();
});

test('Tam sends quick false', async () => {
  const user = userEvent.setup();
  const server = mount();
  await user.click(screen.getByRole('radio', { name: 'Tam' }));
  await user.click(screen.getByRole('button', { name: 'Test setini koş' }));
  await screen.findByText('Hedef davranış belirgin şekilde azaldı.');
  expect(callBody(server, 'POST /api/suite')).toMatchObject({ quick: false });
});

test('summary fallback when target is null and no mde footer', () => {
  compareStore.getState().setLastSuite({
    fp: opsFingerprint([must(basketStore.getState().items[0]).op as never]),
    result: suiteNoTarget,
  });
  mount();
  expect(
    screen.getByText('Bu kavramın test kategorisi yok; aşağıdaki genel sayılara bakın.'),
  ).toBeInTheDocument();
  expect(screen.queryByText(/En küçük saptanabilir düşüş/)).toBeNull();
  expect(screen.queryByText('(hedef)', { exact: false })).toBeNull();
  expect(screen.getByText('Kontrol çekişi yapılmadı.')).toBeInTheDocument();
  expect(screen.getByText('Dosya boyutu değişmez')).toBeInTheDocument();
});

test('belirsiz marks uncertain rows and (hedef) marks the target', async () => {
  compareStore.getState().setLastSuite({
    fp: opsFingerprint([must(basketStore.getState().items[0]).op as never]),
    result: suiteWithTarget,
  });
  mount();
  await screen.findByText(/\(hedef\)/);
  const table = screen.getByRole('table');
  const codeRow = within(table).getByRole('row', { name: /Kod \(hedef\)/ });
  const generalRow = within(table).getByRole('row', { name: /^Genel/ });
  expect(within(codeRow).queryByText('belirsiz')).toBeNull();
  expect(within(generalRow).getByText('belirsiz')).toBeInTheDocument();
  expect(within(generalRow).queryByText(/\(hedef\)/)).toBeNull();
  expect(within(codeRow).getByText(/−60 puan/)).toBeInTheDocument();
  expect(screen.getByText('En küçük saptanabilir düşüş: 18 puan')).toBeInTheDocument();
  expect(screen.getByText(/%80 \(GA %61–%91\) → %20 \(GA %9–%39\)/)).toBeInTheDocument();
  expect(screen.getByText(/rastgele kesim ortalaması/)).toBeInTheDocument();
});

test('stale banner after the basket changes', async () => {
  compareStore.getState().setLastSuite({
    fp: opsFingerprint([must(basketStore.getState().items[0]).op as never]),
    result: suiteWithTarget,
  });
  mount();
  expect(screen.queryByText('Sepet değişti; sayılar eski. Yeniden koş.')).toBeNull();
  basketStore.getState().add([head(2, 3)]);
  expect(await screen.findByText('Sepet değişti; sayılar eski. Yeniden koş.')).toBeInTheDocument();
});

test('refreshed banner shows dropped count and routes to save without a modal', async () => {
  saveFlowStore.setState({ state: { kind: 'reviewReady', dropped: 2 } });
  const user = userEvent.setup();
  mount();
  expect(screen.getByText('Sayılar yenilendi; 2 kesim atıldı.')).toBeInTheDocument();
  expect(screen.queryByRole('dialog')).toBeNull();
  await user.click(screen.getByRole('button', { name: 'Modeli kaydet (1 kesim)' }));
  expect(uiStore.getState().route).toBe('save');
  expect(screen.queryByRole('dialog')).toBeNull();
  expect(saveFlowStore.getState().state).toEqual({ kind: 'reviewReady', dropped: 2 }); // no SAVE_CLICKED, no export
});

test('no rate-level random-control line and no custom test-set button', async () => {
  compareStore.getState().setLastSuite({
    fp: opsFingerprint([must(basketStore.getState().items[0]).op as never]),
    result: suiteWithTarget,
  });
  mount();
  expect(screen.queryByText(/Kendi test setini/)).toBeNull();
  expect(screen.queryByRole('button', { name: /kendi|yükle|custom/i })).toBeNull();
  // The only random control is the KL one; the target-rate card carries no random line.
  const targetCard = (await screen.findByRole('heading', { name: 'Hedef davranış oranı' })).closest('section') as HTMLElement;
  expect(within(targetCard).queryByText(/rastgele/i)).toBeNull();
});

test('a failed suite shows an error card and keeps the old result', async () => {
  const user = userEvent.setup();
  mount({ 'POST /api/suite': () => jsonResponse(500, { detail: 'boom', error_code: 'internal' }) });
  await user.click(screen.getByRole('button', { name: 'Test setini koş' }));
  await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
  expect(screen.getByText('Test seti tamamlanamadı')).toBeInTheDocument();
  expect(compareStore.getState().lastSuite).toBeNull();
});
