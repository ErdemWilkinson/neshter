import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import conceptFixture from '@/api/__fixtures__/concepts.json';
import estimateFixture from '@/api/__fixtures__/estimate.json';
import jobExport from '@/api/__fixtures__/job_export.json';
import { createApi } from '@/api/client';
import { setApi } from '@/api/currentApi';
import type { SystemInfo } from '@/api/schemas';
import { bodyOf, fakeServer, jsonResponse, system, type Handler } from '@/app/__tests__/fakeServer';
import { Header } from '@/app/Header';
import { IDLE } from '@/core/saveflow';
import { basketStore } from '@/core/stores/basket';
import { compareStore } from '@/core/stores/compare';
import { jobsStore } from '@/core/stores/jobs';
import { sessionStore } from '@/core/stores/session';
import { uiStore } from '@/core/stores/ui';
import { registerFakeExploreOp } from '@/core/__tests__/fixtures/fakeExploreOp';
import { head, item, must } from '@/core/__tests__/fixtures/items';
import { saveFlowStore } from '@/features/saveflow/saveFlowStore';
import { setLang } from '@/i18n';
import { Save } from '@/screens/Save';

beforeAll(registerFakeExploreOp);

const sys = (patch: Record<string, unknown> = {}) => ({ ...system, ...patch }) as unknown as SystemInfo;

function mount(overrides: Record<string, Handler> = {}) {
  const server = fakeServer({
    'GET /api/concepts': () => jsonResponse(200, conceptFixture),
    'POST /api/export/estimate': () => jsonResponse(200, estimateFixture),
    'POST /api/export': () => jsonResponse(200, { job_id: jobExport.id }),
    [`GET /api/jobs/${jobExport.id}`]: () => jsonResponse(200, jobExport),
    ...overrides,
  });
  setApi(createApi({ token: 't', fetchFn: server.fetchFn as unknown as typeof fetch, onStale: () => undefined }));
  render(
    <>
      <Header onClose={() => undefined} />
      <Save />
    </>,
  );
  return server;
}

const findCall = (server: ReturnType<typeof fakeServer>, key: string) => must(server.calls.find((c) => c.key === key));
const saveButton = () => screen.getByRole('button', { name: /^Kaydet/ });
const nameBox = () => screen.getByLabelText('Ad') as HTMLInputElement;
/** The estimate answered: the button label carries the sizes and the name is pre-filled. */
const ready = async () => {
  await waitFor(() => expect(saveButton()).toHaveTextContent('gerekli'));
  expect(nameBox().value).toBe('abc123-neshter');
};

beforeEach(() => {
  setLang('tr');
  window.location.hash = '#/';
  uiStore.setState({ route: 'save' });
  sessionStore.setState({ system: sys() });
  basketStore.getState().replaceAll({ items: [head(0, 1), head(0, 2)], shelf: [], savedOps: [] });
});
afterEach(() => {
  cleanup();
  setApi(null);
  sessionStore.setState({ system: null, busy: false });
  jobsStore.setState({ active: null });
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
  compareStore.setState({ lastPrompt: null, lastPair: null, lastSuite: null, refreshedDropped: null });
  saveFlowStore.setState({ state: IDLE, error: null });
  uiStore.setState({ route: 'home' });
  vi.restoreAllMocks();
});

test('arriving on Save does not start an export', async () => {
  const server = mount();
  await ready();
  expect(server.count('POST /api/export')).toBe(0);
  expect(saveFlowStore.getState().state).toEqual(IDLE);
  expect(screen.queryByText('Kaydedildi')).toBeNull();
});

test('estimate fills the button label and the suggested name; info lines are shown', async () => {
  mount();
  await ready();
  expect(saveButton()).toHaveTextContent('Kaydet · 300 KB gerekli, 9,6 GB boş');
  expect(screen.getByText('Tarif (.nester.json), model kartı ve lisans her zaman kaydedilir.')).toBeInTheDocument();
  expect(
    screen.getByText('Orijinal dosya salt okunur; kayıttan sonra parmak izi (SHA-256) kontrol edilir.'),
  ).toBeInTheDocument();
});

test('GGUF disabled without llama.cpp converter; safetensors default', async () => {
  mount();
  await ready();
  expect(screen.getByRole('radio', { name: /safetensors/ })).toBeChecked();
  expect(screen.getByRole('radio', { name: /GGUF Q4_K_M/ })).toBeDisabled();
  expect(screen.getByRole('radio', { name: /GGUF Q8_0/ })).toBeDisabled();
});

test('GGUF enabled with the converter and marked deneysel', async () => {
  sessionStore.setState({ system: sys({ llama_cpp: { found: true, path: 'x', converter: true } }) });
  mount();
  await ready();
  const q8 = screen.getByRole('radio', { name: /GGUF Q8_0/ });
  expect(q8).toBeEnabled();
  expect(q8.closest('label')).toHaveTextContent('deneysel');
});

test('Ollama only with GGUF and ollama on PATH', async () => {
  sessionStore.setState({ system: sys({ llama_cpp: { found: true, path: 'x', converter: true }, ollama: true }) });
  const user = userEvent.setup();
  mount();
  await ready();
  const toggle = screen.getByRole('checkbox', { name: /Ollama/ });
  expect(toggle).toBeDisabled();
  expect(toggle).not.toBeChecked();
  await user.click(screen.getByRole('radio', { name: /GGUF Q8_0/ }));
  expect(toggle).toBeEnabled();
  expect(toggle).not.toBeChecked(); // off by default
  await user.click(screen.getByRole('radio', { name: /safetensors/ }));
  expect(toggle).toBeDisabled();
});

test('Ollama toggle stays disabled for GGUF when ollama is not on PATH', async () => {
  sessionStore.setState({ system: sys({ llama_cpp: { found: true, path: 'x', converter: true }, ollama: false }) });
  const user = userEvent.setup();
  mount();
  await ready();
  await user.click(screen.getByRole('radio', { name: /GGUF Q8_0/ }));
  expect(screen.getByRole('checkbox', { name: /Ollama/ })).toBeDisabled();
});

test('estimate not enough disables Kaydet', async () => {
  mount({ 'POST /api/export/estimate': () => jsonResponse(200, { ...estimateFixture, enough: false }) });
  await waitFor(() => expect(saveButton()).toHaveTextContent('gerekli'));
  expect(saveButton()).toBeDisabled();
  expect(screen.getByText('Diskte yeterli yer yok.')).toBeInTheDocument();
});

test('safety line shown when estimate says safety_ablated', async () => {
  mount({ 'POST /api/export/estimate': () => jsonResponse(200, { ...estimateFixture, safety_ablated: true }) });
  expect(await screen.findByText('Bu model safety-ablated olarak etiketlenecek.')).toBeInTheDocument();
});

test('no safety line otherwise', async () => {
  mount();
  await ready();
  expect(screen.queryByText('Bu model safety-ablated olarak etiketlenecek.')).toBeNull();
});

test('estimate body: savable ops, folder as typed, concept', async () => {
  basketStore.getState().replaceAll({
    items: [head(0, 1, { concept: 'refusal' }), item({ type: 'test_explore', src: 'h:0:1' }, { concept: 'refusal' })],
    shelf: [],
    savedOps: [],
  });
  const server = mount();
  await ready();
  expect(bodyOf(findCall(server, 'POST /api/export/estimate').init)).toEqual({
    format: 'safetensors',
    recipe_ops: [{ type: 'head', layer: 0, head: 1 }],
    out_dir: 'C:\\Users\\x\\model',
    concept: 'refusal',
  });
});

test('invalid name and folder show their messages and block Kaydet', async () => {
  const user = userEvent.setup();
  mount();
  await ready();
  await user.clear(nameBox());
  expect(screen.getByText('Bir ad girin.')).toBeInTheDocument();
  await user.type(nameBox(), 'şiir');
  expect(screen.getByText(/Türkçe harf/)).toBeInTheDocument();
  expect(saveButton()).toBeDisabled();
  await user.clear(nameBox());
  await user.type(nameBox(), 'ok-name');
  expect(saveButton()).toBeEnabled();
  const dir = screen.getByLabelText('Klasör');
  await user.clear(dir);
  await user.type(dir, '/x');
  expect(screen.getByText(/Tam yol girin/)).toBeInTheDocument();
  expect(saveButton()).toBeDisabled();
});

test('export body uses savable ops and joined out_dir', async () => {
  basketStore.getState().replaceAll({
    items: [head(0, 1, { concept: 'refusal' }), head(0, 2, { concept: 'refusal' })],
    shelf: [],
    savedOps: [],
  });
  const user = userEvent.setup();
  const server = mount();
  await ready();
  const dir = screen.getByLabelText('Klasör');
  await user.clear(dir);
  await user.type(dir, 'D:\\Modeller');
  await waitFor(() => expect(saveButton()).toHaveTextContent('gerekli'));
  await user.click(saveButton());
  await screen.findByText('Kaydedildi');
  expect(bodyOf(findCall(server, 'POST /api/export').init)).toEqual({
    recipe_ops: [
      { type: 'head', layer: 0, head: 1 },
      { type: 'head', layer: 0, head: 2 },
    ],
    name: 'abc123-neshter',
    format: 'safetensors',
    ollama: false,
    out_dir: 'D:\\Modeller\\abc123-neshter',
    concept: 'refusal',
  });
});

test('result card shows untouched source, SHA and safety badge; title is Kaydedildi', async () => {
  const done = { ...jobExport, result: { ...jobExport.result, safety_ablated: true } };
  const user = userEvent.setup();
  mount({ [`GET /api/jobs/${jobExport.id}`]: () => jsonResponse(200, done) });
  await ready();
  await user.click(saveButton());
  const title = await screen.findByRole('heading', { name: 'Kaydedildi' });
  const card = must(title.closest('section'));
  expect(card).toHaveTextContent('Orijinal dosyaya dokunulmadı');
  expect(card).toHaveTextContent(jobExport.result.source_sha256);
  expect(card).toHaveTextContent(jobExport.result.path);
  expect(card).toHaveTextContent(jobExport.result.name);
  expect(within(card).getByText('safety-ablated')).toBeInTheDocument();
  expect(within(card).getByText(jobExport.result.source_sha256)).toHaveClass('num');
});

test('result card omits untouched line and badge when the server does not say so', async () => {
  const done = { ...jobExport, result: { ...jobExport.result, source_untouched: false } };
  const user = userEvent.setup();
  mount({ [`GET /api/jobs/${jobExport.id}`]: () => jsonResponse(200, done) });
  await ready();
  await user.click(saveButton());
  const card = must((await screen.findByRole('heading', { name: 'Kaydedildi' })).closest('section'));
  expect(card).not.toHaveTextContent('Orijinal dosyaya dokunulmadı');
  expect(within(card).queryByText('safety-ablated')).toBeNull();
});

test('export done marks basket saved and the unsaved chip disappears', async () => {
  const user = userEvent.setup();
  mount();
  await ready();
  expect(screen.getByText('Kaydedilmemiş 2 kesim')).toBeInTheDocument();
  await user.click(saveButton());
  await screen.findByText('Kaydedildi');
  expect(screen.queryByText(/Kaydedilmemiş \d+ kesim/)).toBeNull();
  expect(basketStore.getState().savedOps).toHaveLength(2);
});

test('a cut added while the export runs is not marked saved', async () => {
  const user = userEvent.setup();
  mount({
    'POST /api/export': () => {
      basketStore.getState().add([head(7, 7)]); // CutButton does not check the job lock
      return jsonResponse(200, { job_id: jobExport.id });
    },
  });
  await ready();
  await user.click(saveButton());
  await screen.findByText('Kaydedildi');
  expect(basketStore.getState().savedOps).toHaveLength(2);
  expect(screen.getByText('Kaydedilmemiş 1 kesim')).toBeInTheDocument();
});

test('export error shows an error card by code and keeps the form', async () => {
  const failed = { ...jobExport, state: 'error', result: null, error: 'folder exists: C:\\x', error_code: 'exists' };
  const user = userEvent.setup();
  mount({ [`GET /api/jobs/${jobExport.id}`]: () => jsonResponse(200, failed) });
  await ready();
  await user.click(saveButton());
  expect(await screen.findByText('Bu ada sahip bir çıktı zaten var.')).toBeInTheDocument();
  expect(screen.queryByText('Kaydedildi')).toBeNull();
  expect(saveFlowStore.getState().state).toEqual(IDLE);
  expect(basketStore.getState().savedOps).toEqual([]);
  await waitFor(() => expect(saveButton()).toBeEnabled());
});

test('Kaydet with an explore item opens the discard dialog first', async () => {
  basketStore.getState().replaceAll({
    items: [head(0, 1), item({ type: 'test_explore', src: 'h:0:1' })],
    shelf: [],
    savedOps: [],
  });
  const user = userEvent.setup();
  const server = mount();
  await ready();
  await user.click(saveButton());
  const dialog = await screen.findByRole('dialog');
  expect(dialog).toHaveTextContent(
    'Şu 1 kesim yalnızca keşif içindir ve kaydedilemez: Kesim: test_explore. Bunlar sepetten atılacak, karşılaştırma yeniden koşulacak. Devam?',
  );
  expect(server.count('POST /api/export')).toBe(0);
  expect(saveFlowStore.getState().state.kind).toBe('needsDiscard');
});

test('Kaydet with nothing savable opens the nothing-to-save dialog', async () => {
  basketStore.getState().replaceAll({ items: [item({ type: 'test_explore', src: 'h:0:1' })], shelf: [], savedOps: [] });
  const user = userEvent.setup();
  const server = mount();
  await ready();
  await user.click(saveButton());
  expect(await screen.findByRole('dialog')).toHaveTextContent('Kaydedilecek kesim kalmıyor.');
  expect(server.count('POST /api/export')).toBe(0);
});
