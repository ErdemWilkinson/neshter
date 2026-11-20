import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import models from '@/api/__fixtures__/models.json';
import type { Api } from '@/api/client';
import { setApi } from '@/api/currentApi';
import { ApiError } from '@/api/errors';
import type { JobState, LoadedSession, ModelEntry } from '@/api/schemas';
import { sessionLoaded } from '@/app/__tests__/fakeServer';
import { basketStore } from '@/core/stores/basket';
import { jobsStore } from '@/core/stores/jobs';
import { measureStore } from '@/core/stores/measure';
import { exploreStore } from '@/features/map/exploreStore';
import type { Measurement } from '@/core/measurement';
import { sessionStore } from '@/core/stores/session';
import { uiStore, type Intent } from '@/core/stores/ui';
import { head } from '@/core/__tests__/fixtures/items';
import { SAMPLE_NAME } from '@/features/models/LoadScreen';
import { pendingImportStore } from '@/features/recipe/pendingImport';
import { IDLE, type SaveFlowState } from '@/core/saveflow';
import { saveFlowStore } from '@/features/saveflow/saveFlowStore';
import { setLang } from '@/i18n';
import { getSessionStorage } from '@/app/sessionStorage';
import { Home } from '../Home';
import { Models } from '../Models';

const tiny = models[0] as unknown as ModelEntry;
const loaded = sessionLoaded as unknown as LoadedSession;

const jobState = (patch: Partial<JobState>): JobState => ({
  id: 'j',
  kind: 'load',
  state: 'done',
  progress: 1,
  message: '',
  result: null,
  error: null,
  error_code: null,
  started_at: null,
  ...patch,
});

/** Jobs finish at once with the state registered for their id; `running` ids hold until `release()`. */
function makeApi(options: { rows?: ModelEntry[]; results?: Record<string, JobState> } = {}) {
  const results = options.results ?? { load1: jobState({ id: 'load1' }) };
  let release: () => void = () => undefined;
  const api = {
    models: vi.fn(async () => options.rows ?? [tiny]),
    drafts: vi.fn(async () => []),
    draft: vi.fn(async () => {
      throw new Error('no draft');
    }),
    loadModel: vi.fn(async () => ({ job_id: 'load1' })),
    sample: vi.fn(async () => ({ job_id: 'sample1' })),
    session: vi.fn(async () => loaded),
    recipeCheck: vi.fn(async () => ({ ok: true, errors: [], sha_mismatch: false })),
    cancelJob: vi.fn(async () => jobState({ state: 'cancelled' })),
    job: vi.fn(async (id: string) => results[id] ?? jobState({ id })),
    streamJob: vi.fn(
      (_id: string, opts: { signal: AbortSignal }) =>
        new Promise<'ended' | 'aborted'>((resolve) => {
          release = () => resolve('ended');
          opts.signal.addEventListener('abort', () => resolve('aborted'), { once: true });
        }),
    ),
  };
  setApi(api as unknown as Api);
  return { api, results, release: () => release() };
}

/** The `Aç` of the model list (the path field has one too, and shows before the list does). */
const rowOpen = async () => within(await screen.findByRole('list', { name: 'Modeller' })).getByRole('button', { name: 'Aç' });

beforeEach(() => {
  getSessionStorage().clear();
  setLang('tr');
  window.location.hash = '#/';
  uiStore.setState({ route: 'models', intent: null });
});
afterEach(() => {
  setApi(null);
  sessionStore.setState({ model: null });
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
  jobsStore.setState({ active: null });
  pendingImportStore.getState().set(null);
  saveFlowStore.setState({ state: IDLE, error: null });
  measureStore.getState().reset();
  exploreStore.getState().reset();
  uiStore.setState({ route: 'home', intent: null });
  vi.restoreAllMocks();
});

test('home offers two intents and expert mode, never "Modeli küçült"', () => {
  makeApi();
  render(<Home />);
  expect(screen.getByRole('heading', { level: 1, name: 'Ne yapmak istiyorsun?' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Bir davranışı azalt/ })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /İçine bak/ })).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Uzman modu' })).toBeInTheDocument();
  expect(screen.getByLabelText('Tarif içe aktar…')).toHaveAttribute('accept', expect.stringContaining('.json'));
  expect(document.body.textContent).not.toContain('Modeli küçült');
});

test('home shows the unsaved work section from the server drafts', async () => {
  const { api } = makeApi();
  api.drafts.mockResolvedValue([
    { sha256: 'a'.repeat(64), model_name: 'neshter/tiny-e2e', unsaved_count: 2, updated_at: 'x' },
  ] as never);
  render(<Home />);
  expect(await screen.findByText('Kaydedilmemiş çalışmalar: neshter/tiny-e2e · 2 kesim')).toBeInTheDocument();
});

test('home cards set uiStore.intent', async () => {
  makeApi();
  render(<Home />);
  const cases: [string | RegExp, Intent, 'button' | 'link'][] = [
    [/Bir davranışı azalt/, 'reduce', 'button'],
    [/İçine bak/, 'inspect', 'button'],
    ['Uzman modu', 'expert', 'link'],
  ];
  for (const [name, intent, role] of cases) {
    uiStore.setState({ route: 'home', intent: null });
    await userEvent.click(screen.getByRole(role, { name }));
    expect(uiStore.getState().intent).toBe(intent);
    expect(uiStore.getState().route).toBe('models');
  }
});

test('empty list shows sample download and starts the job', async () => {
  const { api, release } = makeApi({
    rows: [],
    results: {
      sample1: jobState({ id: 'sample1', kind: 'download', state: 'running', progress: 0.4, message: '200 / 500 MB indirildi', started_at: null }),
      load1: jobState({ id: 'load1' }),
    },
  });
  render(<Models />);
  await userEvent.click(await screen.findByRole('button', { name: 'Örnek modeli indir · Qwen2.5-0.5B' }));
  expect(api.sample).toHaveBeenCalledTimes(1);
  expect(await screen.findByText('200 / 500 MB indirildi')).toBeInTheDocument(); // the server's message, as is
  expect(api.loadModel).not.toHaveBeenCalled();
  api.job.mockImplementation(async (id: string) =>
    id === 'sample1' ? jobState({ id, kind: 'download', result: { path: 'C:\\snap\\abc' } }) : jobState({ id }),
  );
  await act(async () => release()); // stream ends, re-attach sees `done`
  await waitFor(() => expect(api.loadModel).toHaveBeenCalledTimes(1), { timeout: 3000 });
  expect(api.loadModel).toHaveBeenCalledWith({ path: 'C:\\snap\\abc', name: SAMPLE_NAME, dtype: 'auto' });
});

test('Aç sends path, name and dtype auto', async () => {
  const { api } = makeApi({ rows: [{ ...tiny, path: 'C:\\m\\snap', name: 'neshter/tiny-e2e' }] });
  render(<Models />);
  await userEvent.click(await rowOpen());
  await waitFor(() => expect(api.loadModel).toHaveBeenCalled());
  expect(api.loadModel).toHaveBeenCalledWith({ path: 'C:\\m\\snap', name: 'neshter/tiny-e2e', dtype: 'auto' });
});

test('the path field sends no name and rejects a relative path', async () => {
  const { api } = makeApi();
  render(<Models />);
  await screen.findByRole('list', { name: 'Modeller' });
  const input = screen.getByLabelText('Model klasörü (tam yol)');
  await userEvent.type(input, 'models\\x{Enter}');
  expect(await screen.findByRole('alert')).toHaveTextContent('Tam bir klasör yolu yazın');
  expect(api.loadModel).not.toHaveBeenCalled();
  await userEvent.clear(input);
  await userEvent.type(input, 'D:\\models\\x{Enter}');
  await waitFor(() => expect(api.loadModel).toHaveBeenCalled());
  expect(api.loadModel).toHaveBeenCalledWith({ path: 'D:\\models\\x', dtype: 'auto' });
});

test('opening a model while dirty asks the leave guard first', async () => {
  const { api } = makeApi();
  render(<Models />);
  const open = await rowOpen();
  act(() => {
    basketStore.getState().add([head(0, 1)]);
  });
  await userEvent.click(open);
  expect(await screen.findByText('Ayrılıyorsunuz, kesimlerinizi henüz kaydetmediniz')).toBeInTheDocument();
  expect(api.loadModel).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole('button', { name: 'Kaydetmeden çık' }));
  await waitFor(() => expect(api.loadModel).toHaveBeenCalledTimes(1));
});

test('cancelling the guard keeps the list and loads nothing', async () => {
  const { api } = makeApi();
  render(<Models />);
  const open = await rowOpen();
  act(() => {
    basketStore.getState().add([head(0, 1)]);
  });
  await userEvent.click(open);
  await userEvent.click(await screen.findByRole('button', { name: 'Vazgeç' }));
  expect(api.loadModel).not.toHaveBeenCalled();
  expect(screen.getByRole('list', { name: 'Modeller' })).toBeInTheDocument();
});

test.each<[Intent, string]>([
  ['reduce', 'concept'],
  ['inspect', 'explore'],
  ['expert', 'explore'],
  [null, 'explore'],
])('load done fills the session model and routes to the intent screen (%s -> %s)', async (intent, route) => {
  const { api } = makeApi();
  uiStore.setState({ intent });
  render(<Models />);
  await userEvent.click(await rowOpen());
  await waitFor(() => expect(uiStore.getState().route).toBe(route));
  expect(api.session).toHaveBeenCalled();
  expect(sessionStore.getState().model?.source_sha256).toBe(loaded.source_sha256);
});

test('loading a different model forgets the measurements and the explore view, the same model keeps them', async () => {
  makeApi();
  const stale = { ops_fp: 'fp1', kind: 'set_effect' } as Measurement;
  const fill = () => {
    measureStore.getState().put(stale);
    exploreStore.getState().setPrompt('merhaba');
    exploreStore.getState().setResult({ prompt: 'merhaba', tokens: ['a'], powerRaw: [[[1]]] });
  };
  sessionStore.getState().setModel({ ...loaded, source_sha256: 'b'.repeat(64) });
  fill();
  const { unmount } = render(<Models />);
  await userEvent.click(await rowOpen());
  await waitFor(() => expect(sessionStore.getState().model?.source_sha256).toBe(loaded.source_sha256));
  expect(measureStore.getState().byFp).toEqual({});
  expect(exploreStore.getState().result).toBeNull();
  expect(exploreStore.getState().prompt).toBe('');
  unmount();

  fill();
  uiStore.setState({ route: 'models' });
  render(<Models />);
  await userEvent.click(await rowOpen());
  await waitFor(() => expect(uiStore.getState().route).toBe('explore'));
  expect(measureStore.getState().get('fp1')).toBe(stale);
  expect(exploreStore.getState().prompt).toBe('merhaba');
});

test('a pending recipe is applied after the load, then the person moves on', async () => {
  const { api } = makeApi();
  const { recipeToItems, RecipeSchema } = await import('@/core/recipe');
  const recipe = RecipeSchema.parse({
    schema_version: 1,
    neshter_version: 't',
    source_model: { name: 'n', sha256: 'a'.repeat(64) },
    concept: null,
    method: 'ui',
    seed: 0,
    dataset_sha256: null,
    head_mode: 'zero',
    interventions: [{ type: 'head', layer: 0, head: 1 }],
    created_at: '2026-11-11T10:00:00.000Z',
  });
  expect(recipeToItems(recipe, 'x')).toHaveLength(1);
  pendingImportStore.getState().set(recipe);
  render(<Models />);
  await userEvent.click(await rowOpen());
  await waitFor(() => expect(uiStore.getState().route).toBe('explore'));
  expect(api.recipeCheck).toHaveBeenCalledTimes(1);
  expect(basketStore.getState().items).toHaveLength(1);
});

test('loading a different model drops the old basket, the same model keeps it', async () => {
  makeApi();
  sessionStore.getState().setModel({ ...loaded, source_sha256: 'b'.repeat(64) });
  basketStore.getState().add([head(0, 1)]);
  const { unmount } = render(<Models />);
  await userEvent.click(await rowOpen()); // dirty: the guard asks
  await userEvent.click(await screen.findByRole('button', { name: 'Kaydetmeden çık' }));
  await waitFor(() => expect(sessionStore.getState().model?.source_sha256).toBe(loaded.source_sha256));
  expect(basketStore.getState().items).toHaveLength(0);
  unmount();

  basketStore.getState().add([head(0, 2)]);
  uiStore.setState({ route: 'models' });
  render(<Models />);
  await userEvent.click(await rowOpen()); // dirty: the guard asks
  await userEvent.click(await screen.findByRole('button', { name: 'Kaydetmeden çık' }));
  await waitFor(() => expect(uiStore.getState().route).toBe('explore'));
  expect(basketStore.getState().items).toHaveLength(1);
});

test('loading a different model resets the save flow (no stale Kaydedildi card), the same model keeps it', async () => {
  makeApi();
  const done: SaveFlowState = {
    kind: 'done',
    result: { name: 'a', path: '/tmp/x', format: 'safetensors', source_sha256: 'b'.repeat(64) } as never,
  };
  saveFlowStore.setState({ state: done });
  sessionStore.getState().setModel({ ...loaded, source_sha256: 'b'.repeat(64) });
  const { unmount } = render(<Models />);
  await userEvent.click(await rowOpen());
  await waitFor(() => expect(sessionStore.getState().model?.source_sha256).toBe(loaded.source_sha256));
  expect(saveFlowStore.getState().state).toEqual(IDLE);
  unmount();

  saveFlowStore.setState({ state: done });
  uiStore.setState({ route: 'models' });
  render(<Models />);
  await userEvent.click(await rowOpen());
  await waitFor(() => expect(uiStore.getState().route).toBe('explore'));
  expect(saveFlowStore.getState().state).toBe(done);
});

test('a failed load shows the error by code and a way back', async () => {
  const { api } = makeApi({
    results: { load1: jobState({ id: 'load1', state: 'error', error: 'CUDA out of memory', error_code: 'oom' }) },
  });
  render(<Models />);
  await userEvent.click(await rowOpen());
  expect(await screen.findByText('Bellek yetmedi. Daha küçük bir model seçin.')).toBeInTheDocument();
  expect(screen.getByText('CUDA out of memory')).toBeInTheDocument();
  expect(sessionStore.getState().model).toBeNull();
  await userEvent.click(screen.getByRole('button', { name: 'Modellere dön' }));
  expect(await screen.findByRole('list', { name: 'Modeller' })).toBeInTheDocument();
  expect(api.loadModel).toHaveBeenCalledTimes(1);
});

test('a folder that does not exist says so, under a heading that is no longer "loading"', async () => {
  const { api } = makeApi();
  api.loadModel = vi.fn(async () => {
    throw new ApiError(404, 'not_found', "'D:\\yok' bulunamadı.");
  });
  render(<Models />);
  await screen.findByRole('list', { name: 'Modeller' });
  await userEvent.type(screen.getByLabelText('Model klasörü (tam yol)'), 'D:\\yok{Enter}');
  expect(await screen.findByRole('heading', { level: 1, name: 'Model açılamadı' })).toBeInTheDocument();
  expect(screen.getByText('Bu klasör ya da dosya bulunamadı; yolu kontrol edip yeniden deneyin.')).toBeInTheDocument();
  expect(screen.queryByRole('heading', { name: 'Model yükleniyor' })).toBeNull();
});

test('İptal cancels the running load job', async () => {
  const { api } = makeApi({
    results: { load1: jobState({ id: 'load1', state: 'running', progress: 0.3, message: 'Model yükleniyor' }) },
  });
  render(<Models />);
  await userEvent.click(await rowOpen());
  await userEvent.click(await screen.findByRole('button', { name: 'İptal' }));
  expect(api.cancelJob).toHaveBeenCalledWith('load1');
});
