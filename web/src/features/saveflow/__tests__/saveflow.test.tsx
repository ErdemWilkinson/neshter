import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { Mock } from 'vitest';
import type { Api } from '@/api/client';
import { ApiError } from '@/api/errors';
import { setApi } from '@/api/currentApi';
import jobSuite from '@/api/__fixtures__/job_suite.json';
import { getSessionStorage } from '@/app/sessionStorage';
import { IDLE } from '@/core/saveflow';
import { basketStore } from '@/core/stores/basket';
import { compareStore } from '@/core/stores/compare';
import { jobsStore } from '@/core/stores/jobs';
import { uiStore } from '@/core/stores/ui';
import { registerFakeExploreOp } from '@/core/__tests__/fixtures/fakeExploreOp';
import { head, item } from '@/core/__tests__/fixtures/items';
import { setLang } from '@/i18n';
import { DiscardDialog } from '../DiscardDialog';
import { NothingToSaveDialog } from '../NothingToSaveDialog';
import { SaveFlowNotice } from '../SaveFlowNotice';
import { saveFlowStore } from '../saveFlowStore';
import { useSaveFlow } from '../useSaveFlow';

beforeAll(registerFakeExploreOp);

const explore = () => item({ type: 'test_explore', src: 'h:0:1' });
const pair = { original: 'a', modified: 'b', mode: 'batch', target_hit: null };
const doneJob = { ...jobSuite, id: 'j1', kind: 'suite', error: null, error_code: null, started_at: null };

function Harness() {
  const flow = useSaveFlow();
  const ids = flow.state.kind === 'needsDiscard' ? flow.state.exploreIds : [];
  return (
    <div>
      <button type="button" onClick={flow.saveClicked}>
        Kaydet
      </button>
      <output>{flow.state.kind}</output>
      {flow.error !== null && <SaveFlowNotice error={flow.error} />}
      <DiscardDialog open={flow.state.kind === 'needsDiscard'} exploreIds={ids} onConfirm={flow.confirmDiscard} onCancel={flow.cancel} />
      <NothingToSaveDialog open={flow.state.kind === 'nothingToSave'} onClose={flow.dismiss} />
      <button type="button" onClick={flow.cancel}>
        iptal et
      </button>
    </div>
  );
}

function mockApi(patch: Partial<Record<keyof Api, unknown>> = {}) {
  const order: string[] = [];
  const api = {
    concepts: vi.fn(async () => []),
    compare: vi.fn(async () => {
      order.push('compare');
      return pair;
    }),
    suite: vi.fn(async () => {
      order.push('suite');
      return { job_id: 'j1' };
    }),
    job: vi.fn(async () => doneJob),
    streamJob: vi.fn(async () => 'ended'),
    cancelJob: vi.fn(async () => ({ ...doneJob, state: 'cancelled', result: null })),
    exportModel: vi.fn(),
    ...patch,
  };
  setApi(api as unknown as Api);
  return { api: api as unknown as Record<'concepts' | 'compare' | 'suite' | 'job' | 'streamJob' | 'cancelJob' | 'exportModel', Mock>, order };
}

const body = (call: unknown[] | undefined) => call?.[0] as { recipe_ops: unknown[]; quick?: boolean; prompt?: string };

beforeEach(() => {
  setLang('tr');
  getSessionStorage().clear();
  window.location.hash = '#/';
  uiStore.setState({ route: 'save' });
  basketStore.getState().add([head(0, 1), head(0, 2), explore()]);
});
afterEach(() => {
  setApi(null);
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
  compareStore.setState({ lastPrompt: null, lastPair: null, lastSuite: null, refreshedDropped: null });
  saveFlowStore.setState({ state: IDLE, error: null });
  jobsStore.setState({ active: null });
  uiStore.setState({ route: 'home' });
});

test('saveflow with a fake explore op: discard → suite with savable ops only → shelf + refreshed banner, no second modal', async () => {
  compareStore.getState().setLastPrompt('merhaba');
  const { api } = mockApi();
  render(<Harness />);
  await userEvent.click(screen.getByRole('button', { name: 'Kaydet' }));
  const dialog = await screen.findByRole('dialog');
  expect(dialog).toHaveTextContent('Şu 1 kesim yalnızca keşif içindir ve kaydedilemez: test_explore. Bunlar sepetten atılacak, karşılaştırma yeniden koşulacak. Devam?');
  await userEvent.click(screen.getByRole('button', { name: 'Devam' }));

  await waitFor(() => expect(uiStore.getState().route).toBe('compare'));
  expect(body(api.suite.mock.calls[0])).toMatchObject({ quick: true });
  expect(body(api.suite.mock.calls[0]).recipe_ops).toEqual([
    { type: 'head', layer: 0, head: 1 },
    { type: 'head', layer: 0, head: 2 },
  ]);
  expect(body(api.compare.mock.calls[0]).recipe_ops).toHaveLength(2); // pending-discard ops are not sent
  const { items, shelf } = basketStore.getState();
  expect(items).toHaveLength(2);
  expect(items.some((i) => i.pendingDiscard)).toBe(false);
  expect(shelf).toHaveLength(1);
  expect(compareStore.getState().refreshedDropped).toBe(1);
  expect(compareStore.getState().lastPair).toEqual(pair);
  expect(compareStore.getState().lastSuite?.result.categories.length).toBeGreaterThan(0);
  expect(saveFlowStore.getState().state).toEqual({ kind: 'reviewReady', dropped: 1 });
  expect(screen.getAllByRole('dialog', { hidden: true }).every((d) => !d.hasAttribute('open'))).toBe(true); // no second modal
});

test('saveflow re-runs compare before starting the suite job and stores the job id', async () => {
  compareStore.getState().setLastPrompt('merhaba');
  const seen: unknown[] = [];
  let release: (() => void) | undefined;
  const { api, order } = mockApi({
    job: vi.fn(async () => {
      seen.push(saveFlowStore.getState().state);
      await new Promise<void>((resolve) => (release = resolve));
      return doneJob;
    }),
  });
  render(<Harness />);
  await userEvent.click(screen.getByRole('button', { name: 'Kaydet' }));
  await userEvent.click(await screen.findByRole('button', { name: 'Devam' }));
  await waitFor(() => expect(release).toBeDefined());
  expect(order).toEqual(['compare', 'suite']);
  expect(saveFlowStore.getState().state).toMatchObject({ kind: 'recomparing', jobId: 'j1' });
  expect(seen[0]).toMatchObject({ kind: 'recomparing', jobId: 'j1' });
  expect(basketStore.getState().items.filter((i) => i.pendingDiscard)).toHaveLength(1);
  expect(basketStore.getState().shelf).toHaveLength(0); // nothing moves before the job is done
  expect(api.suite).toHaveBeenCalledTimes(1);
  release?.();
  await waitFor(() => expect(basketStore.getState().shelf).toHaveLength(1));
});

test('without a compare prompt only the suite job runs', async () => {
  const { api } = mockApi();
  render(<Harness />);
  await userEvent.click(screen.getByRole('button', { name: 'Kaydet' }));
  await userEvent.click(await screen.findByRole('button', { name: 'Devam' }));
  await waitFor(() => expect(uiStore.getState().route).toBe('compare'));
  expect(api.compare).not.toHaveBeenCalled();
});

test('saveflow: cancelling the re-run leaves the basket unchanged', async () => {
  let onEvent: ((event: string, data: unknown) => void) | undefined;
  const { api } = mockApi({
    job: vi.fn(async () => ({ ...doneJob, state: 'running', result: null })),
    streamJob: vi.fn(
      (_id: string, opts: { onEvent: typeof onEvent }) =>
        new Promise((resolve) => {
          onEvent = opts.onEvent;
          api_resolve = () => resolve('ended');
        }),
    ),
    cancelJob: vi.fn(async () => {
      onEvent?.('state', { ...doneJob, state: 'cancelled', result: null });
      api_resolve?.();
      return { ...doneJob, state: 'cancelled', result: null };
    }),
  });
  let api_resolve: (() => void) | undefined;
  const before = basketStore.getState().items;
  render(<Harness />);
  await userEvent.click(screen.getByRole('button', { name: 'Kaydet' }));
  await userEvent.click(await screen.findByRole('button', { name: 'Devam' }));
  await waitFor(() => expect(saveFlowStore.getState().state).toMatchObject({ kind: 'recomparing', jobId: 'j1' }));
  await userEvent.click(screen.getByRole('button', { name: 'iptal et' }));
  await waitFor(() => expect(api.cancelJob).toHaveBeenCalledWith('j1'));
  await waitFor(() => expect(basketStore.getState().items.some((i) => i.pendingDiscard)).toBe(false));
  expect(basketStore.getState().items.map((i) => i.id)).toEqual(before.map((i) => i.id));
  expect(basketStore.getState().shelf).toHaveLength(0);
  expect(saveFlowStore.getState().state).toEqual(IDLE);
  expect(uiStore.getState().route).toBe('save');
});

test('saveflow: a failed compare (busy) restores the basket and reports the error', async () => {
  compareStore.getState().setLastPrompt('merhaba');
  const { api } = mockApi({ compare: vi.fn(async () => Promise.reject(new ApiError(409, 'busy', 'x'))) });
  render(<Harness />);
  await userEvent.click(screen.getByRole('button', { name: 'Kaydet' }));
  await userEvent.click(await screen.findByRole('button', { name: 'Devam' }));
  expect(await screen.findByText('Model meşgul: çalışan işin bitmesini bekleyin.')).toBeInTheDocument();
  expect(screen.getByText('Sayılar yenilenemedi; sepetiniz değişmedi.')).toBeInTheDocument();
  expect(api.suite).not.toHaveBeenCalled();
  expect(basketStore.getState().items.some((i) => i.pendingDiscard)).toBe(false);
  expect(basketStore.getState().shelf).toHaveLength(0);
  expect(saveFlowStore.getState().state).toEqual(IDLE);
});

test('saveflow: a failed suite job restores the basket', async () => {
  const { api } = mockApi({ job: vi.fn(async () => ({ ...doneJob, state: 'error', result: null, error: 'x', error_code: 'internal' })) });
  render(<Harness />);
  await userEvent.click(screen.getByRole('button', { name: 'Kaydet' }));
  await userEvent.click(await screen.findByRole('button', { name: 'Devam' }));
  expect(await screen.findByText('Beklenmeyen bir hata oldu.')).toBeInTheDocument();
  expect(api.suite).toHaveBeenCalledTimes(1);
  expect(basketStore.getState().items.some((i) => i.pendingDiscard)).toBe(false);
  expect(basketStore.getState().shelf).toHaveLength(0);
});

test('saveflow: only explore ops → nothing-to-save dialog and no export call', async () => {
  basketStore.getState().replaceAll({ items: [explore()], shelf: [], savedOps: [] });
  const { api } = mockApi();
  render(<Harness />);
  await userEvent.click(screen.getByRole('button', { name: 'Kaydet' }));
  expect(await screen.findByText('Kaydedilecek kesim kalmıyor.')).toBeInTheDocument();
  expect(screen.getByRole('dialog', { name: 'Kaydedilecek kesim yok' })).toBeInTheDocument();
  expect(api.exportModel).not.toHaveBeenCalled();
  expect(api.suite).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole('button', { name: 'Tamam' }));
  await act(async () => undefined);
  expect(saveFlowStore.getState().state).toEqual(IDLE);
});

test('an opaque explore item is listed as desteklenmeyen kesim', async () => {
  basketStore.getState().add([item({ type: 'edge_read_v9' })]);
  mockApi();
  render(<Harness />);
  await userEvent.click(screen.getByRole('button', { name: 'Kaydet' }));
  expect(await screen.findByRole('dialog')).toHaveTextContent(/Şu 2 kesim .*: test_explore, desteklenmeyen kesim\./);
});
