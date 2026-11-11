import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { sessionLoaded } from '@/app/__tests__/fakeServer';
import type { Api } from '@/api/client';
import { setApi } from '@/api/currentApi';
import type { EffectResult, JobState } from '@/api/schemas';
import { fromEffectResult, type Measurement } from '@/core/measurement';
import { opsFingerprint } from '@/core/fingerprint';
import { basketStore } from '@/core/stores/basket';
import { jobsStore } from '@/core/stores/jobs';
import { measureStore } from '@/core/stores/measure';
import { selectionStore } from '@/core/stores/selection';
import { sessionStore } from '@/core/stores/session';
import { headId } from '@/core/ids';
import { futilityEffect, notSignificantEffect, significantEffect } from '@/core/__tests__/fixtures/futilityEffect';
import { Header } from '@/app/Header';
import { CutButton } from '@/features/basket/CutButton';
import { setLang, t } from '@/i18n';
import { ControlStatsCard } from '../ControlStatsCard';
import { nMaxBounds } from '../nMax';
import { SelectionPanel } from '../SelectionPanel';

const NOW = '2026-11-11T10:00:00.000Z';
const ops = (...heads: [number, number][]) => heads.map(([layer, head]) => ({ type: 'head' as const, layer, head }));
const measurementFor = (patch: Partial<EffectResult>, selection = ops([3, 5])): Measurement =>
  fromEffectResult({ ...notSignificantEffect, ...patch }, opsFingerprint(selection), NOW);

function doneJob(result: EffectResult): JobState<EffectResult> {
  return {
    id: 'j1',
    kind: 'effect',
    state: 'done',
    progress: 1,
    message: '',
    result,
    error: null,
    error_code: null,
    started_at: null,
  };
}

function fakeApi(result: unknown = notSignificantEffect) {
  const api = {
    effect: vi.fn(async () => ({ job_id: 'j1' })),
    job: vi.fn(async () => doneJob(result as EffectResult)),
    streamJob: vi.fn(async () => 'ended' as const),
    cancelJob: vi.fn(),
  };
  setApi(api as unknown as Api);
  return api;
}

const select = (...heads: [number, number][]) =>
  act(() => selectionStore.getState().set(heads.map(([l, h]) => headId(l, h))));

beforeEach(() => {
  setLang('tr');
  sessionStore.getState().setModel({ ...sessionLoaded, device: 'cpu' } as never);
});
afterEach(() => {
  cleanup();
  setApi(null);
  jobsStore.setState({ active: null });
  measureStore.getState().reset();
  selectionStore.setState({ selected: new Set(), focus: null });
  sessionStore.setState({ model: null, busy: false });
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
});

test('exact cut copy', () => {
  setLang('tr');
  expect(t('cut.button')).toBe('Kes (geçici)');
  expect(t('cut.sub')).toBe('Dosya değişmez');
  setLang('en');
  expect(t('cut.button')).toBe('Cut (temporary)');
});

test('nMaxBounds caps cpu at 199 and other devices at 999', () => {
  expect(nMaxBounds('cpu')).toEqual({ min: 19, max: 199, default: 39 });
  expect(nMaxBounds('cuda')).toEqual({ min: 19, max: 999, default: 39 });
});

test('cut disabled before measuring, enabled after, disabled again on selection change', async () => {
  const user = userEvent.setup();
  fakeApi();
  render(<SelectionPanel values={null} />);
  select([3, 5]);
  const cut = () => screen.getByRole('button', { name: /Kes \(geçici\)/ });
  expect((cut() as HTMLButtonElement).disabled).toBe(true);
  expect(screen.getByText('Önce gerçek etkiyi ölç')).toBeTruthy();
  expect(cut().getAttribute('aria-describedby')).toBe(screen.getByText('Önce gerçek etkiyi ölç').id);

  await user.click(screen.getByRole('button', { name: 'Gerçek etkiyi ölç' }));
  await waitFor(() => expect((cut() as HTMLButtonElement).disabled).toBe(false));
  expect(screen.getByText('Dosya değişmez')).toBeTruthy();

  select([3, 5], [4, 1]);
  expect((cut() as HTMLButtonElement).disabled).toBe(true);
});

test('measure sends n_max 39 by default and heads from the selection ids', async () => {
  const user = userEvent.setup();
  const api = fakeApi();
  render(<SelectionPanel values={null} />);
  select([4, 1], [3, 5]);
  await user.click(screen.getByRole('button', { name: 'Gerçek etkiyi ölç' }));
  await waitFor(() => expect(api.effect).toHaveBeenCalledTimes(1));
  expect(api.effect).toHaveBeenCalledWith({ heads: [[3, 5], [4, 1]], n_max: 39, seed: 0 });
  await screen.findByText('ölçülen');
  expect(sessionStore.getState().model?.measure_count).toBe(3);
});

test('cpu caps n_max input at 199 and shows p_min', async () => {
  const user = userEvent.setup();
  const api = fakeApi();
  render(<SelectionPanel values={null} />);
  select([3, 5]);
  await user.click(screen.getByText('Gelişmiş'));
  const input = screen.getByLabelText('Rastgele çekiş sayısı (en çok)') as HTMLInputElement;
  expect(input.max).toBe('199');
  expect(input.min).toBe('19');
  expect(screen.getByText('en küçük p = 0,025')).toBeTruthy();
  await user.clear(input);
  await user.type(input, '500');
  expect(screen.getByText('en küçük p = 0,005')).toBeTruthy();
  await user.click(screen.getByRole('button', { name: 'Gerçek etkiyi ölç' }));
  await waitFor(() => expect(api.effect).toHaveBeenCalled());
  expect(api.effect).toHaveBeenCalledWith(expect.objectContaining({ n_max: 199 }));
});

test('a malformed effect result is an error card, never a measurement', async () => {
  const user = userEvent.setup();
  fakeApi({ kl: 'x' });
  render(<SelectionPanel values={null} />);
  select([3, 5]);
  await user.click(screen.getByRole('button', { name: 'Gerçek etkiyi ölç' }));
  await screen.findByRole('alert');
  expect(measureStore.getState().byFp).toEqual({});
});

test('selection chips are removable and the zero-ablation sentence is shown', async () => {
  const user = userEvent.setup();
  fakeApi();
  render(<SelectionPanel values={[[0, 0], [0, 0], [0, 0], [0, 0, 0, 0, 0, 0.82]]} />);
  select([3, 5], [4, 1]);
  act(() => selectionStore.getState().setFocus(headId(3, 5)));
  expect(screen.getByText("Head'in çıktısı sıfırlanır (o_proj sütunları sıfır).")).toBeTruthy();
  expect(screen.getByText('Odaktaki head K3·H5: güç 0,82')).toBeTruthy();
  const list = screen.getByRole('list', { name: "Seçili head'ler" });
  expect(within(list).getByText('K3·H5')).toBeTruthy();
  await user.click(within(list).getByRole('button', { name: 'K3·H5 seçimden çıkar' }));
  expect([...selectionStore.getState().selected]).toEqual([headId(4, 1)]);
});

test('ControlStatsCard significant copy', () => {
  render(<ControlStatsCard m={measurementFor(significantEffect)} />);
  expect(
    screen.getByText("Seçilen head'ler, aynı katmanlardan rastgele seçilen head'lerden daha etkili (p = 0,025, 39 çekiş)."),
  ).toBeTruthy();
  expect(screen.getByText('KL 0,0021 ± 0,0003 · 2 istem')).toBeTruthy();
  expect(screen.getByText('ölçülen')).toBeTruthy();
  expect(screen.queryByText(/rastgele \(yaklaşık\)/)).toBeNull(); // no band in this payload
});

test('ControlStatsCard full not-significant copy and band', () => {
  render(<ControlStatsCard m={measurementFor(notSignificantEffect)} />);
  expect(screen.getByText('p = 0,525 > 0,05: rastgele seçimden ayırt edilemedi (39 çekiş).')).toBeTruthy();
  expect(screen.getByText('rastgele (yaklaşık) 0,0011–0,0038')).toBeTruthy();
});

test('ControlStatsCard futility copy uses tr number format', () => {
  setLang('tr');
  render(<ControlStatsCard m={measurementFor({ ...futilityEffect, n_drawn: 3, k_exceed: 2, tau: 0.05 })} />);
  expect(screen.getByText('p > 0,05, 3 çekişte durdu (2 aşım): rastgele seçimden ayırt edilemedi.')).toBeTruthy();
});

test('ControlStatsCard keeps seed and p_kind inside the technical detail', () => {
  render(<ControlStatsCard m={measurementFor(futilityEffect)} />);
  const detail = screen.getByText('Teknik ayrıntı').closest('details');
  expect(detail?.textContent).toContain('tohum 7');
  expect(detail?.textContent).toContain('besag_clifford');
});

test('session hypothesis line only from the second measurement', () => {
  const line = 'Bu oturumda 2 ölçüm yapıldı; seçilerek bulunan bir sonuç hipotezdir.';
  const { rerender } = render(<ControlStatsCard m={measurementFor({ session_measure_index: 1 })} />);
  expect(screen.queryByText(/Bu oturumda/)).toBeNull();
  rerender(<ControlStatsCard m={measurementFor({ session_measure_index: 2 })} />);
  expect(screen.getByText(line)).toBeTruthy();
});

test('cutting two heads adds two items and the header shows Kaydedilmemiş 2 kesim', async () => {
  const user = userEvent.setup();
  const selection = ops([3, 5], [4, 1]);
  measureStore.getState().put(measurementFor({}, selection));
  render(
    <>
      <Header onClose={() => undefined} />
      <CutButton selection={selection} />
    </>,
  );
  await user.click(screen.getByRole('button', { name: /Kes \(geçici\)/ }));
  const items = basketStore.getState().items;
  expect(items).toHaveLength(2);
  expect(items.every((i) => i.semantics === 'zero' && i.origin === 'heatmap' && i.concept === null)).toBe(true);
  expect(items[0]?.measurement).toBe(items[1]?.measurement);
  expect(screen.getByText('Kaydedilmemiş 2 kesim')).toBeTruthy();
});

test('the same head is not added twice', async () => {
  const user = userEvent.setup();
  const selection = ops([3, 5]);
  measureStore.getState().put(measurementFor({}, selection));
  render(<CutButton selection={selection} />);
  await user.click(screen.getByRole('button', { name: /Kes \(geçici\)/ }));
  await user.click(screen.getByRole('button', { name: /Kes \(geçici\)/ }));
  expect(basketStore.getState().items).toHaveLength(1);
});
