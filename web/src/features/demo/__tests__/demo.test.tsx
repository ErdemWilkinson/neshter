import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { sessionLoaded } from '@/app/__tests__/fakeServer';
import type { Api } from '@/api/client';
import { setApi } from '@/api/currentApi';
import type { EffectResult, JobState } from '@/api/schemas';
import { headId } from '@/core/ids';
import { notSignificantEffect } from '@/core/__tests__/fixtures/futilityEffect';
import { jobsStore } from '@/core/stores/jobs';
import { measureStore } from '@/core/stores/measure';
import { selectionStore } from '@/core/stores/selection';
import { sessionStore } from '@/core/stores/session';
import { uiStore } from '@/core/stores/ui';
import { exploreStore } from '@/features/map/exploreStore';
import { MeasureButton } from '@/features/measure/MeasureButton';
import { setLang } from '@/i18n';
import { Explore } from '@/screens/Explore';
import { Home } from '@/screens/Home';
import { DEMO_PROMPT, demoStore } from '../demoStore';

// 2 layers x 3 heads, one token: over the whole map the strongest are L1·H2 (9), then L0·H1 (7).
const POWER_RAW = [[[1], [7], [2]], [[3], [0], [9]]];

const doneJob = (result: EffectResult): JobState<EffectResult> => ({
  id: 'j1',
  kind: 'effect',
  state: 'done',
  progress: 1,
  message: '',
  result,
  error: null,
  error_code: null,
  started_at: null,
});

function fakeApi() {
  const api = {
    activity: vi.fn(async () => ({ tokens: [' is'], power_raw: POWER_RAW })),
    mlp: vi.fn(async () => []),
    effect: vi.fn(async () => ({ job_id: 'j1' })),
    job: vi.fn(async () => doneJob(notSignificantEffect)),
    streamJob: vi.fn(async () => 'ended' as const),
    cancelJob: vi.fn(),
    drafts: vi.fn(async () => []),
  };
  setApi(api as unknown as Api);
  return api;
}

beforeEach(() => {
  setLang('en');
  uiStore.setState({ view: 'list' });
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
});
afterEach(() => {
  cleanup();
  setApi(null);
  demoStore.getState().stop();
  jobsStore.setState({ active: null });
  measureStore.getState().reset();
  selectionStore.setState({ selected: new Set(), focus: null });
  sessionStore.setState({ model: null, busy: false });
  exploreStore.getState().reset();
  uiStore.setState({ route: 'home', intent: null });
  vi.restoreAllMocks();
});

test('the example card starts the example on the way to Models', async () => {
  fakeApi();
  render(<Home />);
  await userEvent.click(screen.getByRole('button', { name: /Try the example/ }));
  expect(demoStore.getState()).toMatchObject({ step: 'load', shown: true });
  expect(uiStore.getState().intent).toBe('inspect');
  expect(uiStore.getState().route).toBe('models');
});

test('another card ends an example that was started before', async () => {
  fakeApi();
  demoStore.getState().start();
  render(<Home />);
  await userEvent.click(screen.getByRole('button', { name: /Look inside/ }));
  expect(demoStore.getState().step).toBe('idle');
});

test('Explore runs the example text once, picks the two strongest heads and measures them', async () => {
  const api = fakeApi();
  sessionStore.getState().setModel({ ...sessionLoaded, device: 'cpu' } as never);
  demoStore.getState().setStep('run');
  act(() => demoStore.setState({ shown: true }));
  render(<Explore />);
  await waitFor(() => expect(api.effect).toHaveBeenCalledTimes(1));
  expect(api.effect).toHaveBeenCalledWith(
    expect.objectContaining({ heads: [[0, 1], [1, 2]], probe: DEMO_PROMPT }),
  );
  expect(demoStore.getState().step).toBe('idle');
  expect(api.activity).toHaveBeenCalledTimes(1);
  expect(api.activity).toHaveBeenCalledWith({ prompt: DEMO_PROMPT });
  expect(exploreStore.getState().prompt).toBe(DEMO_PROMPT);
  expect([...selectionStore.getState().selected]).toEqual([headId(1, 2), headId(0, 1)]);
  expect(screen.getByText(/The 2 strongest heads/)).toBeInTheDocument();
});

test('typing their own text hides the example note', async () => {
  fakeApi();
  sessionStore.getState().setModel(sessionLoaded as never);
  demoStore.setState({ step: 'idle', shown: true });
  render(<Explore />);
  expect(screen.getByText(/The 2 strongest heads/)).toBeInTheDocument();
  await userEvent.type(screen.getByLabelText('Text'), 'x');
  expect(screen.queryByText(/The 2 strongest heads/)).toBeNull();
});

test('a failed run ends the example instead of waiting forever', async () => {
  const api = fakeApi();
  api.activity.mockRejectedValue(new Error('boom'));
  sessionStore.getState().setModel(sessionLoaded as never);
  demoStore.getState().setStep('run');
  render(<Explore />);
  await waitFor(() => expect(demoStore.getState().step).toBe('idle'));
});

test('the measure step starts the measurement without a click, once', async () => {
  const api = fakeApi();
  sessionStore.getState().setModel({ ...sessionLoaded, device: 'cpu' } as never);
  demoStore.getState().setStep('measure');
  const ops = [{ type: 'head' as const, layer: 1, head: 2 }];
  const { rerender } = render(<MeasureButton ops={ops} />);
  await waitFor(() => expect(api.effect).toHaveBeenCalledTimes(1));
  expect(demoStore.getState().step).toBe('idle');
  rerender(<MeasureButton ops={[...ops]} />);
  await act(async () => {});
  expect(api.effect).toHaveBeenCalledTimes(1);
});

test('the measure button waits while nothing is picked', async () => {
  const api = fakeApi();
  sessionStore.getState().setModel({ ...sessionLoaded, device: 'cpu' } as never);
  demoStore.getState().setStep('measure');
  render(<MeasureButton ops={[]} />);
  await act(async () => {});
  expect(api.effect).not.toHaveBeenCalled();
  expect(demoStore.getState().step).toBe('measure');
});
