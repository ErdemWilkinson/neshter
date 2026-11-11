import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { Api } from '@/api/client';
import { setApi } from '@/api/currentApi';
import { ApiError, BusyError } from '@/api/errors';
import type { JobState } from '@/api/schemas';
import { basketStore } from '@/core/stores/basket';
import { jobsStore } from '@/core/stores/jobs';
import { selectionStore } from '@/core/stores/selection';
import { sessionStore } from '@/core/stores/session';
import { uiStore } from '@/core/stores/ui';
import { head } from '@/core/__tests__/fixtures/items';
import { exploreStore } from '@/features/map/exploreStore';
import { VIEWS } from '@/features/map/ViewSwitch';
import { setLang } from '@/i18n';
import { Explore } from '../Explore';

// 2 layers x 2 heads x 2 tokens; at the last token both layers read [0, 1] (min-max per layer), at the first [1, 0] and [0, 1].
const POWER_RAW = [
  [
    [1, 0],
    [0, 1],
  ],
  [
    [2, 5],
    [4, 6],
  ],
];
const TOKENS = ['<b>hi', ' dün'];

function makeApi(activity: () => Promise<unknown> = async () => ({ tokens: TOKENS, power_raw: POWER_RAW })) {
  const api = {
    activity: vi.fn(activity),
    mlp: vi.fn(async () => [
      { layer: 0, neuron: 98, value: 0.0822 },
      { layer: 1, neuron: 84, value: 0.0471 },
    ]),
  };
  setApi(api as unknown as Api);
  return api;
}

const runningJob = { id: 'j', kind: 'effect', state: 'running', progress: 0.1 } as unknown as JobState;

async function run(text = 'merhaba dün') {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText('Metin'), text);
  await user.click(screen.getByRole('button', { name: 'Çalıştır' }));
  await screen.findByRole('list', { name: 'Belirteçler' });
  return user;
}

/** The power column of the Liste view in row order. */
const powers = () =>
  screen
    .getAllByRole('row')
    .slice(1)
    .map((row) => row.querySelectorAll('td')[2]?.textContent);

beforeEach(() => {
  setLang('tr');
  uiStore.setState({ view: 'list' });
  exploreStore.getState().reset();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
});
afterEach(() => {
  cleanup(); // unmount before the stores are reset, or the mounted heatmap repaints with the restored (real) canvas
  setApi(null);
  jobsStore.setState({ active: null });
  sessionStore.setState({ busy: false });
  selectionStore.setState({ selected: new Set(), focus: null });
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
  vi.restoreAllMocks();
});

test('run button label is Çalıştır', () => {
  setLang('tr');
  makeApi();
  render(<Explore />);
  expect(screen.getByRole('button', { name: 'Çalıştır' })).toBeTruthy();
});

test('run button label follows the language', () => {
  setLang('en');
  makeApi();
  render(<Explore />);
  expect(screen.getByRole('button', { name: 'Run' })).toBeTruthy();
});

test('run sends the prompt once without a token index, then shows tokens, legend and the estimated badge', async () => {
  const api = makeApi();
  render(<Explore />);
  await run();
  expect(api.activity).toHaveBeenCalledTimes(1);
  expect(api.activity).toHaveBeenCalledWith({ prompt: 'merhaba dün' });
  expect(screen.getByText('Head çıktı gücü, katman içinde 0–1')).toBeInTheDocument();
  expect(screen.getByText('tahmini')).toBeInTheDocument();
  // default: the last token is chosen
  const chips = screen.getAllByRole('button', { pressed: true });
  expect(chips).toHaveLength(1);
  expect(chips[0]).toHaveTextContent('dün');
});

test('token text with markup is a text node, never an element', async () => {
  makeApi();
  render(<Explore />);
  await run();
  expect(screen.getByText('<b>hi')).toBeInTheDocument();
  expect(document.querySelector('b')).toBeNull();
});

test('token click re-slices without a request', async () => {
  const api = makeApi();
  render(<Explore />);
  const user = await run();
  expect(powers()).toEqual(['0,00', '1,00', '0,00', '1,00']); // last token, per layer
  await user.click(screen.getByRole('button', { name: '<b>hi' }));
  expect(powers()).toEqual(['1,00', '0,00', '0,00', '1,00']);
  expect(api.activity).toHaveBeenCalledTimes(1);
  expect(api.mlp).not.toHaveBeenCalled();
});

test('scale toggle switches normalisation without a request', async () => {
  const api = makeApi(async () => ({ tokens: ['a'], power_raw: [[[2], [4]], [[10], [20]]] }));
  render(<Explore />);
  const user = await run();
  expect(powers()).toEqual(['0,00', '1,00', '0,00', '1,00']);
  await user.click(screen.getByRole('radio', { name: 'Tüm model' }));
  expect(powers()).toEqual(['0,00', '0,11', '0,44', '1,00']);
  expect(screen.getByText('Head çıktı gücü, tüm modelde 0–1')).toBeInTheDocument();
  await user.click(screen.getByRole('radio', { name: 'Katman içi' }));
  expect(powers()).toEqual(['0,00', '1,00', '0,00', '1,00']);
  expect(api.activity).toHaveBeenCalledTimes(1);
});

test('Liste selection reaches the same store the heatmap uses, and basket heads are marked', async () => {
  makeApi();
  basketStore.getState().add([head(1, 0)]);
  render(<Explore />);
  const user = await run();
  await user.click(screen.getByRole('checkbox', { name: 'K0 · H1 seç' }));
  expect(selectionStore.getState().selected.has('h:0:1' as never)).toBe(true);
  expect(screen.getAllByRole('row')[3]).toHaveTextContent('sepette');
});

test('run button and token chips are disabled while a job holds the model', async () => {
  makeApi();
  render(<Explore />);
  await run();
  act(() => jobsStore.setState({ active: runningJob }));
  expect(screen.getByRole('button', { name: 'Çalıştır' })).toBeDisabled();
  const chips = document.querySelectorAll('.token-chip');
  expect(chips).toHaveLength(2);
  chips.forEach((chip) => expect(chip).toBeDisabled());
});

test('a 409 sets the busy flag and shows no error card', async () => {
  makeApi(async () => {
    throw new BusyError('busy');
  });
  render(<Explore />);
  const user = userEvent.setup();
  await user.type(screen.getByLabelText('Metin'), 'x');
  await user.click(screen.getByRole('button', { name: 'Çalıştır' }));
  await waitFor(() => expect(sessionStore.getState().busy).toBe(true));
  expect(screen.queryByRole('alert')).toBeNull();
});

test('another API error shows the error card chosen by code', async () => {
  makeApi(async () => {
    throw new ApiError(409, 'no_model', 'detail text');
  });
  render(<Explore />);
  const user = userEvent.setup();
  await user.type(screen.getByLabelText('Metin'), 'x');
  await user.click(screen.getByRole('button', { name: 'Çalıştır' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Önce bir model yükleyin.');
});

test('a ragged payload becomes an error card, not a crash', async () => {
  makeApi(async () => ({ tokens: ['a'], power_raw: [[[1]], [[]]] }));
  render(<Explore />);
  const user = userEvent.setup();
  await user.type(screen.getByLabelText('Metin'), 'x');
  await user.click(screen.getByRole('button', { name: 'Çalıştır' }));
  expect(await screen.findByRole('alert')).toBeInTheDocument();
});

test('3d view is registered but not rendered', async () => {
  expect(VIEWS).toEqual([
    { id: '2d', enabled: true },
    { id: 'list', enabled: true },
    { id: '3d', enabled: false },
  ]);
  makeApi();
  uiStore.setState({ view: '3d' }); // e.g. a stale value: falls back to the heatmap
  render(<Explore />);
  await run();
  const view = screen.getByRole('group', { name: 'Görünüm' });
  expect(view.querySelectorAll('input[type="radio"]')).toHaveLength(2);
  expect(screen.queryByText('3B')).toBeNull();
  expect(screen.getByRole('radio', { name: '2B ısı haritası' })).toBeChecked();
  expect(screen.getByRole('group', { name: 'Head çıktı gücü ısı haritası' })).toBeInTheDocument();
});

test('the view switch toggles between the heatmap and the Liste table', async () => {
  makeApi();
  uiStore.setState({ view: '2d' });
  render(<Explore />);
  const user = await run();
  expect(screen.queryByRole('table')).toBeNull();
  await user.click(screen.getByRole('radio', { name: 'Liste' }));
  expect(screen.getByRole('table')).toBeInTheDocument();
  expect(uiStore.getState().view).toBe('list');
});

test('MLP unit asks /api/mlp for the current token and lists the neurons', async () => {
  const api = makeApi();
  render(<Explore />);
  const user = await run('merhaba dün');
  await user.click(screen.getByRole('radio', { name: 'MLP · en güçlü 50' }));
  expect(await screen.findByText('K0 · nöron 98 · 0,082')).toBeInTheDocument();
  expect(screen.getByText('K1 · nöron 84 · 0,047')).toBeInTheDocument();
  expect(api.mlp).toHaveBeenCalledWith({ prompt: 'merhaba dün', token_index: 1 });
  await user.click(screen.getByRole('button', { name: '<b>hi' }));
  await waitFor(() => expect(api.mlp).toHaveBeenLastCalledWith({ prompt: 'merhaba dün', token_index: 0 }));
});

test('the explore view survives leaving and coming back', async () => {
  makeApi();
  const first = render(<Explore />);
  await run();
  first.unmount();
  render(<Explore />);
  expect(screen.getByRole('list', { name: 'Belirteçler' })).toBeInTheDocument();
  expect(screen.getByLabelText('Metin')).toHaveValue('merhaba dün');
});
