import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { Api } from '@/api/client';
import { setApi } from '@/api/currentApi';
import type { ChatTurn, DeviceInfo, PhoneApi } from '@/api/device/deviceApi';
import { ApiError } from '@/api/errors';
import type { LoadedSession } from '@/api/schemas';
import { sessionLoaded } from '@/app/__tests__/fakeServer';
import { basketStore } from '@/core/stores/basket';
import { sessionStore } from '@/core/stores/session';
import { head } from '@/core/__tests__/fixtures/items';
import { setLang } from '@/i18n';
import { Chat } from '../Chat';

type ChatBody = Parameters<PhoneApi['chat']>[0];

function fakePhone(headsSupported = true) {
  const calls: ChatBody[] = [];
  let release: (() => void) | null = null;
  const phone: PhoneApi = {
    importModel: () => Promise.reject(new Error('unused')),
    deleteModel: () => Promise.resolve(),
    activeModel: () => ({ heads_supported: headsSupported }) as DeviceInfo,
    async chat(body, onText) {
      calls.push(body);
      onText('Mer');
      onText('haba!');
      if (body.messages.at(-1)?.content === 'uzun') {
        await new Promise<void>((r) => (release = r));
        throw new ApiError(409, 'cancelled', 'İptal edildi.');
      }
      return 'Merhaba!';
    },
    stop: () => {
      release?.();
      return Promise.resolve();
    },
  };
  return { phone, calls };
}

beforeEach(() => {
  setLang('tr');
  sessionStore.setState({ model: sessionLoaded as unknown as LoadedSession });
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
});

afterEach(() => {
  cleanup();
  setApi(null);
  sessionStore.setState({ model: null });
});

test('without an open model the chat points to the model list', () => {
  sessionStore.setState({ model: null });
  setApi(fakePhone().phone as unknown as Api);
  render(<Chat />);
  expect(screen.getByRole('button', { name: 'Modellere git' })).toBeInTheDocument();
});

test('a message gets a streamed answer, and the whole conversation goes with the next one', async () => {
  const { phone, calls } = fakePhone();
  setApi(phone as unknown as Api);
  render(<Chat />);
  await userEvent.type(screen.getByRole('textbox', { name: 'Mesajın' }), 'Selam{Enter}');
  expect(await screen.findByText('Merhaba!')).toBeInTheDocument();
  expect(screen.getByText('Selam')).toBeInTheDocument();
  await userEvent.type(screen.getByRole('textbox', { name: 'Mesajın' }), 'Nasılsın?');
  await userEvent.click(screen.getByRole('button', { name: 'Gönder' }));
  await waitFor(() => expect(calls).toHaveLength(2));
  const turns: ChatTurn[] = calls[1]?.messages ?? [];
  expect(turns.map((m) => m.role)).toEqual(['user', 'assistant', 'user']);
  expect(calls[1]?.cut).toEqual([]);
  expect(calls[1]?.temperature).toBe(0);
});

test('the basket head cuts can be switched on, and the answer is marked as cut', async () => {
  basketStore.getState().replaceAll({ items: [head(2, 5), head(0, 1)], shelf: [], savedOps: [] });
  const { phone, calls } = fakePhone();
  setApi(phone as unknown as Api);
  render(<Chat />);
  await userEvent.click(screen.getByRole('checkbox', { name: 'Sepetteki kesimlerle (2 head)' }));
  await userEvent.click(screen.getByRole('radio', { name: 'Yaratıcı' }));
  await userEvent.type(screen.getByRole('textbox', { name: 'Mesajın' }), 'Selam{Enter}');
  await screen.findByText('Merhaba!');
  expect(calls[0]?.cut).toEqual([
    [2, 5],
    [0, 1],
  ]);
  expect(calls[0]?.temperature).toBe(0.7);
  expect(screen.getByText('kesimli')).toBeInTheDocument();
});

test('a model whose heads cannot be read chats without the cut option', () => {
  basketStore.getState().replaceAll({ items: [head(2, 5)], shelf: [], savedOps: [] });
  setApi(fakePhone(false).phone as unknown as Api);
  render(<Chat />);
  expect(screen.queryByRole('checkbox')).toBeNull();
  expect(screen.getByText(/head'ler ayrı okunamıyor/)).toBeInTheDocument();
});

test('stopping keeps what was written so far and marks it', async () => {
  const { phone } = fakePhone();
  setApi(phone as unknown as Api);
  render(<Chat />);
  await userEvent.type(screen.getByRole('textbox', { name: 'Mesajın' }), 'uzun{Enter}');
  await userEvent.click(await screen.findByRole('button', { name: 'Durdur' }));
  expect(await screen.findByText('(durduruldu)')).toBeInTheDocument();
  expect(screen.getByText(/Merhaba!/)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Gönder' })).toBeInTheDocument();
});
