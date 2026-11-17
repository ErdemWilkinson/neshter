import { createBridge, NativeError, type NativeHost } from '../bridge';

test('events of a call reach its listener until the reply settles it', async () => {
  const target = {} as Window;
  let lastId = '';
  const host: NativeHost = { call: (id) => (lastId = id) };
  const bridge = createBridge(host, target);
  const seen: unknown[] = [];
  const answer = bridge.invoke<string>('chat', { messages: [] }, (e) => seen.push(e));
  target.__neshterEvent?.(lastId, JSON.stringify({ text: 'a' }));
  target.__neshterEvent?.(lastId, 'not json'); // ignored, the call goes on
  target.__neshterEvent?.('other', JSON.stringify({ text: 'x' }));
  target.__neshterReply?.(lastId, true, JSON.stringify('done'));
  await expect(answer).resolves.toBe('done');
  target.__neshterEvent?.(lastId, JSON.stringify({ text: 'late' }));
  expect(seen).toEqual([{ text: 'a' }]);
});

test('a failed reply carries the native error code', async () => {
  const target = {} as Window;
  let lastId = '';
  const bridge = createBridge({ call: (id) => (lastId = id) }, target);
  const p = bridge.invoke('load');
  target.__neshterReply?.(lastId, false, JSON.stringify({ code: 'not_found', message: 'yok' }));
  await expect(p).rejects.toEqual(new NativeError('yok', 'not_found'));
});
