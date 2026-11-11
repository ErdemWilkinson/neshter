import { createApi } from '@/api/client';
import { uiStore } from '@/core/stores/ui';
import { setLang } from '@/i18n';
import { DEFAULT_PREFS, applyMotion, applyPrefs, isMotionReduced, loadPrefs, savePrefs } from '../prefs';
import { bodyOf, fakeServer, jsonResponse, prefs } from './fakeServer';

function apiFor(server = fakeServer()) {
  return { server, api: createApi({ token: 't', fetchFn: server.fetchFn as unknown as typeof fetch, onStale: () => undefined }) };
}

const media = (matches: boolean) => () => ({ matches });

afterEach(() => {
  setLang('tr');
  uiStore.setState({ reducedMotion: 'system', quality: 'balanced' });
  document.documentElement.removeAttribute('data-motion');
});

test('loadPrefs returns the server prefs and falls back to defaults on failure', async () => {
  const stored = { ...prefs, lang: 'en', quality: 'low' };
  const ok = apiFor(fakeServer({ 'GET /api/prefs': () => jsonResponse(200, stored) }));
  expect(await loadPrefs(ok.api)).toEqual(stored);

  const broken = apiFor(fakeServer({ 'GET /api/prefs': () => jsonResponse(500, { detail: 'x', error_code: 'internal' }) }));
  expect(await loadPrefs(broken.api)).toEqual(DEFAULT_PREFS);

  const offline = apiFor(
    fakeServer({
      'GET /api/prefs': () => {
        throw new TypeError('Failed to fetch');
      },
    }),
  );
  expect(await loadPrefs(offline.api)).toEqual(DEFAULT_PREFS);
});

test('savePrefs PUTs the last known prefs merged with the patch, so nothing else is lost', async () => {
  const stored = { ...prefs, quality: 'low', refusal_card_seen: true };
  const { api, server } = apiFor(fakeServer({ 'GET /api/prefs': () => jsonResponse(200, stored) }));
  await loadPrefs(api);
  const saved = await savePrefs(api, { lang: 'en' });
  expect(saved).toEqual({ ...stored, lang: 'en' });
  expect(bodyOf(server.calls.at(-1)?.init)).toEqual({ ...stored, lang: 'en' });

  // the next patch builds on the previous save
  await savePrefs(api, { reduced_motion: 'on' });
  expect(bodyOf(server.calls.at(-1)?.init)).toEqual({ ...stored, lang: 'en', reduced_motion: 'on' });
});

test('motion pref on sets data-motion=reduce, off sets full, system follows the OS', () => {
  const html = document.documentElement;
  applyMotion('on', media(false));
  expect(html.dataset['motion']).toBe('reduce');
  applyMotion('off', media(true));
  expect(html.dataset['motion']).toBe('full');
  applyMotion('system', media(true));
  expect(html.dataset['motion']).toBe('reduce');
  applyMotion('system', media(false));
  expect(html.dataset['motion']).toBe('full');
  applyMotion('system', undefined); // no matchMedia at all
  expect(html.dataset['motion']).toBe('full');
  expect(isMotionReduced('system', media(true))).toBe(true);
});

test('applyPrefs puts language, motion and quality into the stores and the document', () => {
  applyPrefs({ ...prefs, reduced_motion: 'on', quality: 'high' }, 'en', undefined);
  expect(uiStore.getState()).toMatchObject({ lang: 'en', reducedMotion: 'on', quality: 'high' });
  expect(document.documentElement.lang).toBe('en');
  expect(document.documentElement.dataset['motion']).toBe('reduce');
});

test('a save after a failed GET reads the stored prefs again instead of overwriting them with defaults', async () => {
  const stored = { ...prefs, lang: 'en', quality: 'low', reduced_motion: 'on', refusal_card_seen: false };
  let getFails = true;
  const { api, server } = apiFor(
    fakeServer({
      'GET /api/prefs': () => (getFails ? jsonResponse(500, { detail: 'x', error_code: 'internal' }) : jsonResponse(200, stored)),
    }),
  );
  expect(await loadPrefs(api)).toEqual(DEFAULT_PREFS); // the UI runs on defaults
  getFails = false;
  await savePrefs(api, { refusal_card_seen: true });
  expect(bodyOf(server.calls.at(-1)?.init)).toEqual({ ...stored, refusal_card_seen: true });
});

test('a save is refused, not sent, while the stored prefs still cannot be read', async () => {
  const { api, server } = apiFor(fakeServer({ 'GET /api/prefs': () => jsonResponse(500, { detail: 'x', error_code: 'internal' }) }));
  await loadPrefs(api);
  await expect(savePrefs(api, { lang: 'en' })).rejects.toBeDefined();
  expect(server.calls.some((c) => c.key === 'PUT /api/prefs')).toBe(false);
});

test('concurrent saves are serialized and neither patch is lost', async () => {
  const stored = { ...prefs };
  let serverState = { ...stored };
  const { api, server } = apiFor(
    fakeServer({
      'GET /api/prefs': () => jsonResponse(200, serverState),
      'PUT /api/prefs': async (init) => {
        await new Promise((resolve) => setTimeout(resolve, 20)); // slow first write
        serverState = bodyOf(init) as typeof serverState;
        return jsonResponse(200, serverState);
      },
    }),
  );
  await loadPrefs(api);
  await Promise.all([savePrefs(api, { lang: 'en' }), savePrefs(api, { reduced_motion: 'on' })]);
  expect(serverState).toEqual({ ...stored, lang: 'en', reduced_motion: 'on' });
  expect(server.calls.filter((c) => c.key === 'PUT /api/prefs')).toHaveLength(2);
});
