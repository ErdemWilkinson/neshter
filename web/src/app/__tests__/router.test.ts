import { uiStore } from '@/core/stores/ui';
import { ROUTES, navigate, parseHash, startRouter } from '../router';
import { SCREENS } from '../Shell';

afterEach(() => {
  uiStore.setState({ route: 'home' });
  window.history.replaceState(null, '', '/#/');
});

test('parseHash maps known routes and sends everything else home', () => {
  expect(parseHash('#/models')).toBe('models');
  expect(parseHash('#/explore')).toBe('explore');
  expect(parseHash('#/concept/')).toBe('concept');
  expect(parseHash('#/compare?x=1')).toBe('compare');
  expect(parseHash('#/save')).toBe('save');
  expect(parseHash('')).toBe('home');
  expect(parseHash('#')).toBe('home');
  expect(parseHash('#/')).toBe('home');
  expect(parseHash('#/nope')).toBe('home');
  expect(parseHash('#/modelsX')).toBe('home');
  expect(parseHash('#models')).toBe('home');
  expect(parseHash('#token=abc')).toBe('home');
  expect(parseHash('#/__proto__')).toBe('home');
});

test('every route has a Shell entry', () => {
  expect(Object.keys(SCREENS).sort()).toEqual([...ROUTES].sort());
});

test('startRouter reads the current hash, follows hashchange, and stops on cleanup', () => {
  window.history.replaceState(null, '', '/#/compare');
  const stop = startRouter();
  expect(uiStore.getState().route).toBe('compare');

  window.location.hash = '#/save';
  window.dispatchEvent(new HashChangeEvent('hashchange'));
  expect(uiStore.getState().route).toBe('save');

  stop();
  window.location.hash = '#/models';
  window.dispatchEvent(new HashChangeEvent('hashchange'));
  expect(uiStore.getState().route).toBe('save');
});

test('navigate sets the hash and the route at once, and never writes a token', () => {
  navigate('explore');
  expect(window.location.hash).toBe('#/explore');
  expect(uiStore.getState().route).toBe('explore');
  expect(window.location.hash).not.toContain('token');
});
