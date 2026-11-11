import { memStorage } from '@/api/__tests__/helpers';
import { bootstrapToken } from '../bootstrap';
import { getActiveJob, getToken, setActiveJob, clearActiveJob, setToken } from '../sessionStorage';

test('bootstrap moves token to sessionStorage and strips hash', () => {
  const ss = memStorage();
  const hist = { replaceState: vi.fn() };
  expect(bootstrapToken({ hash: '#token=abc', pathname: '/' }, ss, hist, false)).toBe('abc');
  expect(ss.getItem('neshter.token')).toBe('abc');
  expect(hist.replaceState).toHaveBeenCalledWith(null, '', '/#/');
  expect(window.localStorage.length).toBe(0);
});

test('bootstrap reuses the sessionStorage token on reload', () => {
  const ss = memStorage();
  ss.setItem('neshter.token', 'abc');
  const hist = { replaceState: vi.fn() };
  expect(bootstrapToken({ hash: '#/explore', pathname: '/' }, ss, hist, false)).toBe('abc');
  expect(hist.replaceState).not.toHaveBeenCalled();
});

test('a fresh #token= replaces a stored token', () => {
  const ss = memStorage();
  ss.setItem('neshter.token', 'old');
  const hist = { replaceState: vi.fn() };
  expect(bootstrapToken({ hash: '#token=new', pathname: '/app/' }, ss, hist, false)).toBe('new');
  expect(ss.getItem('neshter.token')).toBe('new');
  expect(hist.replaceState).toHaveBeenCalledWith(null, '', '/app/#/');
});

test('bootstrap decodes the token and ignores an empty one', () => {
  const ss = memStorage();
  const hist = { replaceState: vi.fn() };
  expect(bootstrapToken({ hash: '#token=a%2Bb_c-d', pathname: '/' }, ss, hist, false)).toBe('a+b_c-d');
  expect(bootstrapToken({ hash: '#token=', pathname: '/' }, memStorage(), hist, false)).toBeNull();
});

test('bootstrap returns null without token outside dev', () => {
  const hist = { replaceState: vi.fn() };
  expect(bootstrapToken({ hash: '', pathname: '/' }, memStorage(), hist, false)).toBeNull();
  expect(hist.replaceState).not.toHaveBeenCalled();
});

test('bootstrap returns an empty token in dev (the proxy injects it)', () => {
  const hist = { replaceState: vi.fn() };
  expect(bootstrapToken({ hash: '', pathname: '/' }, memStorage(), hist, true)).toBe('');
});

test('sessionStorage helpers use the neshter keys and survive a throwing storage', () => {
  const ss = memStorage();
  setToken('t', ss);
  setActiveJob('job1', ss);
  expect(ss.getItem('neshter.token')).toBe('t');
  expect(getToken(ss)).toBe('t');
  expect(getActiveJob(ss)).toBe('job1');
  clearActiveJob(ss);
  expect(getActiveJob(ss)).toBeNull();

  const broken = { ...memStorage(), getItem: () => { throw new Error('denied'); } } as Storage;
  expect(getToken(broken)).toBeNull();
});
