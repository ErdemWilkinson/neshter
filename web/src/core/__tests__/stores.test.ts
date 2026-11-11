import type { JobState, LoadedSession, SystemInfo } from '@/api/schemas';
import { headId } from '../ids';
import { fromEffectResult } from '../measurement';
import { createCompareStore } from '../stores/compare';
import { createJobsStore, isLocked } from '../stores/jobs';
import { createMeasureStore } from '../stores/measure';
import { createSelectionStore } from '../stores/selection';
import { createSessionStore } from '../stores/session';
import { createUiStore, intentRoute } from '../stores/ui';
import { notSignificantEffect, significantEffect } from './fixtures/futilityEffect';
import { NOW } from './fixtures/items';

const job = (state: JobState['state']): JobState => ({
  id: 'j', kind: 'effect', state, progress: 0, message: '', result: null, error: null, error_code: null, started_at: null,
});

test('ui store intent defaults to null and intentRoute maps reduce -> concept, else explore', () => {
  const ui = createUiStore();
  expect(ui.getState()).toMatchObject({ intent: null, route: 'home', view: '2d', lang: 'tr', reducedMotion: 'system', quality: 'balanced' });
  expect(intentRoute('reduce')).toBe('concept');
  expect(intentRoute('inspect')).toBe('explore');
  expect(intentRoute('expert')).toBe('explore');
  expect(intentRoute(null)).toBe('explore');
  ui.getState().setIntent('reduce');
  ui.getState().setRoute(intentRoute(ui.getState().intent));
  expect(ui.getState().route).toBe('concept');
});

test('selection store toggles, sets and clears without mutating the previous set', () => {
  const sel = createSelectionStore();
  const a = headId(0, 1);
  const b = headId(0, 2);
  sel.getState().toggle(a);
  const first = sel.getState().selected;
  sel.getState().toggle(b);
  expect(first.size).toBe(1);
  expect([...sel.getState().selected]).toEqual([a, b]);
  sel.getState().toggle(a);
  expect([...sel.getState().selected]).toEqual([b]);
  sel.getState().set([a]);
  sel.getState().setFocus(a);
  expect(sel.getState().focus).toBe(a);
  sel.getState().clear();
  expect(sel.getState().selected.size).toBe(0);
});

test('jobs store: isLocked only while queued or running; clearActive(id) leaves a newer job alone', () => {
  const jobs = createJobsStore();
  expect(isLocked(jobs.getState())).toBe(false);
  for (const [state, locked] of [['queued', true], ['running', true], ['done', false], ['error', false], ['cancelled', false]] as const) {
    jobs.getState().setActive(job(state));
    expect(isLocked(jobs.getState())).toBe(locked);
  }
  jobs.getState().setActive({ ...job('running'), id: 'new' });
  jobs.getState().clearActive('old');
  expect(jobs.getState().active?.id).toBe('new');
  jobs.getState().clearActive();
  expect(jobs.getState().active).toBeNull();
});

test('measure store keeps the latest measurement per fingerprint', () => {
  const store = createMeasureStore();
  const a = fromEffectResult(notSignificantEffect, 'fp-a', NOW);
  const b = fromEffectResult(significantEffect, 'fp-a', NOW);
  store.getState().put(a);
  expect(store.getState().get('fp-a')).toBe(a);
  store.getState().put(b);
  expect(store.getState().get('fp-a')).toBe(b);
  expect(store.getState().get('other')).toBeUndefined();
  expect(store.getState().get('toString')).toBeUndefined();
  expect(store.getState().get('__proto__')).toBeUndefined();
});

test('session and compare stores hold what they are given', () => {
  const session = createSessionStore();
  expect(session.getState()).toMatchObject({ system: null, model: null, stale: 'none', busy: false });
  session.getState().setStale('restarted');
  session.getState().setBusy(true);
  session.getState().setSystem({ gpu_name: null } as SystemInfo);
  session.getState().setModel({ loaded: true, name: 'm' } as LoadedSession);
  expect(session.getState()).toMatchObject({ stale: 'restarted', busy: true, system: { gpu_name: null }, model: { name: 'm' } });

  const compare = createCompareStore();
  expect(compare.getState()).toMatchObject({ lastPrompt: null, lastPair: null, lastSuite: null, refreshedDropped: null });
  compare.getState().setLastPrompt('hi');
  compare.getState().setRefreshedDropped(2);
  expect(compare.getState()).toMatchObject({ lastPrompt: 'hi', refreshedDropped: 2 });
});

test('core has no react import', () => {
  const sources = import.meta.glob('../**/*.ts', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;
  const names = Object.keys(sources);
  expect(names.length).toBeGreaterThan(15); // the glob found the core, not an empty set
  const offenders = names.filter((name) => /from ['"](react|react-dom|zustand\/react)['"]/.test(sources[name] ?? ''));
  expect(offenders).toEqual([]);
});
