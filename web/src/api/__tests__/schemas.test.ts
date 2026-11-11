import { FIXTURE_SCHEMAS, models, session } from '../schemas';

const fixtures = import.meta.glob<unknown>('../__fixtures__/*.json', { eager: true, import: 'default' });

function load(name: string): unknown {
  const value = fixtures[`../__fixtures__/${name}.json`];
  if (value === undefined) throw new Error(`fixture ${name} missing`);
  return value;
}

test('every fixture parses with its schema', () => {
  for (const [name, schema] of Object.entries(FIXTURE_SCHEMAS)) {
    expect(() => schema.parse(load(name)), name).not.toThrow();
  }
});

test('every fixture file has a schema and vice versa', () => {
  const files = Object.keys(fixtures)
    .map((path) => /([^/]+)\.json$/.exec(path)?.[1])
    .sort();
  expect(files).toEqual(Object.keys(FIXTURE_SCHEMAS).sort());
});

test('unknown keys pass (loose objects) and are kept', () => {
  const parsed = FIXTURE_SCHEMAS['prefs'].parse({ ...(load('prefs') as object), future_field: 1 });
  expect(parsed).toHaveProperty('future_field', 1);
});

test('session without a model has no measure_count; a loaded one defaults it to 0', () => {
  expect(session.parse({ loaded: false })).toEqual({ loaded: false });
  const older = { ...(load('session_loaded') as Record<string, unknown>) };
  delete older['measure_count'];
  expect(session.parse(older)).toMatchObject({ loaded: true, measure_count: 0 });
});

test('schemas reject wrong shapes', () => {
  expect(() => FIXTURE_SCHEMAS['estimate'].parse({ bytes_needed: 'many' })).toThrow();
  expect(() => FIXTURE_SCHEMAS['job_effect'].parse({ ...(load('job_effect') as object), state: 'weird' })).toThrow();
});

test('the effect result carries the sequential-test fields', () => {
  const job = FIXTURE_SCHEMAS['job_effect'].parse(load('job_effect')) as { result: Record<string, unknown> };
  for (const key of ['n_drawn', 'n_max', 'k_exceed', 'p_kind', 'stopped_early', 'significant', 'session_measure_index']) {
    expect(job.result).toHaveProperty(key);
  }
});

test('a model whose header gave no layer/head counts still parses (scan returns null ints)', () => {
  const [entry] = load('models') as Record<string, unknown>[];
  const odd = {
    ...entry,
    supported: false,
    unsupported_reason: 'unsupported_family',
    arch: { family: 'clip', n_layers: null, n_heads: null, n_kv_heads: null, d_mlp: null },
  };
  const parsed = FIXTURE_SCHEMAS['models'].parse([entry, odd, { ...entry, arch: null }]);
  expect(parsed).toHaveLength(3);
  expect(models.safeParse([{ ...entry, arch: { family: 'x', n_layers: 'two' } }]).success).toBe(false);
});
