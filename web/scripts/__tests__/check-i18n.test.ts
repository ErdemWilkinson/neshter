import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkI18n } from '../check-i18n.mjs';

const WEB_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const TMP_ROOT = join(WEB_DIR, 'node_modules', '.tmp-tests');

interface Fixture {
  tr: Record<string, string>;
  en: Record<string, string>;
  sources: Record<string, string>;
}

function makeProject({ tr, en, sources }: Fixture) {
  mkdirSync(TMP_ROOT, { recursive: true });
  const dir = mkdtempSync(join(TMP_ROOT, 'i18n-'));
  const srcDir = join(dir, 'src');
  mkdirSync(srcDir);
  writeFileSync(join(dir, 'tr.json'), JSON.stringify(tr));
  writeFileSync(join(dir, 'en.json'), JSON.stringify(en));
  for (const [rel, content] of Object.entries(sources)) {
    const full = join(srcDir, rel);
    mkdirSync(dirname(full), { recursive: true });
    writeFileSync(full, content);
  }
  return { trPath: join(dir, 'tr.json'), enPath: join(dir, 'en.json'), srcDir };
}

const used = { 'a.ts': "t('a.one'); t('a.two');" };

test('a consistent project passes', () => {
  const project = makeProject({ tr: { 'a.one': 'bir {n}', 'a.two': 'iki' }, en: { 'a.one': 'one {n}', 'a.two': 'two' }, sources: used });
  expect(checkI18n(project)).toEqual([]);
});

test('check-i18n fails on a missing key and on placeholder mismatch', () => {
  const missing = makeProject({ tr: { 'a.one': 'bir', 'a.two': 'iki' }, en: { 'a.one': 'one' }, sources: used });
  expect(checkI18n(missing)).toEqual(['missing in en: a.two']);

  const extra = makeProject({ tr: { 'a.one': 'bir' }, en: { 'a.one': 'one', 'a.two': 'two' }, sources: used });
  expect(checkI18n(extra)).toEqual(['missing in tr: a.two']);

  const mismatch = makeProject({
    tr: { 'a.one': 'bir {n}', 'a.two': 'iki' },
    en: { 'a.one': 'one {count}', 'a.two': 'two' },
    sources: used,
  });
  expect(checkI18n(mismatch).join('\n')).toMatch(/placeholder mismatch in a\.one/);
});

test('check-i18n ignores usages inside __tests__', () => {
  const project = makeProject({
    tr: { 'a.one': 'bir', 'a.two': 'iki', 'only.test': 'x' },
    en: { 'a.one': 'one', 'a.two': 'two', 'only.test': 'x' },
    sources: { ...used, 'x/__tests__/a.test.ts': "t('only.test');", 'e2e/a.spec.ts': "t('only.test');" },
  });
  expect(checkI18n(project)).toEqual(['unused key: only.test']);
});

test('dynamic prefixes and plural forms count as used', () => {
  const project = makeProject({
    tr: { 'errors.busy': 'a', 'vocab.badge.measured': 'b', 'n.cut': 'c {n}', 'n.cut.one': 'd {n}' },
    en: { 'errors.busy': 'a', 'vocab.badge.measured': 'b', 'n.cut': 'c {n}', 'n.cut.one': 'd {n}' },
    sources: { 'a.tsx': "plural('n.cut', 2);" },
  });
  expect(checkI18n(project)).toEqual([]);
});

test('the real catalogs are consistent', () => {
  expect(
    checkI18n({
      trPath: join(WEB_DIR, 'src/i18n/tr.json'),
      enPath: join(WEB_DIR, 'src/i18n/en.json'),
      srcDir: join(WEB_DIR, 'src'),
    }),
  ).toEqual([]);
});

describe('check-i18n command line', () => {
  const run = (script: string) => spawnSync(process.execPath, [script], { encoding: 'utf8' });

  test('runs and reports ok when started by its plain path', () => {
    const result = run(join(WEB_DIR, 'scripts', 'check-i18n.mjs'));
    expect(result.stdout).toContain('check-i18n: ok');
    expect(result.status).toBe(0);
  });

  test('does not silently do nothing when started through a link (the entry guard compares real paths)', () => {
    mkdirSync(TMP_ROOT, { recursive: true });
    const link = join(mkdtempSync(join(TMP_ROOT, 'link-')), 'scripts-link');
    symlinkSync(join(WEB_DIR, 'scripts'), link, 'junction');
    const result = run(join(link, 'check-i18n.mjs'));
    expect(result.stdout).toContain('check-i18n: ok');
    expect(result.status).toBe(0);
  });
});
