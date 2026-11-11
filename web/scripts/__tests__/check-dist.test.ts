import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkDist } from '../check-dist.mjs';

const TMP_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'node_modules', '.tmp-tests');

function makeTmpDist(files: Record<string, string>): string {
  mkdirSync(TMP_ROOT, { recursive: true });
  const dir = mkdtempSync(join(TMP_ROOT, 'dist-'));
  for (const [rel, content] of Object.entries(files)) {
    const full = join(dir, rel);
    mkdirSync(dirname(full), { recursive: true });
    writeFileSync(full, content);
  }
  return dir;
}

function cleanFixture(): Record<string, string> {
  return {
    'index.html':
      '<!doctype html><html><head><link rel="stylesheet" href="/assets/index-abc.css">' +
      '<script type="module" crossorigin src="/assets/index-abc.js"></script></head><body><div id="root"></div></body></html>',
    'assets/index-abc.js': 'console.log("ok");',
    'assets/index-abc.css': '@font-face{src:url(/assets/inter-latin-ext-wght-normal-x1.woff2)}',
    'assets/inter-latin-ext-wght-normal-x1.woff2': 'wOF2',
    'assets/inter-latin-wght-normal-x2.woff2': 'wOF2',
  };
}

test('flags inline script, external url, data font, blob worker and missing latin-ext', () => {
  const dir = makeTmpDist({
    'index.html': '<script>alert(1)</script><link href="https://fonts.googleapis.com/x">',
    'assets/a.css': '@font-face{src:url(data:font/woff2;base64,AAAA)}',
    'assets/e.js': 'new Worker(URL.createObjectURL(new Blob([s])))',
  });
  const errs = checkDist(dir);
  expect(errs.join('\n')).toMatch(/inline script/);
  expect(errs.join('\n')).toMatch(/external url/);
  expect(errs.join('\n')).toMatch(/data: font/);
  expect(errs.join('\n')).toMatch(/blob worker/);
  expect(errs.join('\n')).toMatch(/latin-ext/);
});

test('clean dist passes', () => {
  expect(checkDist(makeTmpDist(cleanFixture()))).toEqual([]);
});

test('flags a module script without type=module and a protocol-relative url', () => {
  const files = cleanFixture();
  files['index.html'] =
    '<link href="//cdn.example.com/x.css"><script src="/assets/index-abc.js"></script>';
  const errs = checkDist(makeTmpDist(files)).join('\n');
  expect(errs).toMatch(/type="module"/);
  expect(errs).toMatch(/external url/);
});

test('flags an entry chunk over the gzip budget', () => {
  const files = cleanFixture();
  files['assets/index-abc.js'] = Array.from({ length: 4000 }, (_, i) => `v${i}=${Math.sin(i)};`).join('');
  expect(checkDist(makeTmpDist(files), { entryBudgetGz: 100 }).join('\n')).toMatch(/entry chunk/);
  expect(checkDist(makeTmpDist(files))).toEqual([]);
});
