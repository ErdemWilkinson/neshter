// Deterministic digest of the web/ sources that feed the bundle. Re-implemented in tests/test_web_dist.py
// (Task 15) against the same golden vector in __tests__/__fixtures__/hash/.
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const ROOT_FILES = ['index.html', 'vite.config.ts', 'tsconfig.json', 'package-lock.json'];
const ROOT_DIRS = ['src', 'public'];

function walk(webDir, rel, out) {
  for (const name of readdirSync(join(webDir, rel))) {
    const childRel = `${rel}/${name}`;
    if (statSync(join(webDir, childRel)).isDirectory()) walk(webDir, childRel, out);
    else out.push(childRel);
  }
}

function excluded(rel) {
  return `/${rel}`.includes('/__tests__/') || rel.includes('/__fixtures__/') || rel.includes('.test.');
}

function normalizeEol(buf) {
  const out = Buffer.allocUnsafe(buf.length);
  let n = 0;
  for (let i = 0; i < buf.length; i += 1) {
    if (buf[i] === 0x0d && buf[i + 1] === 0x0a) continue;
    out[n] = buf[i];
    n += 1;
  }
  return out.subarray(0, n);
}

/** @param {string} webDir */
export function sourceHash(webDir) {
  const files = [];
  for (const dir of ROOT_DIRS) if (existsSync(join(webDir, dir))) walk(webDir, dir, files);
  for (const f of ROOT_FILES) if (existsSync(join(webDir, f))) files.push(f);
  files.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  const digest = createHash('sha256');
  for (const rel of files) {
    if (excluded(rel)) continue;
    const content = normalizeEol(readFileSync(join(webDir, rel)));
    digest.update(`${rel}\0${createHash('sha256').update(content).digest('hex')}\n`);
  }
  return digest.digest('hex');
}
