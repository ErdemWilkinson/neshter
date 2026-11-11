// Build guard: the bundle must run under the server CSP and make no request outside its own origin.
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, join, relative, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gzipSync } from 'node:zlib';

const DEFAULT_ENTRY_BUDGET_GZ = 250_000;
const TEXT_EXT = /\.(html|css|js|mjs|json|svg|txt|map)$/i;
const EXTERNAL_URL = /https?:\/\/|(?<=["'(=\s,])\/\/[A-Za-z0-9]/;
const DATA_FONT = /data:(?:font|application\/(?:x-)?font)/i;
const BLOB_WORKER = /createObjectURL\(\s*new Blob/;

function listFiles(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) out.push(...listFiles(full));
    else out.push(full);
  }
  return out;
}

/**
 * @param {string} dir built dist directory
 * @param {{entryBudgetGz?: number}} [opts]
 * @returns {string[]} error list (empty when the bundle is clean)
 */
export function checkDist(dir, opts = {}) {
  const budget = opts.entryBudgetGz ?? DEFAULT_ENTRY_BUDGET_GZ;
  const errors = [];
  const files = existsSync(dir) ? listFiles(dir) : [];
  const relOf = (f) => relative(dir, f).split(sep).join('/');

  const indexPath = join(dir, 'index.html');
  let entrySrc = null;
  if (!existsSync(indexPath)) {
    errors.push('index.html is missing');
  } else {
    const html = readFileSync(indexPath, 'utf8');
    for (const m of html.matchAll(/<script\b([^>]*)>/gi)) {
      const attrs = m[1] ?? '';
      const src = /\bsrc\s*=\s*["']([^"']+)["']/i.exec(attrs);
      if (!src) errors.push('index.html: inline script (every <script> needs src)');
      if (!/\btype\s*=\s*["']module["']/i.test(attrs)) errors.push('index.html: <script> without type="module"');
      if (src && entrySrc === null) entrySrc = src[1];
    }
    if (EXTERNAL_URL.test(html)) errors.push('index.html: external url (http://, https:// or //host)');
  }

  for (const file of files) {
    const rel = relOf(file);
    if (!TEXT_EXT.test(file)) continue;
    const text = readFileSync(file, 'utf8');
    if (/\.css$/i.test(file) && EXTERNAL_URL.test(text)) errors.push(`${rel}: external url in css`);
    if (DATA_FONT.test(text)) errors.push(`${rel}: data: font URI (fonts must be emitted files)`);
    if (/\.m?js$/i.test(file) && BLOB_WORKER.test(text)) errors.push(`${rel}: blob worker (createObjectURL(new Blob ...))`);
  }

  if (!files.some((f) => basename(f).includes('latin-ext'))) {
    errors.push('no latin-ext font emitted (Turkish ş ğ İ need it)');
  }

  if (entrySrc !== null) {
    const entryPath = join(dir, entrySrc.replace(/^\//, ''));
    if (!existsSync(entryPath)) errors.push(`entry chunk ${entrySrc} not found`);
    else {
      const gz = gzipSync(readFileSync(entryPath)).length;
      if (gz > budget) errors.push(`entry chunk ${entrySrc} is ${gz} bytes gzipped, budget ${budget}`);
    }
  }
  return errors;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const dist = fileURLToPath(new URL('../../neshter/web/dist/', import.meta.url));
  const errors = checkDist(dist);
  if (errors.length > 0) {
    for (const e of errors) console.error(`check-dist: ${e}`);
    process.exit(1);
  }
  console.log('check-dist: ok');
}
