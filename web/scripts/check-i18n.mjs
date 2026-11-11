// Build guard for the tr/en catalogs: same keys, same {placeholders}, no key without a use in src/**.
import { readdirSync, readFileSync, realpathSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

/** Keys under these prefixes are built at run time (`errors.${code}`), so a literal use is not required. */
const DYNAMIC_PREFIXES = ['errors.', 'vocab.badge.', 'concept.category.'];
const PLURAL_SUFFIX = /\.(zero|one|two|few|many|other)$/;
const KEY_LITERAL = /(['"`])([A-Za-z][\w-]*(?:\.[\w-]+)+)\1/g;
const SOURCE_FILE = /\.(ts|tsx)$/;

const placeholders = (message) => [...message.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();

function listSources(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    const segments = relative(dir, full).split(sep);
    if (name === 'node_modules' || name === '__tests__' || name === 'e2e') continue;
    if (statSync(full).isDirectory()) out.push(...listSources(full));
    else if (SOURCE_FILE.test(name) && !segments.includes('__tests__')) out.push(full);
  }
  return out;
}

function usedLiterals(srcDir) {
  const used = new Set();
  for (const file of listSources(srcDir)) {
    for (const match of readFileSync(file, 'utf8').matchAll(KEY_LITERAL)) used.add(match[2]);
  }
  return used;
}

/**
 * @param {{trPath: string, enPath: string, srcDir: string}} paths
 * @returns {string[]} problems (empty when the catalogs are consistent)
 */
export function checkI18n({ trPath, enPath, srcDir }) {
  const tr = JSON.parse(readFileSync(trPath, 'utf8'));
  const en = JSON.parse(readFileSync(enPath, 'utf8'));
  const errors = [];
  for (const key of Object.keys(tr)) if (!Object.hasOwn(en, key)) errors.push(`missing in en: ${key}`);
  for (const key of Object.keys(en)) if (!Object.hasOwn(tr, key)) errors.push(`missing in tr: ${key}`);
  for (const key of Object.keys(tr)) {
    if (!Object.hasOwn(en, key)) continue;
    const a = placeholders(tr[key]).join(',');
    const b = placeholders(en[key]).join(',');
    if (a !== b) errors.push(`placeholder mismatch in ${key}: tr {${a}} vs en {${b}}`);
  }
  const used = usedLiterals(srcDir);
  for (const key of new Set([...Object.keys(tr), ...Object.keys(en)])) {
    if (DYNAMIC_PREFIXES.some((prefix) => key.startsWith(prefix))) continue;
    // `plural(base, n)` looks up `base.one` etc. itself.
    const base = PLURAL_SUFFIX.test(key) ? key.replace(PLURAL_SUFFIX, '') : key;
    if (!used.has(key) && !used.has(base)) errors.push(`unused key: ${key}`);
  }
  return errors;
}

/** Run as a script? Node resolves the main module to its real path, so compare real paths (a link, junction or other spelling of argv[1] must not turn the guard into a silent no-op). */
function isMain() {
  if (!process.argv[1]) return false;
  try {
    return realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (isMain()) {
  const web = fileURLToPath(new URL('..', import.meta.url));
  const errors = checkI18n({
    trPath: join(web, 'src/i18n/tr.json'),
    enPath: join(web, 'src/i18n/en.json'),
    srcDir: join(web, 'src'),
  });
  if (errors.length > 0) {
    console.error(`check-i18n: ${errors.length} problem(s)\n${errors.map((e) => `  - ${e}`).join('\n')}`);
    process.exit(1);
  }
  console.log('check-i18n: ok');
}
