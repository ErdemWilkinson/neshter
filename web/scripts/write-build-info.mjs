// Stamps neshter/web/dist/build-info.json so a stale committed bundle can be detected (tests/test_web_dist.py).
import { mkdirSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { sourceHash } from './source-hash.mjs';

const webDir = fileURLToPath(new URL('..', import.meta.url));
const distDir = fileURLToPath(new URL('../../neshter/web/dist/', import.meta.url));
mkdirSync(distDir, { recursive: true });
const info = { source_hash: sourceHash(webDir), built_at: new Date().toISOString() };
writeFileSync(`${distDir}build-info.json`, `${JSON.stringify(info)}\n`);
console.log(`build-info: ${info.source_hash}`);
