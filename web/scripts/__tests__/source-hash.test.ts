import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { sourceHash } from '../source-hash.mjs';

const FIXTURE_DIR = join(dirname(fileURLToPath(import.meta.url)), '__fixtures__', 'hash');

test('source hash matches the golden vector (CRLF normalised, tests excluded)', () => {
  expect(sourceHash(FIXTURE_DIR)).toBe(readFileSync(join(FIXTURE_DIR, 'expected.txt'), 'utf8').trim());
});
