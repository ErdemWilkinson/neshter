import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ESLint } from 'eslint';

// Pins where the scoped ESLint overrides apply. Breaks when eslint.config.js stops matching top-level
// anchored paths (the repo folder name contains U+0130, see the workaround at the top of eslint.config.js).
const WEB_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const eslint = new ESLint({ cwd: WEB_DIR });

async function ruleIds(relPath: string, code: string): Promise<string[]> {
  const [result] = await eslint.lintText(code, { filePath: join(WEB_DIR, relPath) });
  return (result?.messages ?? []).map((m) => m.ruleId ?? 'parse');
}

const LOCAL = 'export const n = localStorage.length;\n';
const SESSION = 'export const n = sessionStorage.length;\n';

test.each(['e2e/x.spec.ts', 'e2e/sub/x.spec.ts', 'e2e/helpers/y.ts', 'src/__tests__/z.test.ts'])(
  'localStorage is allowed and sessionStorage still banned in %s',
  async (path) => {
    expect(await ruleIds(path, LOCAL)).toEqual([]);
    expect(await ruleIds(path, SESSION)).toEqual(['no-restricted-globals']);
  },
);

test('product code bans both storages, except sessionStorage in src/app/sessionStorage.ts', async () => {
  expect(await ruleIds('src/features/a.ts', LOCAL)).toEqual(['no-restricted-globals']);
  expect(await ruleIds('src/features/a.ts', SESSION)).toEqual(['no-restricted-globals']);
  expect(await ruleIds('src/app/sessionStorage.ts', SESSION)).toEqual([]);
  expect(await ruleIds('src/app/sessionStorage.ts', LOCAL)).toEqual(['no-restricted-globals']);
});

test('src/core bans react and zod, except zod in src/core/zod.ts', async () => {
  expect(await ruleIds('src/core/a.ts', "import { useState } from 'react';\nexport { useState };\n")).toEqual([
    'no-restricted-imports',
  ]);
  expect(await ruleIds('src/core/a.ts', "import { z } from 'zod';\nexport { z };\n")).toEqual(['no-restricted-imports']);
  expect(await ruleIds('src/core/zod.ts', "import { z } from 'zod';\nexport { z };\n")).toEqual([]);
  expect(await ruleIds('src/features/a.ts', "import { z } from 'zod';\nexport { z };\n")).toEqual(['no-restricted-imports']);
});

test('the golden-hash fixture tree is ignored by lint', async () => {
  expect(await eslint.isPathIgnored(join(WEB_DIR, 'scripts/__tests__/__fixtures__/hash/src/a.ts'))).toBe(true);
  expect(await eslint.isPathIgnored(join(WEB_DIR, 'src/App.tsx'))).toBe(false);
});
