# Neşter web (2D UI) — developer notes

Everything below runs from the repo root. `<WEBENV>` means:

```powershell
$env:npm_config_cache = 'D:\neshter\npm-cache'; $env:PLAYWRIGHT_BROWSERS_PATH = 'D:\neshter\pw-browsers'
```

## Dev loop

1. Backend: `neshter serve --port 8765 --no-browser` — copy the printed token.
2. Frontend: `NESHTER_DEV_TOKEN=<token> NESHTER_DEV_PORT=8765 npm --prefix web run dev`
   (Vite proxies `/api` to `127.0.0.1:$NESHTER_DEV_PORT`, drops `Origin`, adds `X-Neshter-Token`).
3. Checks: `npm --prefix web run test | lint | build`. `build` writes `neshter/web/dist` (+ `build-info.json`) and
   runs `scripts/check-dist.mjs`; `dist` is committed only in Task 15 / at releases.

## API contract fixtures

`src/api/__fixtures__/*.json` are shape-only copies of live server responses (lists cut to 2 items, strings to 200
characters, absolute paths replaced by `C:\\Users\\x\\model`). `tests/test_web_contract.py` compares their key shapes with a
tiny model behind `TestClient`; after a server response changes, regenerate and review the diff:

```bash
NESHTER_UPDATE_FIXTURES=1 CUDA_VISIBLE_DEVICES= D:/neshter/venv/Scripts/python -m pytest -q -p no:cacheprovider tests/test_web_contract.py
```

Then update `src/api/schemas.ts` (`FIXTURE_SCHEMAS` must name every fixture). UI tests that need specific values keep
hand-made fixtures next to the test.

## D: workspace and the node_modules junction

- `web/node_modules` is a **directory junction** to `D:\neshter\web\node_modules` (path keeps a `node_modules`
  segment on purpose, see Global Constraints E1). Nothing is installed or cached on C:
  (`npm_config_cache` = `D:\neshter\npm-cache`, Vite/Vitest caches live under `node_modules/`).
- Create it once: `cmd /c mklink /J web\node_modules D:\neshter\web\node_modules`.
- **`npm install` / `npm ci` run inside `web/` replace the junction with a real directory on C: (npm unlinks it
  as a "non-directory" and recreates it).** Observed on npm 11.16 / Node 24.18. So install on D: instead and let
  the junction see the result:

  ```powershell
  # <WEBENV>
  Copy-Item web\package.json, web\package-lock.json, web\.npmrc D:\neshter\web\ -Force
  npm --prefix D:\neshter\web ci            # reproduce the lock
  # adding or bumping a package:
  #   npm --prefix D:\neshter\web install --save-exact <pkg>@<ver>   (or --save-dev)
  #   Copy-Item D:\neshter\web\package.json, D:\neshter\web\package-lock.json web\ -Force
  ```

- After any install verify (E2): `(Get-Item web\node_modules).LinkType` is `Junction` and
  `Test-Path D:\neshter\web\node_modules\react\package.json` is `True`. `npm run <script>`, `npx`, vitest, vite and
  eslint do not touch the junction.
- If it became a real directory by mistake:
  `robocopy web\node_modules D:\neshter\web\node_modules /E /MOVE /NFL /NDL /NJH /NJS /NP`, then recreate the junction.
- OneDrive: pause sync or exclude `web/node_modules` during web work; retry a `vite build` once on `EBUSY`/`EPERM`.

## ESLint and the repo folder name

`@eslint/config-array`'s Windows `path.relative` lowercases whole paths and slices by index. This repo's folder
(`NÖRAL SİNİR AĞI GEÇER`) contains `İ` (U+0130), which is two UTF-16 units when lowercased, so the offsets drift and
the first characters of the base-relative path are lost (`src/core/zod.ts` -> `c/core/zod.ts`, `e2e/x.spec.ts` ->
`e/x.spec.ts`). Top-level-anchored `files`/`ignores` globs then silently stop matching.

- Root-cause workaround: the top of `eslint.config.js` makes U+0130 lowercase to a single character in the eslint
  process (only when the engine would otherwise expand it). With it, literal globs such as `src/core/**` and
  `e2e/**` match again.
- Pattern rule anyway: write scoped overrides with a `**/` prefix (`**/core/**/*.{ts,tsx}`, `**/app/sessionStorage.ts`,
  `**/__tests__/**`, `**/e2e/**`, `**/__fixtures__/**`) so they also survive a regression of the workaround; never rely on a
  top-level literal segment alone.
- Check: `scripts/__tests__/lint-scope.test.ts` (runs in `npm --prefix web run test`) lints snippets under e2e/, src/core,
  src/app/sessionStorage.ts etc. through the ESLint API (`localStorage` allowed and `sessionStorage` still banned in tests
  and e2e; react/zod banned in core; fixtures ignored). Manual equivalent:
  `echo 'export const a = localStorage.length;' | npx eslint --stdin --stdin-filename e2e/x.spec.ts` must print nothing.

## Pinned versions

Rule: newest release published on or before 2026-10-28 within the allowed major (exact pins, `save-exact=true`).
`npm install` resolved without `ERESOLVE`, no `--force` / `--legacy-peer-deps`; no package had to step down a minor.
`@testing-library/dom` is not in the plan list; it is the required peer of `@testing-library/react@16` (10.x only).

| Package | Version | Published |
|---|---|---|
| react | 19.3.1 | 2026-10-14 |
| react-dom | 19.3.1 | 2026-10-14 |
| zustand | 5.0.16 | 2026-10-22 |
| zod | 4.6.7 | 2026-10-12 |
| @fontsource-variable/inter | 5.3.0 | 2026-07-19 |
| @fontsource-variable/space-grotesk | 5.3.0 | 2026-07-19 |
| @fontsource-variable/jetbrains-mono | 5.3.0 | 2026-07-19 |
| typescript | 6.0.3 | 2026-04-16 |
| vite | 8.3.5 | 2026-10-21 |
| @vitejs/plugin-react | 6.1.2 | 2026-10-05 |
| vitest | 5.0.5 | 2026-10-15 |
| jsdom | 30.1.3 | 2026-10-15 |
| eslint | 9.39.5 | 2026-07-10 |
| @eslint/js | 9.39.5 | 2026-07-10 |
| typescript-eslint | 8.72.1 | 2026-10-26 |
| eslint-plugin-react-hooks | 7.1.1 | 2026-04-17 |
| eslint-plugin-jsx-a11y | 6.10.2 | 2024-10-26 |
| @testing-library/react | 16.3.3 | 2026-08-27 |
| @testing-library/dom | 10.4.2 | 2026-09-13 |
| @testing-library/user-event | 14.6.8 | 2026-10-13 |
| @testing-library/jest-dom | 7.0.1 | 2026-08-09 |
| @playwright/test | 1.64.1 | 2026-10-14 |
| @axe-core/playwright | 4.14.0 | 2026-10-15 |
| @types/react | 19.3.0 | 2026-09-09 |
| @types/react-dom | 19.3.0 | 2026-09-09 |
| @types/node | 24.19.2 | 2026-10-09 |

Peer checks: vitest 5.0.5 accepts vite `^8`; @vitejs/plugin-react 6.1.2 needs vite `^8`; typescript-eslint 8.72.1
needs typescript `<6.1.0`; eslint-plugin-react-hooks 7.1.1 accepts eslint 9; jsx-a11y 6.10.2 stops at eslint 9.

## Junction check after the first `npm ci` (E2)

Task 1 install result: after `npm --prefix D:\neshter\web ci`, `(Get-Item web\node_modules).LinkType` = `Junction`
and `D:\neshter\web\node_modules\react\package.json` present. Running `npm ci` directly in `web/` broke the
junction (see above), which is why the D: mirror procedure exists.
