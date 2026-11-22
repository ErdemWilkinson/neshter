# Neshter web UI: developer notes

All commands run from the repo root. You need Node 24+ and the Python project set up (`uv sync`).

## Dev loop

1. Backend: `neshter serve --port 8765 --no-browser`. Copy the printed token.
2. Frontend: `NESHTER_DEV_TOKEN=<token> NESHTER_DEV_PORT=8765 npm --prefix web run dev`
   (Vite proxies `/api` to `127.0.0.1:$NESHTER_DEV_PORT`, drops `Origin` and adds `X-Neshter-Token`).
3. Checks: `npm --prefix web run test`, `npm --prefix web run lint`, `npm --prefix web run build`.
   `build` writes `neshter/web/dist` (plus `build-info.json`) and runs `scripts/check-dist.mjs`.

Dependencies are pinned exactly (`web/.npmrc` sets `save-exact=true`); `npm ci` reproduces the lock file.

## The committed `dist`

`neshter/web/dist` is committed, so a clone runs without Node. `dist/build-info.json` holds a hash of the sources, and
`tests/test_web_dist.py` fails when `web/` changed without a rebuild. After touching anything under `web/src`,
`index.html`, `vite.config.ts`, `tsconfig.json` or `package-lock.json`, run `npm --prefix web run build` and commit
`neshter/web/dist` together with the change. `npm --prefix web run build:mobile` builds the phone variant into
`mobile/web-dist/` (not committed).

## API contract fixtures

`src/api/__fixtures__/*.json` are shape-only copies of live server responses (lists cut to 2 items, strings to 200
characters, absolute paths replaced by `C:\\Users\\x\\model`). `tests/test_web_contract.py` compares their key shapes with a
tiny model behind `TestClient`. After a server response changes, regenerate and review the diff:

```bash
NESHTER_UPDATE_FIXTURES=1 CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider tests/test_web_contract.py
```

Then update `src/api/schemas.ts` (`FIXTURE_SCHEMAS` must name every fixture). UI tests that need specific values keep
hand-made fixtures next to the test.

## Language

The UI starts in English; the language picker in the header switches to Turkish. Every string lives in
`src/i18n/en.json` and `src/i18n/tr.json`; `npm run lint` also checks that both have the same keys and placeholders.
The server's job progress messages are Turkish; `src/api/jobMessage.ts` translates them for the English UI, and a test
scans the Python sources so a new message without a translation fails.

## ESLint on Windows

`@eslint/config-array` lowercases whole paths on Windows and slices them by index. A repo folder containing `İ`
(U+0130) breaks that, and top-level-anchored globs then stop matching. `eslint.config.js` works around it; keep writing
scoped overrides with a `**/` prefix (`**/core/**/*.{ts,tsx}`, `**/__tests__/**`, `**/e2e/**`). The check is
`scripts/__tests__/lint-scope.test.ts`, which runs with the normal test command.

## End-to-end tests (Playwright)

`npm --prefix web run e2e` runs Chromium only, with one worker, against a **real** Neshter server and the tiny random
model from `tests/tiny_models.py`. Build first (`npm --prefix web run build`): the server serves `neshter/web/dist`.

- One-time browser install: `npm --prefix web exec -- playwright install chromium`.
- `e2e/serve_e2e.py --port P --workdir W [--token T]` builds the model once under `W/hub/...` (kept across restarts, so
  the SHA stays stable), sets `NESHTER_HOME=W/home` and serves it. `e2e/server.ts` starts it on a free port and kills the
  process tree afterwards; `NESHTER_PYTHON` overrides the interpreter. Specs wipe their own work folder with
  `freshWorkdir(name)`. The token is passed as `--token=<t>` because a base64url token can start with `-`.
- The specs run the UI in Turkish (`server.ts` saves `lang: 'tr'` in the preferences before the first page load).
- Specs: `happy-path` (Home → Explore → measure → cut → Compare → Save), `restore` (server restart on another port, then
  on the same port with a new token; allow about 25 s for the 10 s heartbeat), `leave` (second tab, leave dialog,
  `beforeunload`, quit), `offline-csp` (only 127.0.0.1 requests, no `securitypolicyviolation`, empty `localStorage`,
  bundled fonts) and `a11y` (axe on every screen, list parity, reduced motion, focus ring).
- `offline-csp` is the only proof that zod's jitless mode works under the real CSP (jsdom does not enforce CSP): run it
  before every `dist` commit. The Vite dev proxy injects the token for **dev only**; the built app always reads
  `#token=` itself.
