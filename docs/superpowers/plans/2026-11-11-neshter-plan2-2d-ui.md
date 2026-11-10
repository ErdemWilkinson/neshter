# Neşter Plan 2 — 2D React UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the local 2D web UI of Neşter (Karşılama → Model seç → Ağ haritası / Davranış bul → Karşılaştır → Kaydet) on top of backend v0.1.1, built so Plan 3's lazy-loaded 3D view plugs into the same stores, ids, basket and save flow without rework.

**Architecture:** A Vite + React + TypeScript app in `web/` builds into `neshter/web/dist`, which FastAPI already serves (`WEB_DIST` in `neshter/server/app.py`) under the existing CSP. Domain state lives in framework-free Zustand vanilla stores in `web/src/core` (canonical node ids `h:L:H`, a basket of ops with `isSavable` and an explore shelf, a pure save-flow reducer); React screens only read and dispatch. The server stays stateless about cuts; it only mirrors drafts and prefs on disk keyed by model SHA-256 so restore survives the random port, and the 2D real-effect measurement moves to a sequential Besag–Clifford permutation test.

**Tech Stack:** Node 24 / npm 11 (dev only), React 19, TypeScript 6.0 (strict), Vite 8, Zustand 5 (vanilla stores), zod 4 (jitless), Vitest + jsdom + Testing Library, ESLint 9 flat config, Playwright (Chromium only) + axe-core, `@fontsource-variable` Inter / Space Grotesk / JetBrains Mono (latin + latin-ext files only). Backend: Python ≥ 3.11, FastAPI, pydantic v2, pytest (existing).

**Spec:** `docs/superpowers/specs/2026-11-10-neshter-design.md` (flows, screens, copy, i18n, safety). Binding amendments: `.superpowers/sdd/2026-11-11-decisions-debate/synthesis.md`, `ux-architect.md` (§ refs "UX"), `interp-critic.md` and `scope-skeptic.md` (Round 2 sections override Round 1), and for design tokens `docs/superpowers/specs/2026-11-10-neshter-3d-graph-design.md` §8 ("3D §8"). Plan-2 items from `.superpowers/sdd/2026-11-10-neshter-backend/audit-roadmap.md` ("AR", §2 and §4) and `audit-security-product.md` ("AS"). Executors read the spec and the section a task cites.

## Global Constraints

**Environment (every task)**
- Repo path has spaces and Turkish letters (`...\NÖRAL SİNİR AĞI GEÇER`): quote it everywhere. Shell examples are Git Bash unless marked PowerShell.
- `<PYTEST>` = `CUDA_VISIBLE_DEVICES= D:/neshter/venv/Scripts/python -m pytest -q -p no:cacheprovider` (run from the repo root). No GPU, no WSL, no model downloads; Python tests use the tiny random models from `tests/conftest.py` in float32.
- `<WEBENV>` = `npm_config_cache='D:\neshter\npm-cache' PLAYWRIGHT_BROWSERS_PATH='D:\neshter\pw-browsers'`. Every `npm`/`npx` command is prefixed with `<WEBENV>` and run as `npm --prefix web ...` from the repo root (npm sets the script cwd to `web/`).
- `web/node_modules` is a directory junction to `D:\neshter\web-node_modules` (created in Task 1). Nothing is cached or installed on C:. Vite/Vitest caches stay under `node_modules/` (so on D:), Playwright `outputDir` is `node_modules/.pw-results`, ESLint runs without `--cache`, TypeScript runs `--noEmit` without incremental build info.
- Never touch `D:/datasets`, `D:/vox_*`, or any folder named `AI EĞİTİM`. E2E scratch lives only in `D:\neshter\e2e-work`.
- Work happens on branch `feat/plan2-ui` (Task 1 creates it from `main`). Commit steps list explicit paths; never `git add -A` or `git add .`.

**Dependencies**
- Exact versions in `web/package.json` (no `^`/`~`; `web/.npmrc` has `save-exact=true` and `engine-strict=true`), lock committed, `npm --prefix web ci` after the lock exists.
- Version rule: per package, the newest release **published on or before 2026-10-28** (≥ 2 weeks before plan date) within these majors: react/react-dom 19, @types/react(-dom) 19, typescript **6.0.x** (typescript-eslint's peer range is `<6.1.0`), vite 8, @vitejs/plugin-react (major whose peer accepts vite 8), vitest (major whose peer accepts vite 8), jsdom (any), zustand 5, zod 4, eslint **9.x** (eslint-plugin-jsx-a11y's peer stops at 9), @eslint/js 9, typescript-eslint 8, eslint-plugin-react-hooks (major with eslint 9 peer), eslint-plugin-jsx-a11y 6, @testing-library/react 16, @testing-library/user-event 14, @testing-library/jest-dom (any), @playwright/test 1.x, @axe-core/playwright 4.x, @fontsource-variable/{inter,space-grotesk,jetbrains-mono} 5, @types/node 24. Install must succeed without `--force`/`--legacy-peer-deps`; if peers conflict, step down one minor and record why in `web/DEV.md`.
- Allowed runtime deps are exactly: react, react-dom, zustand, zod, the three fontsource packages. No router library (own hash router), no i18n library, no CSS-in-JS runtime, no three.js / R3F / drei in Plan 2.

**Shipping the bundle (decision)**
- `neshter/web/dist` **is committed** to git. Reason: end users run `uv run --frozen neshter` / `neshter.bat` from a clone or ZIP with no Node; building in the launcher would need Node + npm + network on every user machine, which breaks the local one-click story. `.gitignore` already anchors `/dist/` (the AR R3 trap is fixed) and `pyproject.toml` already lists `artifacts = ["neshter/web/dist/**"]`; Task 1 pins both with tests. To keep history small, `dist` is built locally in every task but committed **only in Task 15** (and at each release); `dist/build-info.json` carries a source hash and a pytest fails when the committed bundle is stale.

**CSP and locality**
- Server CSP (Task 1): `default-src 'self'; script-src 'self'; worker-src 'self'; font-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; object-src 'none'`.
- Build rules: no inline `<script>`; `build.assetsInlineLimit: 0`; no `data:` fonts; no `?worker&inline` / blob workers; no WASM; no request to any host but the serving origin; no CDN, no Google Fonts. `z.config({ jitless: true })` runs before any schema is used (zod's eval probe would otherwise raise a CSP violation).
- No `dangerouslySetInnerHTML`, no `innerHTML`. Model output, token pieces, model names, recipe text and server `detail` are attacker-influenced (AS S3) and are always rendered as React text.

**Security and persistence**
- The token is read from `#token=`, kept in memory and `sessionStorage` key `neshter.token`, then `history.replaceState(null, '', pathname + '#/')`. **Never `localStorage`**: ESLint bans `localStorage` everywhere; `sessionStorage` is touched only in `src/app/sessionStorage.ts` (keys `neshter.token`, `neshter.activeJob`).
- Drafts and prefs are stored **server-side** under `<NESHTER_HOME>/.state/` (default `~/Neshter/.state/`), keyed by model SHA-256 (`^[0-9a-f]{64}$`). Drafts hold ops, prompts and the model name/SHA; never `source_path`, never model output text.
- **Invariant:** the client basket is the only source of truth for cuts. The server never holds cuts; the draft store is a mirror for restore.

**Domain contracts shared with Plan 3**
- Canonical ids: `h:{layer}:{head}`, `n:{layer}:{index}`, `t:{pos}`, `tid:{token_id}`, `out`, `e:{edge_id}`. A heatmap cell **is** node `h:L:H`.
- Stores are Zustand **vanilla** stores in `web/src/core/stores` with no React import (ESLint-enforced for `src/core/**`).
- Ops: `HeadOp {type:"head", layer, head}`, `DirectionOp {type:"direction", d_model, vector_b64}` mirror `neshter/recipe.py` exactly (zod `.strict()`); other types go through `registerOpKind`; unknown types survive as opaque "bu sürümde desteklenmeyen kesim" items and are never sent to the server.
- The view switch has `2D` and `Liste`; a registered but disabled `3D` slot is not rendered.

**Copy (exact; tr / en) — every user-visible string comes from `web/src/i18n/{tr,en}.json`**
- Cut button `Kes (geçici)` / `Cut (temporary)`, sub-line `Dosya değişmez` / `The file is not changed`.
- Header chip `Kaydedilmemiş {n} kesim` / `{n} unsaved cuts`.
- Leave dialog title `Ayrılıyorsunuz, kesimlerinizi daha kaydetmediniz` / `You are leaving and your cuts are not saved yet`; buttons `Kaydet` / `Kaydetmeden çık` / `Vazgeç` (en `Save` / `Leave without saving` / `Cancel`).
- Restore prompt `Önceki oturumdan kaydedilmemiş {n} kesim var: geri yükle?` / `{n} unsaved cuts from an earlier session: restore?`.
- Discard dialog `Şu {n} kesim yalnızca keşif içindir ve kaydedilemez: {list}. Bunlar sepetten atılacak, karşılaştırma yeniden koşulacak. Devam?`; only-explore case `Kaydedilecek kesim kalmıyor.`
- After re-run `Sayılar yenilendi; {n} kesim atıldı.` Basket line `Kaydedilecek: {s} · Yalnızca keşif: {e}`. Save button `Modeli kaydet ({n} kesim)`.
- Measurement: significant `Seçilen head'ler, aynı katmanlardan rastgele seçilen head'lerden daha etkili (p = {p}, {n} çekiş).`; full run not significant `p = {p} > {tau}: rastgele seçimden ayırt edilemedi ({n} çekiş).`; futility `p > {tau}, {n} çekişte durdu ({k} aşım): rastgele seçimden ayırt edilemedi.`; session line `Bu oturumda {k} ölçüm yapıldı; seçilerek bulunan bir sonuç hipotezdir.`
- Head silencing is described as zero-ablation: `Head'in çıktısı sıfırlanır (o_proj sütunları sıfır).` Never "ortalama değeriyle".
- Heatmap measure label `Head çıktı gücü, katman içinde 0–1` with the badge `tahmini`.
- Stale/closed: `Oturum bitti, sunucu yeniden başlatılmış`, `Sunucu kapandı; kesimlerin taslakta`, multi-tab `Neşter başka bir sekmede açık`.
- Terminal line on Ctrl+C (Python): `{n} kaydedilmemiş kesim taslağa yazıldı; sonraki açılışta geri yüklenir.`
- Forbidden (AR §2.4): invented ETAs ("~2 dk"), "Kaynağa güveniyorum", "Güvenli formata dönüştür", "katmanları sırayla yükler", "MLP nöronları" tab, "Kendi test setini yükle", attention panel, rate-level random-control line, "Modeli küçült".
- Language: `prefs.lang`, else `tr` when `navigator.language` starts with `tr`, else `en`; `<html lang>` follows it. Server `detail` text appears only inside the "Teknik ayrıntı" disclosure; everything else is chosen by `error_code`.

**Look (3D §8 tokens, one `tokens.css`)**
- `--surface-0 #0B0E13`, `--surface-1 #121722` (glass panel, `backdrop-filter: blur(12px)`), `--border #222A38` (decorative only), `--border-control #5F6B7F` (inputs/buttons, ≥ 3:1 on `--surface-0`), `--text-1 #E6EAF2`, `--text-2 #A7B0C0`, `--accent-increase #2EE6C5`, `--accent-decrease #9FC3FF`, `--accent-basket #FF7A45` (basket only), `--badge-explore #FFD27A`, `--focus-ring #9FC3FF`. Headings Space Grotesk, text Inter, numbers JetBrains Mono. `html[data-motion="reduce"]` disables transitions.

**Python side**
- Server messages stay Turkish; `error_code` stays English snake_case; new codes: `too_large` (413), `unsupported` (501).
- Existing guard, `trust_remote_code=False`, atomic writes (`.tmp` → `os.replace`) and the source-untouched check are unchanged.

## Review Focus

1. **Attacker-controlled text** (a token piece like `<img src=x onerror=alert(1)>`, model output with HTML, a model or recipe name with markup): it must appear as literal text and never create elements. → Task 10 `token strip renders markup as text`, Task 13 `pair view renders model output as text`, Task 9 `model name with markup renders as text`.
2. **Server restarted while a tab is open** (new port, new token): the old tab must show one "Oturum bitti" screen instead of a stream of errors, and the cuts must come back from the server draft on the new origin. → Task 5 `403 marks the session stale once`, Task 15 `restore.spec.ts` (restart on a different port).
3. **A draft or recipe holding an op type this version does not know** (written by Plan 3): it must survive as an "unsupported" row, count as not savable, and never be sent to compare, suite or export. → Task 6 `decodeDraft keeps unknown ops opaque`, Task 13 `compare body excludes opaque and pending-discard ops`.
4. **The model is busy (409)** because a hunt or measurement holds the lock: the user sees a "model meşgul" banner and disabled actions, not an error card. → Task 5 `409 busy throws BusyError and calls onBusy`, Task 10 `run button disabled while a job holds the model`, Task 13 `compare shows busy banner on 409`.
5. **Unusual model shapes** (1 layer × 1 head, 3 × 5, 32 × 32; layer scores with `n_layers + 1` entries): layout, keyboard navigation and labels must stay in bounds. → Task 10 `keyboard navigation clamps at edges for 1x1 and 3x5`, Task 12 `layer chart labels n_layers + 1 bars`.

---

## File map

| Path | Responsibility |
|---|---|
| `web/package.json`, `package-lock.json`, `.npmrc`, `tsconfig.json`, `vite.config.ts`, `eslint.config.js`, `vitest.setup.ts`, `index.html`, `DEV.md` | toolchain, pins, CSP-safe build, dev loop |
| `web/scripts/{source-hash.mjs,write-build-info.mjs,check-dist.mjs,check-i18n.mjs}` | build guards |
| `web/src/styles/{tokens.css,fonts.css,base.css}` | design tokens, bundled fonts |
| `web/src/api/*` | typed client, zod response schemas, SSE, job runner, fixtures |
| `web/src/app/*` | bootstrap, router, shell, header, single-tab lock, stale screens |
| `web/src/core/*` | ids, ops, basket, measurement, recipe, draft codec, save-flow reducer, normalisation, vanilla stores |
| `web/src/i18n/*` | `t()`, tr/en catalogs |
| `web/src/ui/*` | Button, Dialog, LiveRegion, ProgressCard, ErrorCard, TechDetail, Segmented |
| `web/src/features/{draft,leave,saveflow,models,recipe,map,measure,basket,concept,compare,save}/*` | feature components |
| `web/src/screens/{Home,Models,Explore,Concept,Compare,Save}.tsx` | screens |
| `web/e2e/*`, `web/playwright.config.ts` | end-to-end, a11y, offline/CSP |
| `neshter/server/state.py` | **new**: `neshter_home`, drafts/prefs store |
| `neshter/server/app.py`, `errors.py`, `neshter/cli.py`, `neshter/metrics.py` | prerequisites (Tasks 1–4, 14) |
| `tests/tiny_models.py` | **new**: tiny model builder shared by conftest and E2E |
| `tests/{test_state.py,test_web_dist.py,test_web_contract.py}` | **new** Python tests |
| `neshter/web/dist/**` | built bundle (committed in Task 15) |

---

### Task 1: Workspace on D:, pinned scaffold, CSP-safe build served by FastAPI

UX §3.8, §3.4, §2.8 (fonts); AR R3, B21; AS S2.

**Files:**
- Create: `web/package.json`, `web/package-lock.json`, `web/.npmrc`, `web/tsconfig.json`, `web/vite.config.ts`, `web/eslint.config.js`, `web/vitest.setup.ts`, `web/index.html`, `web/DEV.md`, `web/src/main.tsx`, `web/src/App.tsx`, `web/src/styles/{tokens.css,fonts.css,base.css}`, `web/scripts/{source-hash.mjs,write-build-info.mjs,check-dist.mjs}`, `web/scripts/__tests__/check-dist.test.ts`, `web/src/__tests__/smoke.test.tsx`, `tests/test_web_dist.py`
- Modify: `.gitignore` (add `web/node_modules`), `neshter/server/errors.py` (`SECURITY_HEADERS`)

**Interfaces:**
- Produces:
  - npm scripts: `dev`, `build` (= `tsc --noEmit -p tsconfig.json && vite build && node scripts/write-build-info.mjs && node scripts/check-dist.mjs`), `test` (= `vitest run`), `lint` (= `eslint .`), `e2e` (= `playwright test`, Task 15).
  - `scripts/source-hash.mjs`: `export function sourceHash(webDir: string): string` — files = everything under `src/` and `public/` (if present) plus `index.html`, `vite.config.ts`, `tsconfig.json`, `package-lock.json`; exclude paths containing `/__tests__/`, `/__fixtures__/` or `.test.`; sort by POSIX path relative to `web/`; digest = sha256 over `relpath + "\0" + sha256hex(content with CRLF→LF) + "\n"` per file; hex.
  - `dist/build-info.json` = `{ "source_hash": string, "built_at": ISO string }`.
  - `scripts/check-dist.mjs`: `export function checkDist(dir: string, opts?: {entryBudgetGz?: number}): string[]` (error list; CLI exits 1 if non-empty). Rules: every `<script>` in `index.html` has `src` and `type="module"`; no `http://`, `https://` or protocol-relative `//host` in `index.html` or any `.css`; no `data:font`/`data:application/font`/`data:application/x-font` anywhere; no `createObjectURL(new Blob` in any `.js`; at least one emitted font filename contains `latin-ext`; the entry chunk (the `src` of the module script) gzipped ≤ `entryBudgetGz` (default 250 000 bytes).
  - Vite config: `build.outDir: '../neshter/web/dist'`, `emptyOutDir: true`, `assetsInlineLimit: 0`, `modulePreload: { polyfill: false }`, `server.fs.allow` includes `D:/neshter/web-node_modules`; resolve aliases `@fonts/inter`, `@fonts/space-grotesk`, `@fonts/jetbrains-mono` → `node_modules/@fontsource-variable/<pkg>/files`; dev proxy `/api` → `http://127.0.0.1:${NESHTER_DEV_PORT}` with `changeOrigin: true` and a `proxyReq` hook that removes `origin` and sets `x-neshter-token` from `process.env.NESHTER_DEV_TOKEN`.
  - `fonts.css`: hand-written `@font-face` rules for only `*-latin-wght-normal.woff2` and `*-latin-ext-wght-normal.woff2` of each family (the fontsource-variable packages ship no per-subset CSS), with the `unicode-range` values copied from the package's `wght.css`. Family names `"Inter Variable"`, `"Space Grotesk Variable"`, `"JetBrains Mono Variable"`.
  - ESLint (flat): typescript-eslint strict, react-hooks, jsx-a11y recommended; `no-restricted-globals: ['localStorage']`; `no-restricted-properties` for `window.localStorage`; `no-restricted-syntax` for `JSXAttribute[name.name='dangerouslySetInnerHTML']` and `AssignmentExpression[left.property.name='innerHTML']`; `no-restricted-imports` for `@react-three/drei`, `i18next`, `react-i18next` and the pattern `*?worker&inline`; an override for `src/core/**` banning `react`, `react-dom`, `zustand/react`; `sessionStorage` allowed only in `src/app/sessionStorage.ts`.
  - `SECURITY_HEADERS["Content-Security-Policy"]` = the Global Constraints string.

- [ ] **Step 1: Branch and D: workspace (PowerShell, repo root)**

```powershell
git switch -c feat/plan2-ui main
New-Item -ItemType Directory -Force D:\neshter\web-node_modules, D:\neshter\npm-cache, D:\neshter\pw-browsers | Out-Null
New-Item -ItemType Directory -Force web | Out-Null
cmd /c mklink /J "web\node_modules" "D:\neshter\web-node_modules"
```
Expected: `Junction created for web\node_modules <<===>> D:\neshter\web-node_modules`. Add `web/node_modules` to `.gitignore`. Verify: `git check-ignore -q web/node_modules/x; echo $?` → `0`.

- [ ] **Step 2: Choose versions.** For each package in Global Constraints run `<WEBENV> npm view <pkg> time --json` and take the newest version in the allowed major dated ≤ 2026-10-28. Write the table (package, version, publish date) into `web/DEV.md` together with the dev loop (backend `neshter serve --port 8765 --no-browser`, copy the printed token into `NESHTER_DEV_TOKEN`, `NESHTER_DEV_PORT=8765 npm --prefix web run dev`) and the junction/cache rules.

- [ ] **Step 3: Install once to create the lock.** Create `web/package.json` (`"private": true`, `"type": "module"`, `"engines": {"node": ">=24"}`) and `web/.npmrc`, then `<WEBENV> npm --prefix web install --save-exact <pkg>@<ver> ...` (runtime) and `--save-dev` (dev). Expected: no `ERESOLVE`, `web/package-lock.json` exists, `ls D:/neshter/web-node_modules/react` lists files. From now on use `<WEBENV> npm --prefix web ci`.

- [ ] **Step 4: Write the failing tests**

`web/scripts/__tests__/check-dist.test.ts`:
```ts
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
test('clean dist passes', () => { expect(checkDist(makeTmpDist(cleanFixture()))).toEqual([]); });
```
(`makeTmpDist` writes under `node_modules/.tmp-tests/`, i.e. on D:.)

`web/src/__tests__/smoke.test.tsx`: `render(<App />)` → `screen.getByText('Neşter')`.

`tests/test_web_dist.py`:
```python
def test_csp_states_script_worker_font_self():
    csp = SECURITY_HEADERS["Content-Security-Policy"]
    for part in ("script-src 'self'", "worker-src 'self'", "font-src 'self'", "img-src 'self' data:", "connect-src 'self'"):
        assert part in csp
    assert "unsafe-eval" not in csp and "wasm-unsafe-eval" not in csp

def test_serves_built_index_with_csp(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<!doctype html><title>Neşter</title>", encoding="utf-8")
    monkeypatch.setattr(app_module, "WEB_DIST", tmp_path)
    r = TestClient(create_app("t", 47998, roots={}), base_url="http://127.0.0.1:47998").get("/")
    assert r.status_code == 200 and "Neşter" in r.text
    assert r.headers["content-security-policy"] == SECURITY_HEADERS["Content-Security-Policy"]

def test_built_ui_path_not_gitignored():   # skip if git is unavailable
    assert subprocess.run(["git", "check-ignore", "-q", "neshter/web/dist/index.html"], cwd=ROOT).returncode == 1

def test_wheel_artifacts_include_dist():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "neshter/web/dist/**" in cfg["tool"]["hatch"]["build"]["targets"]["wheel"]["artifacts"]
```

- [ ] **Step 5: Run to verify failure**

Run: `<WEBENV> npm --prefix web run test` → FAIL (`Cannot find module '../check-dist.mjs'`, `App`). Run: `<PYTEST> tests/test_web_dist.py` → FAIL on `script-src`.

- [ ] **Step 6: Implement** the config, scripts, `App` (renders the word `Neşter` inside `<main>`), `main.tsx` (imports `zod` config first, then styles, then `App`), `tokens.css`/`fonts.css`/`base.css` per Global Constraints, and the new CSP string in `errors.py`.

- [ ] **Step 7: Verify**

Run: `<WEBENV> npm --prefix web run test` → all pass. `<WEBENV> npm --prefix web run lint` → no errors. `<WEBENV> npm --prefix web run build` → `check-dist: ok`, `neshter/web/dist/index.html` and `build-info.json` exist. `<PYTEST>` → all pass (811 + 4 new). Manual: `D:/neshter/venv/Scripts/neshter.exe serve --no-browser`, open the printed URL in Chromium, DevTools console shows no CSP error and the Network tab shows only `127.0.0.1` requests.

- [ ] **Step 8: Commit** (not `neshter/web/dist`)

```bash
git add .gitignore web/package.json web/package-lock.json web/.npmrc web/tsconfig.json web/vite.config.ts web/eslint.config.js web/vitest.setup.ts web/index.html web/DEV.md web/src web/scripts neshter/server/errors.py tests/test_web_dist.py
git commit -m "feat(web): pinned Vite/React scaffold on D: junction, CSP-safe build, explicit CSP directives"
```

---

### Task 2: Server-side drafts and prefs, `source_sha256`, `NESHTER_HOME`, estimate safety flag

UX §1.1, §1.3, §1.4; AS §2.5 item 4.

**Files:**
- Create: `neshter/server/state.py`, `tests/test_state.py`
- Modify: `neshter/server/app.py` (`create_app`, `get_session`, `load`, `get_system`, `export_estimate`), `neshter/server/errors.py` (`_STATUS_CODES[413] = "too_large"`), `tests/test_server.py`

**Interfaces:**
- Produces (`neshter/server/state.py`):
  - `neshter_home() -> Path`: `Path(os.environ["NESHTER_HOME"])` if set and non-empty, else `Path.home() / "Neshter"`. Read at call time.
  - Constants `SHA_RE = r"^[0-9a-f]{64}$"`, `DRAFT_MAX_BYTES = 8 * 1024 * 1024`, `DRAFT_KEEP = 10`, `DRAFT_MAX_AGE_DAYS = 60`.
  - `class DraftIn(BaseModel, extra="forbid")`: `schema_version: Literal[1]`, `sha256: str = Field(pattern=SHA_RE)`, `model_name: str = Field(max_length=256)`, `unsaved_count: int = Field(ge=0, le=10_000)`, `basket: list[dict[str, Any]] = Field(max_length=1000)`, `shelf: list[dict[str, Any]] = Field(max_length=1000)`, `saved_ops: list[str] = Field(max_length=1000)`, `prompts: list[Annotated[str, Field(max_length=8000)]] = Field(max_length=20)`. Ops inside are opaque to the server.
  - `class Draft(DraftIn)`: `updated_at: AwareDatetime`. `class DraftSummary(BaseModel)`: `sha256, model_name, updated_at, unsaved_count`.
  - `class Prefs(BaseModel, extra="forbid")`: `lang: Literal["tr", "en"] | None = None`, `quality: Literal["high", "balanced", "low"] = "balanced"`, `reduced_motion: Literal["system", "on", "off"] = "system"`, `refusal_card_seen: bool = False`.
  - `class StateStore(root: Path)`: lazy (no directory until the first write); files `root/drafts/<sha>.json`, `root/prefs.json`; `_atomic_write(path, data: bytes)` via `<file>.tmp` + `os.replace`; a corrupt or schema-invalid file is logged and treated as missing.
    - `list_drafts(now: datetime | None = None) -> list[DraftSummary]` (newest first; deletes files older than 60 days)
    - `get_draft(sha: str) -> Draft | None`
    - `put_draft(sha: str, d: DraftIn, now: datetime | None = None) -> Draft` (`ValueError` if `d.sha256 != sha`; after writing keeps the newest 10)
    - `delete_draft(sha: str) -> bool`
    - `get_prefs() -> Prefs`, `put_prefs(p: Prefs) -> Prefs`
    - `unsaved_written() -> int` (sum over SHAs written by this process of the latest `unsaved_count`; a delete removes the SHA)
    - `exit_line() -> str | None` (`None` when `unsaved_written() == 0`, else the exact Ctrl+C sentence from Global Constraints)
- Routes (token-guarded like every `/api` route; `sha` is a path param with `pattern=SHA_RE`):
  - `GET /api/drafts` → `list[DraftSummary]`
  - `GET /api/draft/{sha}` → `Draft`, or 404 `not_found`
  - `PUT /api/draft/{sha}` (body `DraftIn`) → `Draft`; 413 `too_large` when the body exceeds `DRAFT_MAX_BYTES` (check `content-length` and the read length before parsing); 422 on SHA mismatch
  - `DELETE /api/draft/{sha}` → 204
  - `GET /api/prefs` → `Prefs`; `PUT /api/prefs` (body `Prefs`) → `Prefs`
- `create_app` builds `app.state.store = StateStore(neshter_home() / ".state")`.
- `/api/session` (loaded) gains `"source_sha256": lm.source_sha256`; the load job result gains `"source_sha256"`.
- `/api/system.default_out_dir = str(neshter_home())`; `export_estimate` falls back to `neshter_home()`.
- `/api/export/estimate` response gains `"safety_ablated": bool` (the `safety` value it already computes).

- [ ] **Step 1: Write the failing tests**

`tests/test_state.py` (each test uses `tmp_path`):
```python
SHA = "a" * 64
def draft_in(sha=SHA, n=1): return DraftIn(schema_version=1, sha256=sha, model_name="m", unsaved_count=n,
                                           basket=[{"type": "head", "layer": 0, "head": 1}], shelf=[], saved_ops=[], prompts=["merhaba"])
def test_lazy_no_dir_until_write(tmp_path): StateStore(tmp_path / "s").list_drafts(); assert not (tmp_path / "s").exists()
def test_put_get_roundtrip(tmp_path):
    s = StateStore(tmp_path); d = s.put_draft(SHA, draft_in())
    assert s.get_draft(SHA).basket == d.basket and d.updated_at.tzinfo is not None
def test_put_rejects_sha_mismatch(tmp_path):
    with pytest.raises(ValueError): StateStore(tmp_path).put_draft("b" * 64, draft_in())
def test_atomic_write_leaves_no_tmp(tmp_path):
    s = StateStore(tmp_path); s.put_draft(SHA, draft_in()); assert not list(tmp_path.rglob("*.tmp"))
def test_keeps_newest_ten(tmp_path):   # 12 SHAs written at t0 + i minutes
    assert [x.sha256 for x in s.list_drafts(now=t0 + timedelta(minutes=20))] == shas[::-1][:10]
def test_prunes_older_than_60_days(tmp_path):
    s.put_draft(SHA, draft_in(), now=t0); assert s.list_drafts(now=t0 + timedelta(days=61)) == []
    assert s.get_draft(SHA) is None
def test_corrupt_file_is_ignored(tmp_path):
    (tmp_path / "drafts").mkdir(); (tmp_path / "drafts" / f"{SHA}.json").write_text("{bozuk")
    assert StateStore(tmp_path).get_draft(SHA) is None and StateStore(tmp_path).list_drafts() == []
def test_prefs_defaults_and_roundtrip(tmp_path):
    s = StateStore(tmp_path); assert s.get_prefs() == Prefs()
    s.put_prefs(Prefs(lang="en", refusal_card_seen=True)); assert StateStore(tmp_path).get_prefs().lang == "en"
def test_unsaved_written_latest_per_sha_and_delete(tmp_path):
    s.put_draft(SHA, draft_in(n=3)); s.put_draft(SHA, draft_in(n=2)); s.put_draft("b" * 64, draft_in("b" * 64, 1))
    assert s.unsaved_written() == 3; s.delete_draft("b" * 64); assert s.unsaved_written() == 2
def test_exit_line(tmp_path):
    s = StateStore(tmp_path); assert s.exit_line() is None; s.put_draft(SHA, draft_in(n=2))
    assert s.exit_line() == "2 kaydedilmemiş kesim taslağa yazıldı; sonraki açılışta geri yüklenir."
def test_neshter_home_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NESHTER_HOME", str(tmp_path)); assert neshter_home() == tmp_path
```

`tests/test_server.py` (a fixture sets `NESHTER_HOME` to `tmp_path` for these):
```python
def test_draft_routes_roundtrip(home_client, auth): put → 200; get → same basket; GET /api/drafts → one summary; delete → 204; get → 404 not_found
def test_draft_bad_sha_path_is_422(home_client, auth): for bad in ("g" * 64, "A" * 64, "a" * 63): assert put(bad).status_code == 422
def test_draft_body_sha_must_match_path(home_client, auth): → 422 invalid_input
def test_draft_too_large_is_413(home_client, auth): prompts padded past 8 MB → 413, error_code "too_large"
def test_draft_unknown_field_rejected(home_client, auth): {"extra": 1} → 422
def test_prefs_roundtrip_and_bad_lang(home_client, auth): put lang "en" → get "en"; lang "de" → 422
def test_session_reports_source_sha256(ready): re.fullmatch("[0-9a-f]{64}", session["source_sha256"])
def test_system_default_out_dir_follows_neshter_home(home_client, auth, tmp_path): == str(tmp_path)
def test_estimate_reports_safety_flag(ready): concept "refusal" → safety_ablated True; HEAD_OPS without concept → False
```
Add the five new routes to the existing parametrized "token required" table.

- [ ] **Step 2: Run to verify failure** — `<PYTEST> tests/test_state.py tests/test_server.py -k "draft or prefs or sha256 or neshter_home or safety_flag or token"` → FAIL (`ModuleNotFoundError: neshter.server.state`).

- [ ] **Step 3: Implement** `state.py` and the route changes per Interfaces.

- [ ] **Step 4: Verify** — `<PYTEST>` → all pass.

- [ ] **Step 5: Commit**

```bash
git add neshter/server/state.py neshter/server/app.py neshter/server/errors.py tests/test_state.py tests/test_server.py
git commit -m "feat(server): drafts and prefs stored server-side by model SHA, source_sha256 in session, estimate safety flag"
```

---

### Task 3: `POST /api/shutdown` and the Ctrl+C draft line

UX §1.5; synthesis D.

**Files:**
- Modify: `neshter/server/app.py` (lifespan, new route), `neshter/server/errors.py` (`_STATUS_CODES[501] = "unsupported"`), `neshter/cli.py` (`_serve`), `tests/test_server.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 2 `app.state.store.exit_line()`.
- Produces:
  - `app.state.request_exit: Callable[[], None] | None` (set to `None` in `create_app`).
  - `POST /api/shutdown` → 202 `{"ok": true}` after calling `request_exit()`; 501 `unsupported` when it is `None`.
  - Lifespan shutdown: `jobs.shutdown()`, then `print(line, flush=True)` if `store.exit_line()` is not `None`.
  - `cli._run_server(app, host: str, port: int) -> None`: `server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))`; if `app` has `state`, sets `app.state.request_exit = lambda: setattr(server, "should_exit", True)`; `server.run()`. `_serve` calls it instead of `uvicorn.run`.

- [ ] **Step 1: Write the failing tests**

```python
def test_shutdown_calls_request_exit(client, auth):
    called = []; client.app.state.request_exit = lambda: called.append(1)
    r = client.post("/api/shutdown", headers=auth); assert r.status_code == 202 and called == [1]
def test_shutdown_without_hook_is_501(client, auth):
    r = client.post("/api/shutdown", headers=auth); assert r.status_code == 501 and r.json()["error_code"] == "unsupported"
def test_lifespan_prints_unsaved_line(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("NESHTER_HOME", str(tmp_path))
    with TestClient(create_app(TOKEN, PORT, roots={}), base_url=BASE) as c:
        c.put(f"/api/draft/{SHA}", json=draft_body(unsaved_count=2), headers=AUTH)
    assert "2 kaydedilmemiş kesim taslağa yazıldı; sonraki açılışta geri yüklenir." in capsys.readouterr().out
def test_lifespan_silent_without_unsaved(...): no "taslağa yazıldı" in output
```
Add `("post", "/api/shutdown")` to the token table. In `tests/test_cli.py` change the `served` fixture to patch `cli._run_server` (same `host`/`port` keyword assertions) and add:
```python
def test_run_server_wires_request_exit(monkeypatch):
    class FakeServer:
        def __init__(self, config): self.should_exit = False; FakeServer.inst = self
        def run(self): app.state.request_exit()
    monkeypatch.setattr(cli.uvicorn, "Server", FakeServer)
    app = create_app("t", 47997, roots={}); cli._run_server(app, "127.0.0.1", 47997)
    assert FakeServer.inst.should_exit is True
```

- [ ] **Step 2: Run to verify failure** — `<PYTEST> tests/test_server.py tests/test_cli.py -k "shutdown or lifespan or serve or run_server"` → FAIL (404 on `/api/shutdown`).

- [ ] **Step 3: Implement** per Interfaces.

- [ ] **Step 4: Verify** — `<PYTEST>` → all pass.

- [ ] **Step 5: Commit**

```bash
git add neshter/server/app.py neshter/server/errors.py neshter/cli.py tests/test_server.py tests/test_cli.py
git commit -m "feat(server): token-guarded shutdown route and one-line unsaved-draft notice on Ctrl+C"
```

---

### Task 4: Sequential Besag–Clifford permutation test for the 2D real effect

synthesis "Controls"; scope-skeptic Round 2 items 1–4; interp-critic Round 2 items 2–6.

**Files:**
- Modify: `neshter/metrics.py`, `neshter/server/app.py` (`EffectReq`, `effect`, `Session`, `get_session`), `tests/test_metrics.py`, `tests/test_server.py` (replace the `n_random` effect tests at the `/api/effect` shape and bounds tests)

**Interfaces:**
- Produces (`neshter/metrics.py`):
  - `DEFAULT_N_MAX = 39`, `MIN_N_MAX = 19`, `CPU_N_MAX_CAP = 199`, `TAU = 0.05`. `DEFAULT_N_RANDOM` stays for the suite path.
  - `futility_k(n_max: int, tau: float = TAU) -> int`: the largest `k` with `(1 + k) / (1 + n_max) <= tau`, i.e. `floor(tau * (n_max + 1) + 1e-9) - 1`.
  - `@dataclass PermutationOutcome(n_drawn: int, k_exceed: int, p_value: float, p_kind: Literal["full", "besag_clifford"], significant: bool, stopped_early: bool)`
  - `sequential_permutation(target: float, draws: Iterable[float], n_max: int, tau: float = TAU) -> tuple[PermutationOutcome, list[float]]`. Consume draws one at a time; a draw `>= target` is an exceedance. When `k > futility_k(n_max, tau)`: stop, `p = k / n_drawn`, `p_kind="besag_clifford"`, `significant=False`, `stopped_early=True`. After `n_max` draws: `p = (1 + k) / (1 + n_max)`, `p_kind="full"`, `significant = p <= tau`. Never draws past the stop.
  - `_draw_controls(lm, interventions, n: int, seed: int) -> tuple[list[list[Intervention]], bool]`: the control list `recipe_effect` builds today (same RNG order); `recipe_effect` is refactored to use it.
  - `@dataclass SequentialEffect(kl: float, kl_std: float, kl_se: float, n_prompts: int, random_kls: list[float], random_kl: float | None, random_p05: float | None, random_p95: float | None, percentile: float | None, n_drawn: int, n_max: int, k_exceed: int, tau: float, p_value: float, p_kind: str, p_min: float, significant: bool, stopped_early: bool, layer_matched: bool, seed: int)`; `p_min = 1 / (n_max + 1)`; band and percentile over the drawn values (`None` when `n_drawn < 2`).
  - `sequential_real_effect(lm, heads: list[HeadAblation], prompts: list[str], n_max: int = DEFAULT_N_MAX, tau: float = TAU, seed: int = 0, progress: ProgressFn | None = None, cancel: threading.Event | None = None) -> SequentialEffect`; `ValueError` if `n_max < MIN_N_MAX`; progress message `Rastgele kontrol {i}/{n_max} (erken durabilir)`.
- Server:
  - `EffectReq`: `heads`, `prompts`, `n_max: int = Field(DEFAULT_N_MAX, ge=MIN_N_MAX, le=999)`, `seed: int = 0`; `n_random` is removed. On a CPU model `n_max > CPU_N_MAX_CAP` → 422 `invalid_input` ("CPU'da en çok 199 çekiliş").
  - `Session.measure_count: int` (0 on load/free); a finished, not cancelled effect job increments it; the result is `asdict(SequentialEffect) | {"session_measure_index": count}`; `/api/session` gains `"measure_count"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
def test_futility_k():
    assert futility_k(39, 0.05) == 1 and futility_k(19, 0.05) == 0
    assert futility_k(199, 0.05) == 9 and futility_k(39, 0.025) == 0
def test_sequential_stops_at_second_exceedance():
    out, seen = sequential_permutation(1.0, iter([0.1, 2.0, 0.2, 3.0] + [0.0] * 100), 39)
    assert (out.n_drawn, out.k_exceed, out.p_value, out.p_kind) == (4, 2, 0.5, "besag_clifford")
    assert out.stopped_early and not out.significant and len(seen) == 4
def test_sequential_full_run_no_exceedance():
    out, _ = sequential_permutation(1.0, iter([0.0] * 39), 39)
    assert out.p_value == pytest.approx(1 / 40) and out.significant and out.p_kind == "full"
def test_one_exceedance_is_boundary_significant():
    out, _ = sequential_permutation(1.0, iter([2.0] + [0.0] * 38), 39)
    assert out.p_value == pytest.approx(0.05) and out.significant
def test_ties_count_as_exceedance():
    out, _ = sequential_permutation(1.0, iter([1.0, 1.0] + [0.0] * 37), 39); assert out.stopped_early
def test_draws_not_consumed_after_stop():
    pulled = []; gen = (pulled.append(v) or v for v in [5.0, 5.0, 5.0])
    sequential_permutation(1.0, gen, 39); assert len(pulled) == 2
def test_sequential_real_effect_prefix_matches_recipe_effect(lm):
    s = sequential_real_effect(lm, [HeadAblation(0, 1)], PROMPTS[:4], n_max=19, seed=3)
    r = recipe_effect(lm, [HeadAblation(0, 1)], PROMPTS[:4], n_random=19, seed=3)
    assert s.kl == pytest.approx(r.kl) and s.random_kls == pytest.approx(r.random_kls[: s.n_drawn])
def test_sequential_real_effect_same_seed_same_result(lm): two calls → identical asdict
def test_sequential_rejects_small_n_max(lm): n_max=18 → ValueError
def test_sequential_cancel_raises(lm): pre-set cancel event → Cancelled
```
`tests/test_server.py`:
```python
def test_effect_returns_sequential_fields(ready):
    res = wait_done(...)["result"]
    assert {"kl", "n_drawn", "n_max", "k_exceed", "p_value", "p_kind", "p_min", "tau", "significant",
            "stopped_early", "seed", "session_measure_index"} <= res.keys()
    assert res["n_max"] == 39 and res["p_min"] == pytest.approx(0.025) and res["seed"] == 0
def test_effect_n_max_bounds(ready): 18 and 1000 → 422; 19 → job accepted
def test_effect_cpu_cap(ready): 200 → 422 with error_code "invalid_input"
def test_session_measure_count_increments(ready): measure_count goes up by 1 per finished effect job
```

- [ ] **Step 2: Run to verify failure** — `<PYTEST> tests/test_metrics.py tests/test_server.py -k "futility or sequential or exceedance or effect or measure_count"` → FAIL (`ImportError: futility_k`).

- [ ] **Step 3: Implement** per Interfaces; the effect job calls `sequential_real_effect` with `sub_progress(report, 0.1, 1.0)`.

- [ ] **Step 4: Verify** — `<PYTEST>` → all pass.

- [ ] **Step 5: Commit**

```bash
git add neshter/metrics.py neshter/server/app.py tests/test_metrics.py tests/test_server.py
git commit -m "feat(metrics): sequential Besag-Clifford permutation test for real effect (N_max 39, futility stop, pinned seed)"
```

---

### Task 5: Bootstrap, API client, SSE, job runner, contract fixtures

UX §1.4, §3.6; AR F2, F12; AS S1.

**Files:**
- Create: `web/src/app/{bootstrap.ts,sessionStorage.ts}`, `web/src/api/{client.ts,errors.ts,schemas.ts,sse.ts,jobs.ts,useJob.ts,heartbeat.ts,eta.ts}`, `web/src/api/__fixtures__/*.json`, tests `web/src/api/__tests__/{client,sse,jobs,schemas,eta}.test.ts`, `web/src/app/__tests__/bootstrap.test.ts`, `tests/tiny_models.py`, `tests/test_web_contract.py`
- Modify: `tests/conftest.py` (build tiny Qwen2 via `tiny_models.build_tiny_qwen2`)

**Interfaces:**
- Consumes: routes of Tasks 2–4 and the existing `app.py`.
- Produces:
  - `bootstrapToken(loc: Pick<Location,'hash'|'pathname'>, ss: Storage, hist: Pick<History,'replaceState'>, dev: boolean): string | null` — `#token=X` → `ss.setItem('neshter.token', X)`, `hist.replaceState(null, '', loc.pathname + '#/')`, return X; otherwise the stored token; in dev with none, `''` (the proxy injects it); otherwise `null` (app shows "open the address printed in the terminal").
  - `sessionStorage.ts`: `getToken/setToken`, `getActiveJob/setActiveJob/clearActiveJob` (keys `neshter.token`, `neshter.activeJob`).
  - `ErrorCode` = `'forbidden'|'not_found'|'busy'|'invalid_input'|'no_model'|'oom'|'unsupported_arch'|'unsafe_file'|'llama_cpp_missing'|'ollama_missing'|'disk_full'|'source_modified'|'exists'|'cancelled'|'internal'|'too_large'|'unsupported'|'network'`. `class ApiError extends Error { status: number; code: ErrorCode; detail: unknown }`, `class BusyError extends ApiError`.
  - `createApi(opts: { token: string; fetchFn?: typeof fetch; onStale: () => void; onBusy?: () => void }): Api` with methods named after routes: `system, models, concepts, session, loadModel, sample, activity, mlp, compare, recipeCheck, effect, hunt, suite, estimate, exportModel, jobs, job, cancelJob, drafts, draft, putDraft, deleteDraft, prefs, putPrefs, shutdown`. Every request sends `X-Neshter-Token`; responses are parsed with the zod schema of the same name in `schemas.ts` (loose objects: unknown keys pass). 403 → `onStale()` (once per api instance) then throw `ApiError('forbidden')`; 409 `busy` → `onBusy()` then throw `BusyError`; network `TypeError` → `ApiError(status 0, 'network')`.
  - `streamSse(url: string, opts: { token: string; signal: AbortSignal; onEvent: (event: string, data: unknown) => void; fetchFn?: typeof fetch }): Promise<'ended'|'aborted'>` — fetch streaming parser for `event:`/`data:` blocks separated by a blank line, CRLF tolerant, `:` comment lines ignored, chunk boundaries anywhere.
  - `JobState<R> = { id: string; kind: string; state: 'queued'|'running'|'done'|'error'|'cancelled'; progress: number; message: string; result: R | null; error: string | null; error_code: ErrorCode | null; started_at: string | null }`.
  - `attachJob<R>(api: Api, id: string, onUpdate: (s: JobState<R>) => void, signal: AbortSignal, opts?: { maxReconnects?: number; retryMs?: number }): Promise<JobState<R>>` — `GET /api/jobs/{id}` first, then the SSE stream; when the stream ends without a terminal event, re-`GET` and reconnect (default 5 tries, 1000 ms); resolves on `done|error|cancelled`. While running, the id is in `neshter.activeJob`; cleared at the end.
  - `useJob<R>(): { job: JobState<R> | null; start: (begin: () => Promise<{ job_id: string }>) => Promise<JobState<R>>; cancel: () => Promise<void>; reset: () => void }` — also writes the active job into `jobsStore` (Task 6).
  - `etaSeconds(progress: number, elapsedMs: number): number | null` — `null` while `progress < 0.05`, else `elapsedMs * (1 - progress) / progress / 1000`.
  - `startHeartbeat(api: Api, onDown: () => void, intervalMs = 10_000): () => void` — `GET /api/session` each interval; a 403 already flows to `onStale`; a network error calls `onDown`.
  - `tests/tiny_models.py`: `build_tiny_tokenizer() -> PreTrainedTokenizerFast`, `build_tiny_qwen2(path: Path) -> Path` (exactly today's conftest builder, seed 0).
  - `tests/test_web_contract.py`: hits the routes with a loaded tiny model through `TestClient` and compares the **key shape** (recursive dict key sets; lists compared by their first element) with `web/src/api/__fixtures__/<name>.json`; with `NESHTER_UPDATE_FIXTURES=1` it rewrites the files instead (lists truncated to 2 items, strings to 200 chars). Fixture names: `system, session_empty, session_loaded, models, concepts, activity, mlp, compare, recipe_check, job_effect, job_hunt, job_suite, estimate, job_export, jobs, drafts, draft, prefs, error_busy`.

- [ ] **Step 1: Write the failing tests**

```ts
test('bootstrap moves token to sessionStorage and strips hash', () => {
  const ss = memStorage(); const hist = { replaceState: vi.fn() };
  expect(bootstrapToken({ hash: '#token=abc', pathname: '/' }, ss, hist, false)).toBe('abc');
  expect(ss.getItem('neshter.token')).toBe('abc');
  expect(hist.replaceState).toHaveBeenCalledWith(null, '', '/#/');
  expect(window.localStorage.length).toBe(0);
});
test('bootstrap reuses the sessionStorage token on reload', ...);   // hash '#/explore' → 'abc'
test('bootstrap returns null without token outside dev', ...);
test('client sends X-Neshter-Token and maps error_code', async () => { /* 422 {error_code:'invalid_input'} → ApiError code invalid_input */ });
test('403 marks the session stale once', async () => { /* two 403s → onStale called 1 time, both throw forbidden */ });
test('409 busy throws BusyError and calls onBusy', ...);
test('network failure maps to network code', ...);
test('sse parser handles split chunks, CRLF and keepalive comments', async () => {
  const chunks = ['event: progress\r\nda', 'ta: {"progress":0.5}\r\n\r\n: keepalive\n\n', 'event: done\ndata: {"state":"done"}\n\n'];
  expect(events).toEqual([['progress', { progress: 0.5 }], ['done', { state: 'done' }]]);
});
test('attachJob re-attaches after the stream drops and resolves on done', ...);   // first stream ends early; second GET says done
test('attachJob stores and clears neshter.activeJob', ...);
test('etaSeconds is null below 5% progress', () => { expect(etaSeconds(0.04, 9000)).toBeNull(); expect(etaSeconds(0.5, 10_000)).toBe(10); });
test('every fixture parses with its schema', () => { for (const [name, schema] of Object.entries(FIXTURE_SCHEMAS)) expect(() => schema.parse(load(name))).not.toThrow(); });
```
Python: `test_web_fixtures_match_live_shapes(ready_contract)` — one test per fixture name (parametrized), failing with the differing key path.

- [ ] **Step 2: Run to verify failure** — `<WEBENV> npm --prefix web run test -- src/api src/app` → FAIL (modules missing). `<PYTEST> tests/test_web_contract.py` → FAIL (fixtures missing).

- [ ] **Step 3: Generate fixtures** — `NESHTER_UPDATE_FIXTURES=1 <PYTEST> tests/test_web_contract.py` → files written; review them for absolute paths (`source_path`, `path` values) and replace any user path with `C:\\Users\\x\\model` before committing.

- [ ] **Step 4: Implement** the modules per Interfaces; `conftest.py` imports `build_tiny_qwen2` (`from tiny_models import build_tiny_qwen2`).

- [ ] **Step 5: Verify** — `<WEBENV> npm --prefix web run test` and `run lint` → pass; `<PYTEST>` → all pass (fixtures now compared).

- [ ] **Step 6: Commit**

```bash
git add web/src/app web/src/api tests/tiny_models.py tests/conftest.py tests/test_web_contract.py
git commit -m "feat(web): token bootstrap, typed API client, fetch-SSE with re-attach, job runner, contract fixtures"
```

---

### Task 6: Domain core and vanilla stores (pure TypeScript)

UX §3.1–§3.3, §1.2, §1.6; 3D spec §7 (Op vs ExploreOp).

**Files:**
- Create: `web/src/core/{zod.ts,ids.ts,ops.ts,fingerprint.ts,basket.ts,measurement.ts,recipe.ts,draft.ts,saveflow.ts,normalize.ts}`, `web/src/core/stores/{session.ts,selection.ts,basket.ts,jobs.ts,ui.ts,compare.ts,measure.ts}`, tests in `web/src/core/__tests__/`, fixture `web/src/core/__tests__/fixtures/fakeExploreOp.ts`

**Interfaces:**
- Consumes: Task 5 `JobState`, effect result schema.
- Produces:
  - `zod.ts`: calls `z.config({ jitless: true })`; every core module imports `z` from here.
  - `ids.ts`: branded `NodeId`, `EdgeId`; `headId(layer, head)`, `neuronId(layer, index)`, `tokenPosId(pos)`, `tokenIdId(id)`, `OUT_ID = 'out'`, `edgeId(id: string)`; `parseNodeId(s: string): ParsedNode | null` where `ParsedNode = {kind:'head', layer, head} | {kind:'neuron', layer, index} | {kind:'tpos', pos} | {kind:'tid', id} | {kind:'out'}`; non-canonical strings (`h:01:2`, `h:-1:2`, `h:1`) → `null`.
  - `ops.ts`: `HeadOpSchema`, `DirectionOpSchema` (strict; `vector_b64` decodes to `4 * d_model` bytes); `KnownOp`; `OpaqueOp = { type: string; opaque: true; raw: Record<string, unknown> }`; `AnyOp = KnownOp | OpaqueOp`; `Semantics = 'zero'|'direction'|'path'|'read'|'steer'|'unknown'`; `registerOpKind(type: string, spec: { schema: ZodType; savable: boolean; semantics: Semantics })` (built-ins `head` savable/zero, `direction` savable/direction); `parseOp(raw: unknown): AnyOp` (registered + valid → known; anything else → opaque); `isKnown(op)`, `isSavable(op)` (known and spec savable), `isSendable(op)` (known); `opKey(op): string` (canonical JSON with sorted keys).
  - `fingerprint.ts`: `opsFingerprint(ops: readonly KnownOp[]): string` (sorted `opKey` joined by `|`).
  - `basket.ts`: `Origin = 'heatmap'|'concept'|'graph'|'import'`; `BasketItem = { id: string; op: AnyOp; semantics: Semantics; measurement?: Measurement; origin: Origin; concept: string | null; addedAt: string; pendingDiscard?: boolean }`; selectors `unsavedCount(items, savedOps)`, `isDirty(items, savedOps)` (= `unsavedCount > 0`; an item is unsaved when its `opKey` is not in `savedOps`), `countSavable(items)`, `countExploreOnly(items)` (not savable, not pending), `savableOps(items): KnownOp[]`, `sendableOps(items): KnownOp[]` (known, not `pendingDiscard`).
  - `measurement.ts`: `ControlStats = { n_drawn, n_max, k_exceed, p_value, p_kind: 'full'|'besag_clifford', p_min, tau, significant, stopped_early, band: [number, number] | null, percentile: number | null }`; `Measurement = { kind: 'set_effect'|'edge'|'steer'; value: number; se: number | null; kl: number; kl_se: number | null; control: ControlStats | null; uncertain: boolean; n_prompts: number; seed: number; session_index: number; ops_fp: string; measured_at: string }`; `fromEffectResult(res: EffectResult, opsFp: string, now: string): Measurement` (kind `set_effect`, `value = kl`, `se = kl_se`, `uncertain = !significant`, `session_index = session_measure_index`, `band = [random_p05, random_p95]` or `null`); `verdict(m): 'significant'|'not_significant'|'futility'|'no_control'`.
  - `recipe.ts`: `RecipeSchema` (strict mirror of `neshter/recipe.py` `Recipe`, `schema_version: 1`); `parseRecipeText(text: string): { ok: true; recipe: Recipe } | { ok: false; error: 'json'|'schema'|'too_large' }` (> 4 MB → too_large); `recipeToItems(recipe, now): BasketItem[]` (origin `import`, `concept` from the recipe); `itemsToRecipe(items, model: { name: string; sha256: string }, concept: string | null, now: string): Recipe` (savable ops only; `method: 'ui'`, `seed: 0`, `dataset_sha256: null`, `head_mode: 'zero'`).
  - `draft.ts`: `DRAFT_SCHEMA_VERSION = 1`; `encodeDraft(input: { items; shelf; savedOps; model: { name; sha256 }; prompts: string[] }): DraftPayload` (matches server `DraftIn`; `unsaved_count = unsavedCount(...)`; `pendingDiscard` dropped; opaque ops written back as their `raw`); `decodeDraft(json: unknown): { ok: true; items; shelf; savedOps; prompts; unsupported: number } | { ok: false; reason: 'schema_version'|'invalid' }`.
  - `saveflow.ts`: `SaveFlowState = {kind:'idle'} | {kind:'nothingToSave'} | {kind:'needsDiscard'; exploreIds: string[]} | {kind:'recomparing'; dropping: string[]; jobId: string | null} | {kind:'reviewReady'; dropped: number} | {kind:'exportRequested'} | {kind:'exporting'; jobId: string} | {kind:'done'; result: ExportResult}`; events `SAVE_CLICKED | DISCARD_CONFIRMED | RECOMPARE_STARTED{jobId} | RECOMPARE_DONE | RECOMPARE_FAILED | EXPORT_STARTED{jobId} | EXPORT_DONE{result} | EXPORT_FAILED | CANCEL | DISMISS`; `saveFlowReducer(state, event, ctx: { items: readonly BasketItem[] }): SaveFlowState`. Rules: `SAVE_CLICKED` (from `idle` or `reviewReady`): no savable item → `nothingToSave`; any not-savable, not-pending item → `needsDiscard` with their ids; else `exportRequested`. `DISCARD_CONFIRMED` → `recomparing{dropping: exploreIds, jobId: null}`. `RECOMPARE_DONE` → `reviewReady{dropped: dropping.length}`. `RECOMPARE_FAILED` or `CANCEL` from any state except `done` → `idle`. `EXPORT_DONE` → `done`. `DISMISS` from `nothingToSave`/`done`/`reviewReady` → `idle`. Any other pair returns the same state object.
  - `normalize.ts`: `sliceToken(powerRaw: number[][][], t: number): number[][]` (negative `t` from the end), `normalizePerLayer(m)`, `normalizeGlobal(m)` — the same min–max rules as `neshter/activity.py` (constant row/matrix → zeros).
  - Stores (`createXStore()` factories plus default singletons; zustand `createStore` with `subscribeWithSelector`):
    - `session`: `{ system: System | null; model: SessionLoaded | null; stale: 'none'|'restarted'|'down'; busy: boolean }`
    - `selection`: `{ selected: ReadonlySet<NodeId>; focus: NodeId | null }`; `toggle(id)`, `set(ids)`, `clear()`, `setFocus(id)`
    - `basket`: `{ items; shelf; savedOps: string[]; lastRemoved: BasketItem | null }`; `add(items): number` (skips items whose `opKey` is already in `items`), `remove(id)`, `undoRemove()`, `markPendingDiscard(ids)`, `clearPendingDiscard()`, `moveToShelf(ids)`, `restoreFromShelf(id)`, `markSaved()` (savedOps = keys of savable items), `replaceAll({ items, shelf, savedOps })`
    - `jobs`: `{ active: JobState<unknown> | null }`; `isLocked(s)` = active in `queued|running`
    - `ui`: `{ route: Route; view: '2d'|'list'|'3d'; lang: 'tr'|'en'; reducedMotion: 'system'|'on'|'off'; quality: 'high'|'balanced'|'low' }`; `Route = 'home'|'models'|'explore'|'concept'|'compare'|'save'`
    - `compare`: `{ lastPrompt: string | null; lastPair: ComparePair | null; lastSuite: { fp: string; result: SuiteResult } | null; refreshedDropped: number | null }`
    - `measure`: `{ byFp: Record<string, Measurement> }`; `put(m)`, `get(fp)`

- [ ] **Step 1: Write the failing tests** (`web/src/core/__tests__/`)

```ts
test('headId round-trips and rejects non-canonical ids', () => {
  expect(parseNodeId(headId(3, 5))).toEqual({ kind: 'head', layer: 3, head: 5 });
  for (const bad of ['h:01:2', 'h:-1:2', 'h:1', 'x:1:2', 'h:1:2:3']) expect(parseNodeId(bad)).toBeNull();
});
test('parseOp: head/direction known, fake explore known but not savable, unknown opaque', () => {
  registerFakeExploreOp();                     // type 'test_explore', savable false, semantics 'path'
  expect(isSavable(parseOp({ type: 'head', layer: 0, head: 1 }))).toBe(true);
  const fake = parseOp({ type: 'test_explore', src: 'h:0:1' }); expect(isSendable(fake) && !isSavable(fake)).toBe(true);
  const unk = parseOp({ type: 'edge_read_v9', x: 1 }); expect(unk).toMatchObject({ opaque: true }); expect(isSendable(unk)).toBe(false);
  expect(parseOp({ type: 'head', layer: 0, head: 1, extra: 1 })).toMatchObject({ opaque: true });   // strict
});
test('direction op validates vector length', ...);   // 4*d_model bytes else opaque
test('unsavedCount and isDirty follow savedOps', ...); // markSaved → 0; add new item → 1
test('decodeDraft keeps unknown ops opaque', () => { const r = decodeDraft(draftWith([{ type: 'edge_read_v9' }])); expect(r.ok && r.unsupported === 1 && r.items.length === 1).toBe(true); });
test('decodeDraft rejects other schema versions', ...);  // schema_version 2 → {ok:false, reason:'schema_version'}
test('encodeDraft round-trips through decodeDraft and drops pendingDiscard', ...);
test('itemsToRecipe keeps only savable ops and parses with RecipeSchema', ...);
test('parseRecipeText rejects unknown fields and >4 MB', ...);
test('fromEffectResult maps futility and significance', ...);   // fixtures job_effect + hand-made futility result
test.each(SAVEFLOW_TABLE)('saveflow %s', (_name, start, event, items, expected) => expect(saveFlowReducer(start, event, { items })).toEqual(expected));
// table rows: all savable → exportRequested; savable + fake explore → needsDiscard[fakeId]; only fake → nothingToSave;
// opaque counts as not savable; DISCARD_CONFIRMED → recomparing; RECOMPARE_DONE → reviewReady{1};
// CANCEL during recomparing → idle; RECOMPARE_FAILED → idle; reviewReady + SAVE_CLICKED (now all savable) → exportRequested;
// exporting + EXPORT_DONE → done; done + CANCEL → done (unchanged)
test('normalize mirrors python: constant row → zeros', () => expect(normalizePerLayer([[2, 2], [1, 3]])).toEqual([[0, 0], [0, 1]]));
test('basket store: add dedupes, remove + undoRemove, moveToShelf + restoreFromShelf', ...);
test('core has no react import', ...);   // reads src/core/**/*.ts, asserts no /from 'react'/ (backs the ESLint rule)
```

- [ ] **Step 2: Run to verify failure** — `<WEBENV> npm --prefix web run test -- src/core` → FAIL.

- [ ] **Step 3: Implement** per Interfaces.

- [ ] **Step 4: Verify** — `<WEBENV> npm --prefix web run test` and `run lint` → pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/core
git commit -m "feat(web): domain core (canonical ids, op registry with opaque ops, basket, measurement, draft codec, save-flow reducer) and vanilla stores"
```

---

### Task 7: i18n, app shell, header, prefs, accessibility primitives

UX §3.5, §2.8; AR F3, F6; spec §5 (i18n).

**Files:**
- Create: `web/src/i18n/{index.ts,tr.json,en.json}`, `web/scripts/check-i18n.mjs`, `web/src/app/{App.tsx,Shell.tsx,Header.tsx,Stepper.tsx,router.ts,prefs.ts,ErrorBoundary.tsx,StaleScreen.tsx,BusyBanner.tsx,DeviceBanner.tsx}`, `web/src/ui/{Button.tsx,Dialog.tsx,LiveRegion.tsx,ProgressCard.tsx,ErrorCard.tsx,TechDetail.tsx,Segmented.tsx}`, tests next to them in `__tests__/`
- Modify: `web/src/App.tsx` (replaced by `app/App.tsx`), `web/package.json` (`lint` = `eslint . && node scripts/check-i18n.mjs`)

**Interfaces:**
- Consumes: Task 5 api, `startHeartbeat`; Task 6 `uiStore`, `sessionStore`, `basketStore`, `unsavedCount`.
- Produces:
  - `t(key: MessageKey, params?: Record<string, string | number>): string`, `MessageKey = keyof typeof tr` (flat dotted keys, `{name}` interpolation, numbers formatted with `Intl.NumberFormat(lang)` so tr shows `0,025`); `plural(key, n)` via `Intl.PluralRules`; `setLang(lang)` updates `uiStore` and `document.documentElement.lang`. Key groups: `app.*`, `header.*`, `home.*`, `models.*`, `map.*`, `measure.*`, `basket.*`, `cut.*`, `concept.*`, `compare.*`, `save.*`, `leave.*`, `restore.*`, `saveflow.*`, `errors.<ErrorCode>`, `jobs.*`, `vocab.*` (layer `K{n}`, head, neuron, token, `tahmini`, `ölçülen`, the four badges, op semantics).
  - `check-i18n.mjs`: exits 1 when tr/en key sets differ, a value's `{placeholders}` differ between languages, or a key is unused in `src/**` (keys under the prefixes `errors.`, `vocab.badge.`, `concept.category.` count as used dynamically).
  - `router.ts`: `parseHash(hash: string): Route` (unknown → `home`), `navigate(route: Route)`, `startRouter()` (listens to `hashchange`; never touches the token, which bootstrap already removed).
  - `prefs.ts`: `loadPrefs(api)` (GET; on failure defaults), `savePrefs(api, patch)` (PUT merged), `applyMotion(pref, matchMedia)` sets `html[data-motion]` to `reduce` or `full`.
  - `Header` props `{ onClose: () => void }`: chips GPU/VRAM/RAM (`gpu_name === null` → `Yalnızca CPU`), model chip `name · dtype · device`, unsaved chip (hidden at 0), language select, motion toggle ("Animasyonları azalt"), `Kapat` button → `onClose`.
  - `Stepper`: Model → Davranışı bul / Ağ haritası → Karşılaştır → Kaydet (hidden for `home`).
  - `ErrorCard({ code, detail })`: text from `errors.<code>`; `detail` only inside `<TechDetail>` (`<details>` "Teknik ayrıntı").
  - `Dialog({ open, title, children, actions, onCancel })`: native `<dialog>` with `showModal()`; Esc → `onCancel`; initial focus on the first action.
  - `LiveRegion`: `announce(text)` into an `aria-live="polite"` node.
  - `ProgressCard({ job, onCancel })`: progress bar, message, elapsed, ETA from `etaSeconds` only when not null, `İptal`.
  - `StaleScreen({ kind: 'restarted'|'down', hasDraft: boolean })`; `BusyBanner` shown while `sessionStore.busy` or `jobs.isLocked`.

- [ ] **Step 1: Write the failing tests**

```ts
test('t interpolates and formats numbers per language', () => {
  setLang('tr'); expect(t('measure.sig', { p: 0.025, n: 39 })).toContain('p = 0,025');
  setLang('en'); expect(t('cut.button')).toBe('Cut (temporary)');
});
test('setLang sets html lang', () => { setLang('tr'); expect(document.documentElement.lang).toBe('tr'); });
test('default language follows navigator when prefs.lang is null', ...);
test('check-i18n fails on a missing key and on placeholder mismatch', ...);   // runs the script against tmp catalogs
test('exact pinned copy', () => {
  setLang('tr');
  expect(t('cut.button')).toBe('Kes (geçici)'); expect(t('cut.sub')).toBe('Dosya değişmez');
  expect(t('leave.title')).toBe('Ayrılıyorsunuz, kesimlerinizi daha kaydetmediniz');
  expect(t('header.unsaved', { n: 2 })).toBe('Kaydedilmemiş 2 kesim');
});
test('header shows Yalnızca CPU when gpu_name is null and hides unsaved chip at 0', ...);
test('ErrorCard never shows detail outside the disclosure', ...);   // detail text only inside <details>
test('Dialog: Esc calls onCancel', ...);
test('ProgressCard hides ETA below 5% progress', ...);
test('motion pref on sets data-motion=reduce', ...);
test('forbidden copy is absent from catalogs', () => {
  const all = JSON.stringify(tr) + JSON.stringify(en);
  for (const s of ['ortalama değeriyle', 'Kaynağa güveniyorum', 'sırayla yükler', 'Modeli küçült', 'Kendi test setini yükle']) expect(all).not.toContain(s);
});
```

- [ ] **Step 2: Run to verify failure** — `<WEBENV> npm --prefix web run test -- src/i18n src/app src/ui` → FAIL.

- [ ] **Step 3: Implement** per Interfaces. `App` order: bootstrap token → create api (`onStale` → `sessionStore.stale = 'restarted'`) → `loadPrefs` → `GET /api/system` + `/api/session` → heartbeat (`onDown` → `stale = 'down'`) → router.

- [ ] **Step 4: Verify** — `<WEBENV> npm --prefix web run test`, `run lint` (includes check-i18n), `run build` → pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/i18n web/src/app web/src/ui web/src/App.tsx web/scripts/check-i18n.mjs web/package.json
git commit -m "feat(web): typed tr/en i18n with parity check, app shell, header chips, prefs, a11y primitives"
```

---

### Task 8: Draft autosave, restore, leave guard, single-tab lock, save-flow wiring

UX §1.2, §1.3, §1.6, §1.7; 3D spec §7, §8 (leave warning); synthesis D, E.

**Files:**
- Create: `web/src/features/draft/{autosave.ts,RestorePrompt.tsx,UnsavedWork.tsx}`, `web/src/features/leave/{useLeaveGuard.ts,LeaveDialog.tsx}`, `web/src/features/saveflow/{saveFlowStore.ts,useSaveFlow.ts,DiscardDialog.tsx,NothingToSaveDialog.tsx}`, `web/src/app/{singleTab.ts,MultiTabScreen.tsx}`, tests in `__tests__/`
- Modify: `web/src/app/App.tsx` (wire autosave, tab lock, leave guard into `Header.onClose`)

**Interfaces:**
- Consumes: Task 5 api (`draft`, `putDraft`, `deleteDraft`, `drafts`, `recipeCheck`, `suite`, `compare`, `shutdown`), `useJob`; Task 6 `basketStore`, `encodeDraft`, `decodeDraft`, `itemsToRecipe`, `saveFlowReducer`, `sendableOps`, `compareStore`.
- Produces:
  - `startAutosave(deps: { api; basket: BasketStore; session: SessionStore; prompts: () => string[]; debounceMs?: number }): () => void` — on every basket change (debounced 500 ms) and on `visibilitychange` → `hidden` (flush now): `PUT /api/draft/{sha}` with `encodeDraft`; when items and shelf are both empty and a draft was written, `DELETE` it; no-op while no model is loaded.
  - `RestorePrompt`: after a load job finishes, `GET /api/draft/{sha}`; shown only if `decodeDraft` is ok, it has ≥ 1 item and the basket is empty. Buttons `Geri yükle` (runs `recipeCheck(itemsToRecipe(...))`; `ok` → `basket.replaceAll`, else an ErrorCard listing `errors` and no restore), `Sil` (DELETE), `Sonra` (close). Unsupported ops count is shown as `{n} kesim bu sürümde desteklenmiyor; sepette gri görünür`.
  - `UnsavedWork` (Home section): `GET /api/drafts` → rows `Kaydedilmemiş çalışmalar: {model_name} · {n} kesim`, plus `Taslakları sil` (DELETE all after confirm). It states that prompts are stored in drafts.
  - `useLeaveGuard(): { request(kind: 'switch_model'|'close'|'home', proceed: () => void): void; dialog: ReactNode }` — not dirty → `proceed()` immediately; dirty → `LeaveDialog` with `Kaydet` (navigate `save`, do not proceed), `Kaydetmeden çık` (proceed; the server draft is kept so the work can still be restored), `Vazgeç`. While dirty, a `beforeunload` listener calls `preventDefault()` and sets `returnValue = ''`; it is removed when clean.
  - `Kapat` flow: `request('close', () => api.shutdown())` → after 202 a full-page "Neşter kapandı; bu sekmeyi kapatabilirsiniz".
  - `acquireTabLock(locks?: LockManager): Promise<'owner'|'other'>` — `navigator.locks.request('neshter-ui', { ifAvailable: true }, cb)` where `cb` holds a never-resolving promise; `null` lock → `'other'`; missing API → `'owner'`. `'other'` renders `MultiTabScreen` and starts nothing else.
  - `saveFlowStore` (vanilla) + `useSaveFlow(): { state: SaveFlowState; saveClicked(): void; confirmDiscard(): void; cancel(): void; dismiss(): void }`. Effects: `needsDiscard` → `DiscardDialog` (lists each explore item's label, opaque ones as `desteklenmeyen kesim`); `DISCARD_CONFIRMED` → `basket.markPendingDiscard(ids)`, start a quick suite job with `sendableOps` (pending excluded) and the basket concept's `suite_target`, plus a compare re-run when `compare.lastPrompt` exists; job done → `basket.moveToShelf(ids)`, `compare.refreshedDropped = n`, `RECOMPARE_DONE`, navigate `compare`; job cancelled/failed → `basket.clearPendingDiscard()`, `RECOMPARE_FAILED`. `nothingToSave` → `NothingToSaveDialog` (`Kaydedilecek kesim kalmıyor.`). `exportRequested` is consumed by the Save screen (Task 14).

- [ ] **Step 1: Write the failing tests** (fake timers, mock api)

```ts
test('autosave debounces to one PUT after 500 ms', ...);   // 3 adds within 200 ms → 1 PUT, unsaved_count 3
test('autosave flushes on visibilitychange hidden', ...);
test('autosave deletes the draft when basket and shelf become empty', ...);
test('restore prompt shows count and restores after recipe check', ...);   // text 'Önceki oturumdan kaydedilmemiş 2 kesim var: geri yükle?'
test('restore keeps an unknown op as an unsupported row', ...);
test('restore is not offered when the basket is not empty', ...);
test('leave guard proceeds at once when clean', ...);
test('leave guard shows the three exact buttons when dirty and Vazgeç keeps state', ...);
test('beforeunload listener exists only while dirty', ...);   // spy add/removeEventListener('beforeunload')
test('second tab shows the multi-tab screen', async () => { expect(await acquireTabLock(fakeLocks(null))).toBe('other'); });
test('saveflow with a fake explore op: discard → suite with savable ops only → shelf + refreshed banner, no second modal', ...);
test('saveflow: cancelling the re-run leaves the basket unchanged', ...);
test('saveflow: only explore ops → nothing-to-save dialog and no export call', ...);
```

- [ ] **Step 2: Run to verify failure** — `<WEBENV> npm --prefix web run test -- src/features/draft src/features/leave src/features/saveflow src/app` → FAIL.

- [ ] **Step 3: Implement** per Interfaces.

- [ ] **Step 4: Verify** — `run test`, `run lint` → pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/features/draft web/src/features/leave web/src/features/saveflow web/src/app
git commit -m "feat(web): server-side draft autosave and restore, leave guard with beforeunload, single-tab lock, transactional save flow"
```

---

### Task 9: Home, model picker, loader, recipe import

Spec §2 steps 1–2; AR §2.1 Main and Models, F4, F5; AS P9.

**Files:**
- Create: `web/src/screens/{Home.tsx,Models.tsx}`, `web/src/features/models/{ModelRow.tsx,SampleDownload.tsx,LoadScreen.tsx,PathField.tsx,format.ts}`, `web/src/features/recipe/ImportRecipe.tsx`, tests in `__tests__/`

**Interfaces:**
- Consumes: api `models`, `sample`, `loadModel`, `session`, `recipeCheck`, `drafts`; `useJob`; `useLeaveGuard`; `UnsavedWork`; `parseRecipeText`, `recipeToItems`.
- Produces:
  - Home: title `Ne yapmak istiyorsun?`, cards `Bir davranışı azalt` (→ models, then concept) and `İçine bak` (→ models, then explore), link `Uzman modu` (→ models, then explore; sets `uiStore` intent `expert`), `UnsavedWork`, `Tarif içe aktar…` (file input `.json`). No "Modeli küçült".
  - `formatArch(arch): string` → `24 katman · 14Q/2KV head · 4 864 MLP` (number via `Intl.NumberFormat(lang)`); `formatBytes(n)`; `badgeLabel(badge)`: `ok` → `rahat`, `slow` → `yavaş · RAM desteği`, `no` → `sığmaz`.
  - `ModelRow`: name (text), source, fmt, size, structure, memory, badge, license (`null` → `lisans bilinmiyor`); unsupported rows disabled with `models.reason.<unknown_arch|unsupported_family>`; `Aç` button.
  - Empty state: `Örnek modeli indir · Qwen2.5-0.5B` → sample job (bytes message from the server) → on done load `result.path`.
  - `PathField`: absolute path text input + `Aç`.
  - `LoadScreen`: `ProgressCard` for the load job with `İptal`; on done fills `sessionStore.model` from `/api/session`, triggers `RestorePrompt`, then routes to the intent's screen.
  - Loading while dirty goes through `useLeaveGuard().request('switch_model', ...)`.
  - Recipe import: `parseRecipeText` → pending import kept in memory; after a model is loaded, `recipeCheck`; `ok && !sha_mismatch` → items added; `sha_mismatch` → confirm `Bu tarif başka bir model sürümü için hazırlanmış olabilir. Yine de ekle?`; errors → ErrorCard.

- [ ] **Step 1: Write the failing tests**

```ts
test('home offers two intents and expert mode, never "Modeli küçült"', ...);
test('formatArch uses the language number format', () => {
  setLang('tr');
  const mlp = new Intl.NumberFormat('tr').format(4864);
  expect(formatArch({ family: 'qwen2', n_layers: 24, n_heads: 14, n_kv_heads: 2, d_mlp: 4864 })).toBe(`24 katman · 14Q/2KV head · ${mlp} MLP`);
});
test('unsupported row is disabled with its reason', ...);
test('badge labels', ...);
test('empty list shows sample download and starts the job', ...);
test('model name with markup renders as text', () => { renderRow({ name: '<img src=x onerror=alert(1)>' }); expect(document.querySelector('img')).toBeNull(); });
test('opening a model while dirty asks the leave guard first', ...);
test('load done fills the session model and routes to the intent screen', ...);
test('recipe import with sha mismatch asks before adding', ...);
```

- [ ] **Step 2: Run to verify failure** — `<WEBENV> npm --prefix web run test -- src/screens src/features/models src/features/recipe` → FAIL.

- [ ] **Step 3: Implement** per Interfaces.

- [ ] **Step 4: Verify** — `run test`, `run lint`, `run build` → pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/screens/Home.tsx web/src/screens/Models.tsx web/src/features/models web/src/features/recipe
git commit -m "feat(web): home intents, model picker with fit badges and reasons, sample download, load job, recipe import"
```

---

### Task 10: Explore — Canvas 2D heatmap, CellLayout, Liste table, token strip, MLP list

Spec §2 step 3a; UX §2.7 (Plan-2 hook), §2.8, §3.7; AR §2.1 Explore, F7.

**Files:**
- Create: `web/src/screens/Explore.tsx`, `web/src/features/map/{cellLayout.ts,drawHeatmap.ts,Heatmap.tsx,HeatmapTable.tsx,TokenStrip.tsx,MlpList.tsx,ViewSwitch.tsx,ScaleToggle.tsx,colorScale.ts,useHeatmapKeys.ts}`, tests in `__tests__/`

**Interfaces:**
- Consumes: api `activity` (`token_index` omitted: one call returns `tokens` + `power_raw [L][H][T]`), `mlp`; `normalize.ts`; `selectionStore`, `basketStore`, `jobsStore.isLocked`, `uiStore.view`.
- Produces:
  - `interface CellLayout { n_layers: number; n_heads: number; cellRect(id: NodeId): { x: number; y: number; w: number; h: number } | null; cellColor(id: NodeId): [number, number, number] }` (viewport px, Plan 3's fly-in start positions).
  - `makeCellLayout(dims: { n_layers; n_heads }, box: DOMRect, values: number[][], scale: (v: number) => [number, number, number]): CellLayout` — rows = layers (K0 at the top), columns = heads; ids outside the grid → `null`.
  - `drawHeatmap(ctx: CanvasRenderingContext2D, layout: CellLayout, values: number[][], marks: { selected: ReadonlySet<NodeId>; basket: ReadonlySet<NodeId>; focus: NodeId | null }): void` — one `fillRect` per cell; selected cells outlined with `--focus-ring`, basket cells with `--accent-basket`; colors read once from CSS tokens.
  - `Heatmap` (canvas, `tabIndex=0`, `role="group"`, `aria-roledescription` from i18n, `aria-describedby` → legend): keys via `useHeatmapKeys(dims)` — arrows move focus clamped to the grid, `Home`/`End` row ends, `Space`/`Enter` toggles `headId(L,H)` in `selectionStore`; every move announces `K{L} · H{H} · güç {v} · seçili|seçili değil` through `LiveRegion`. Exposes `getCellLayout(): CellLayout` via ref.
  - `HeatmapTable` (Liste view): `<table>` with one row per head (`K`, `H`, power, selected checkbox, basket mark), sortable by power; selection and every later action (measure, cut) work from here too.
  - `TokenStrip`: one chip per token rendered as text with `white-space: pre`; default the last token; clicking re-slices `power_raw` client-side (no request).
  - `ScaleToggle`: `Katman içi` (default) / `Tüm model`, client-side; legend `Head çıktı gücü, katman içinde 0–1` and the `tahmini` badge.
  - `MlpList`: `Birim: Head | MLP · en güçlü 50` → `POST /api/mlp` for the current token; rows `K{layer} · nöron {neuron} · {value}`.
  - `ViewSwitch`: `2B ısı haritası | Liste`; view registry `[{ id: '2d', enabled: true }, { id: 'list', enabled: true }, { id: '3d', enabled: false }]`; disabled entries are not rendered.
  - Run button and token chips are disabled while `jobs.isLocked`; a 409 sets the busy banner.

- [ ] **Step 1: Write the failing tests**

```ts
test('cellLayout returns rects for 3x5 and null outside', () => {
  const L = makeCellLayout({ n_layers: 3, n_heads: 5 }, rect(0, 0, 500, 300), zeros(3, 5), gray);
  expect(L.cellRect(headId(2, 4))).toEqual({ x: 400, y: 200, w: 100, h: 100 });
  expect(L.cellRect(headId(3, 0))).toBeNull(); expect(L.cellRect('n:0:1' as NodeId)).toBeNull();
});
test('drawHeatmap fills one rect per cell and outlines selected and basket cells', ...);   // fake ctx records calls
test('keyboard navigation clamps at edges for 1x1 and 3x5', ...);
test('Space toggles h:L:H in the selection store and announces it', ...);
test('Liste view selection is the same store as the heatmap', ...);
test('token strip renders markup as text', () => { renderStrip(['<img src=x onerror=alert(1)>']); expect(document.querySelector('img')).toBeNull(); expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeTruthy(); });
test('token click re-slices without a request', ...);   // api.activity called once
test('scale toggle switches normalisation without a request', ...);
test('run button disabled while a job holds the model', ...);
test('3d view is registered but not rendered', ...);
test('works for a 32x32 model', ...);   // 1024 cells drawn, table has 1024 rows
```

- [ ] **Step 2: Run to verify failure** — `<WEBENV> npm --prefix web run test -- src/features/map src/screens/Explore` → FAIL.

- [ ] **Step 3: Implement** per Interfaces (jsdom has no canvas: the component tests mock `getContext`; `drawHeatmap` is tested with a recording fake).

- [ ] **Step 4: Verify** — `run test`, `run lint`, `run build` → pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/screens/Explore.tsx web/src/features/map
git commit -m "feat(web): canvas heatmap behind CellLayout, keyboard grid with live readout, Liste table, token strip, MLP top-50"
```

---

### Task 11: Selection panel, sequential measurement, "Kes (geçici)", basket and shelf

Spec §2 step 3a ("Gerçek etkiyi ölç", silence only after measuring), §3 (zero-ablation, KL + random control); UX §1.6, §3.3; synthesis "Controls".

**Files:**
- Create: `web/src/features/measure/{SelectionPanel.tsx,MeasureButton.tsx,ControlStatsCard.tsx,nMax.ts}`, `web/src/features/basket/{BasketPanel.tsx,BasketRow.tsx,ShelfView.tsx,CutButton.tsx,labels.ts}`, tests in `__tests__/`
- Modify: `web/src/screens/Explore.tsx` (panels), `web/src/app/Header.tsx` (unsaved chip reads the basket)

**Interfaces:**
- Consumes: api `effect`; `useJob`; `fromEffectResult`, `verdict`, `opsFingerprint`, `measureStore`, `basketStore`, `selectionStore`, `sessionStore` (`device`, `measure_count`); `useSaveFlow().saveClicked`.
- Produces:
  - `nMaxBounds(device: string): { min: 19; max: 199 | 999; default: 39 }` (`cpu` → 199).
  - `SelectionPanel`: chips `K{L}·H{H}` for selected heads (removable), focus head power, zero-ablation sentence, `MeasureButton`, `ControlStatsCard` for the current selection's fingerprint, `CutButton`.
  - `MeasureButton`: `Gerçek etkiyi ölç`; an "Gelişmiş" disclosure with an N_max number input (bounds above) showing `en küçük p = {p_min}`; body `{ heads: [[L,H], ...], n_max, seed: 0 }`; progress text from the server message plus elapsed; the ETA is labelled `en çok` (early stop can make it shorter).
  - `ControlStatsCard({ m: Measurement })`: KL ± SE, `n_prompts`, band `rastgele (yaklaşık) {p05}–{p95}` when present, verdict copy per `verdict(m)` (sig / full not-sig / futility), session line when `m.session_index >= 2`, `ölçülen` badge; seed and `p_kind` inside `TechDetail`.
  - `CutButton({ selection })`: label `Kes (geçici)` + sub-line `Dosya değişmez`; enabled only when `measureStore.get(opsFingerprint(selection ops))` exists; disabled reason `Önce gerçek etkiyi ölç`; adds one `HeadOp` item per selected head (`semantics 'zero'`, `origin 'heatmap'`, shared measurement, `concept null`).
  - `BasketPanel`: rows with badge (`kaydedilebilir` teal, `yalnızca keşif` `--badge-explore`, `desteklenmeyen` gray), pending rows struck through with `atılacak`, remove + `Geri al` (undoRemove), line `Kaydedilecek: {s} · Yalnızca keşif: {e}`, buttons `Karşılaştır` (→ compare) and `Modeli kaydet ({n} kesim)` (→ `saveClicked()`, disabled at n = 0); `ShelfView` titled `Keşif rafı` with `Sepete geri al`.
  - `labels.ts`: `itemLabel(item): string` (`K3 · head 5`, `Yön · K12 sonrası`, opaque `desteklenmeyen kesim ({type})`).

- [ ] **Step 1: Write the failing tests**

```ts
test('cut disabled before measuring, enabled after, disabled again on selection change', ...);
test('measure sends n_max 39 by default and heads from the selection ids', ...);
test('cpu caps n_max input at 199 and shows p_min', ...);
test('ControlStatsCard significant copy', ...);
test('ControlStatsCard futility copy uses tr number format', () => { setLang('tr'); renderCard(futility({ n_drawn: 3, k_exceed: 2, tau: 0.05 })); expect(screen.getByText('p > 0,05, 3 çekişte durdu (2 aşım): rastgele seçimden ayırt edilemedi.')).toBeTruthy(); });
test('session hypothesis line only from the second measurement', ...);
test('cutting two heads adds two items and the header shows Kaydedilmemiş 2 kesim', ...);
test('the same head is not added twice', ...);
test('remove then Geri al restores the row', ...);
test('basket line counts savable and explore-only with the fake explore op', ...);
test('Modeli kaydet label counts savable items and dispatches SAVE_CLICKED', ...);
test('shelf restore moves an item back to the basket', ...);
```

- [ ] **Step 2: Run to verify failure** — `<WEBENV> npm --prefix web run test -- src/features/measure src/features/basket` → FAIL.

- [ ] **Step 3: Implement** per Interfaces.

- [ ] **Step 4: Verify** — `run test`, `run lint`, `run build` → pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/features/measure web/src/features/basket web/src/screens/Explore.tsx web/src/app/Header.tsx
git commit -m "feat(web): sequential real-effect measurement card, measured-only Kes (geçici), basket with badges, undo and explore shelf"
```

---

### Task 12: Davranış bul (concept hunt)

Spec §2 step 3b, §3 (refusal card, safety label); AR §2.1 Concept, F9; AS §2.5.

**Files:**
- Create: `web/src/screens/Concept.tsx`, `web/src/features/concept/{PresetChips.tsx,ExampleEditor.tsx,MethodPicker.tsx,LayerChart.tsx,FoundHeads.tsx,FoundDirection.tsx,RefusalCard.tsx,slug.ts}`, tests in `__tests__/`

**Interfaces:**
- Consumes: api `concepts`, `hunt`, `prefs`/`putPrefs`; `useJob`; `sessionStore.model.default_method`, `arch.n_layers`; `basketStore`, `selectionStore`, `navigate`.
- Produces:
  - `PresetChips`: API order (`code` selected by default, `refusal` second); name from `name_tr`/`name_en` by language; `+ Kendi kavramın`.
  - `RefusalCard`: shown the first time a `safety_ablation` preset is selected while `prefs.refusal_card_seen` is false; `Anladım` → `PUT /api/prefs` with `refusal_card_seen: true`. Copy states that the saved model will be labelled `safety-ablated` and that the label is best-effort.
  - `ExampleEditor`: two textareas (one request per line, empty lines ignored), live counts `{n} / en az 32`, note `Örnekler istektir, cevap değil · %20 doğrulamaya ayrılır`; `Ara` disabled until both sides have ≥ 32.
  - `slugify(name: string): string` → `^[A-Za-z0-9._-]{1,64}$`: Turkish letters transliterated (`ı→i ş→s ğ→g ü→u ö→o ç→c` and upper-case forms, `İ→I`), runs of whitespace → `-`, other characters dropped, cut to 64, empty → `kavram`; custom concept body `{ concept: { id, name_tr, name_en, positive, negative } }`.
  - `MethodPicker`: `attribution` "Tahmini etki" (note `GPU'da önerilen; CPU'da çok yavaş`), `activation_diff` "Etkinlik farkı", `direction` "Davranış yönü"; default `session.default_method`; no durations.
  - Hunt job with `ProgressCard` and `İptal`.
  - `LayerChart({ scores, refLayer })`: `scores.length === n_layers + 1` bars; label index 0 `gömme`, index i `K{i-1} sonrası`; `ref_layer` bar marked; `val_accuracy` line `doğrulamada doğruluk %{a} (GA %{lo}–%{hi}, n={n})`.
  - `FoundHeads`: top 8 by score, `Daha fazla göster` adds 8; columns K, H, tahmini skor, ölçülen `signal_drop` / `neutral_kl` or `ölçülmedi` when `measured` is null; `Haritada göster ve ölç` → `selectionStore.set(headIds)` + navigate `explore` (heads are cut only from the measured map).
  - `FoundDirection`: `direction_layer`, `Sepete ekle (geçici)` → item `{ op: { type: 'direction', d_model: direction_shape[0], vector_b64: direction }, semantics: 'direction', origin: 'concept', concept: <concept id, or 'refusal' for a safety concept> }`.

- [ ] **Step 1: Write the failing tests**

```ts
test('presets keep API order with code selected', ...);
test('refusal card shows once and stores the flag server-side', ...);   // second selection: no card; PUT body refusal_card_seen true
test('Ara stays disabled below 32 examples per side', ...);
test('slugify makes a valid id from Turkish text', () => expect(slugify('Şiir yazma ığüöç')).toBe('Siir-yazma-iguoc'));
test('method default follows session.default_method and shows no durations', () => { /* no /\d+\s*(dk|sn)/ in the picker */ });
test('layer chart labels n_layers + 1 bars', () => { renderChart({ scores: [0, 0.5, 1], refLayer: 2 }); expect(labels()).toEqual(['gömme', 'K0 sonrası', 'K1 sonrası']); });
test('found heads shows top 8 then 8 more', ...);
test('unmeasured head shows ölçülmedi', ...);
test('direction maps to DirectionOp with d_model from direction_shape', ...);
test('safety concept direction carries concept refusal', ...);
```

- [ ] **Step 2: Run to verify failure** — `<WEBENV> npm --prefix web run test -- src/features/concept src/screens/Concept` → FAIL.

- [ ] **Step 3: Implement** per Interfaces.

- [ ] **Step 4: Verify** — `run test`, `run lint`, `run build` → pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/screens/Concept.tsx web/src/features/concept
git commit -m "feat(web): concept hunt screen with presets, example editor, method choice, layer chart, heads and direction"
```

---

### Task 13: Compare

Spec §2 step 4, §3 (Wilson, McNemar/Newcombe, Holm, "belirsiz"); AR §2.1 Compare, F10; UX §1.6 (refreshed banner).

**Files:**
- Create: `web/src/screens/Compare.tsx`, `web/src/features/compare/{PairView.tsx,SuiteRunner.tsx,SummaryBanner.tsx,StatCards.tsx,SuiteTable.tsx,RefreshedBanner.tsx,basketConcept.ts}`, tests in `__tests__/`

**Interfaces:**
- Consumes: api `compare` (sync), `suite` (job), `concepts`; `sendableOps`, `savableOps`, `opsFingerprint`, `compareStore`, `saveFlowStore`, `useJob`.
- Produces:
  - `basketConcept(items): { conceptId: string | null; suiteTarget: string | null }` — `refusal` if any item has it; else the single shared concept id; `suiteTarget` from `/api/concepts` (`null` for `tone` or no concept).
  - `PairView`: prompt box `Kendin dene`, `İkisine de sor` → `POST /api/compare { prompt, recipe_ops: sendableOps(items), concept_id, max_new_tokens: 200 }`; two columns `Orijinal` / `Kesilmiş` rendered as text (`white-space: pre-wrap`); per-side `hedef davranış var/yok` badge from `target_hit`; quiet line `sıcaklık 0 · seed 42 · {mode}`; disabled with zero sendable ops; 409 → busy banner.
  - `SuiteRunner`: `Test setini koş` with `Hızlı` (default when `device === 'cpu'`) / `Tam`; body `{ recipe_ops: sendableOps(items), target: suiteTarget, quick }`; result stored as `compare.lastSuite = { fp: opsFingerprint(sendableOps), result }`.
  - Stale banner when `lastSuite.fp !== opsFingerprint(sendableOps(items))`: `Sepet değişti; sayılar eski. Yeniden koş.`
  - `SummaryBanner`: `summary.tr`/`summary.en` by language with `level` styling; when `summary` is null: `Bu kavramın test kategorisi yok; aşağıdaki genel sayılara bakın.`
  - `StatCards`: target rate orig → int with Wilson CI; KL `nötr istekler` with `kl_random`, `kl_random_p`, `kl_percentile` (and `kl_random_note` as footnote); perplexity orig → int; `Dosya boyutu değişmez`.
  - `SuiteTable`: category names from `concept.category.<id>` (`code, general, math, turkish, harmful_refusal`), n, orig/int rates with Wilson CI, difference with the paired interval, `belirsiz` when `diff.uncertain`, `(hedef)` on the target row, broken counts, footer `En küçük saptanabilir düşüş: {mde} puan`.
  - `RefreshedBanner`: when `saveFlow.state.kind === 'reviewReady'`: `Sayılar yenilendi; {n} kesim atıldı.` plus the ordinary `Modeli kaydet ({n} kesim)` button → navigate `save` (no modal).

- [ ] **Step 1: Write the failing tests**

```ts
test('compare body excludes opaque and pending-discard ops', ...);
test('pair view renders model output as text', () => { /* original '<b>x</b>' → no <b> element */ });
test('compare shows busy banner on 409', ...);
test('target_hit badges per side', ...);
test('suite runs quick on cpu with the basket concept target', ...);
test('summary fallback when target is null', ...);
test('belirsiz marks uncertain rows and (hedef) marks the target', ...);   // fixture job_suite
test('stale banner after the basket changes', ...);
test('refreshed banner shows dropped count and routes to save without a modal', ...);
test('no rate-level random-control line and no custom test-set button', ...);
```

- [ ] **Step 2: Run to verify failure** — `<WEBENV> npm --prefix web run test -- src/features/compare src/screens/Compare` → FAIL.

- [ ] **Step 3: Implement** per Interfaces.

- [ ] **Step 4: Verify** — `run test`, `run lint`, `run build` → pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/screens/Compare.tsx web/src/features/compare
git commit -m "feat(web): compare screen with side-by-side outputs, suite stats, uncertainty marks and post-discard refresh"
```

---

### Task 14: Save

Spec §2 step 4 (formats, Ollama off by default, recipe and card always), §4; AR §2.1 Save, F11; AS §2.5 items 3–4.

**Files:**
- Create: `web/src/screens/Save.tsx`, `web/src/features/save/{SaveForm.tsx,formats.ts,nameRule.ts,outDir.ts,ResultCard.tsx}`, tests in `__tests__/`

**Interfaces:**
- Consumes: api `system`, `estimate`, `exportModel`; `useJob`; `saveFlowStore` / `useSaveFlow`; `savableOps`, `basketConcept`, `basketStore.markSaved`.
- Produces:
  - `checkName(name: string): 'ok'|'empty'|'chars'|'reserved'|'trailing'` — `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`; reserved `CON PRN AUX NUL COM1–9 LPT1–9` (case-insensitive, with or without extension); trailing `.` or space. Messages `save.name.<code>` (the `chars` message says Turkish letters are not allowed in file names).
  - `checkOutDir(dir: string): 'ok'|'relative'|'unc'|'control'` — absolute (`^[A-Za-z]:[\\/]` or `/`), not `\\` or `//`, no control characters. `out_dir` sent = `join(dir, name)` (the server refuses an existing folder: `exists` error card).
  - `formatOptions(system): { id: 'safetensors'|'gguf_q4_k_m'|'gguf_q8_0'; enabled: boolean; note: 'experimental'|'needs_llama'|null }[]` — GGUF enabled only when `llama_cpp.found && llama_cpp.converter`, then labelled `deneysel`; default `safetensors`. Ollama toggle default off, enabled only for GGUF and when `system.ollama`.
  - Estimate: `POST /api/export/estimate { format, recipe_ops: savableOps, out_dir: dir, concept }` debounced 300 ms → `Kaydet · {bytes_needed} gerekli, {free_bytes} boş`; `enough === false` disables the button; `suggested_name` pre-fills the name once; `safety_ablated` shows the line `Bu model safety-ablated olarak etiketlenecek.`
  - Info lines: `Tarif (.nester.json), model kartı ve lisans her zaman kaydedilir.` and `Orijinal dosya salt okunur; kayıttan sonra parmak izi (SHA-256) kontrol edilir.`
  - `Kaydet` → `saveClicked()`; on `exportRequested` the form starts `POST /api/export { recipe_ops: savableOps, name, format, ollama, out_dir, concept }` and dispatches `EXPORT_STARTED`; job done → `EXPORT_DONE`, `basket.markSaved()` (autosave then writes `unsaved_count` 0); error → `EXPORT_FAILED` + ErrorCard.
  - `ResultCard`: path, name, `Orijinal dosyaya dokunulmadı` when `source_untouched`, source SHA-256 (monospace), `safety-ablated` badge when true.

- [ ] **Step 1: Write the failing tests**

```ts
test.each([['qwen2.5-0.5b-neshter', 'ok'], ['şiir', 'chars'], ['CON', 'reserved'], ['nul.txt', 'reserved'], ['model.', 'trailing'], ['', 'empty']])('checkName %s', (n, r) => expect(checkName(n)).toBe(r));
test.each([['C:\\Modeller', 'ok'], ['modeller', 'relative'], ['\\\\srv\\x', 'unc'], ['C:\\a\nb', 'control']])('checkOutDir %s', ...);
test('GGUF disabled without llama.cpp converter; safetensors default', ...);
test('Ollama only with GGUF and ollama on PATH', ...);
test('estimate not enough disables Kaydet', ...);
test('safety line shown when estimate says safety_ablated', ...);
test('export body uses savable ops and joined out_dir', ...);
test('result card shows untouched source, SHA and safety badge', ...);
test('export done marks basket saved and the unsaved chip disappears', ...);
test('Kaydet with an explore item opens the discard dialog first', ...);
```

- [ ] **Step 2: Run to verify failure** — `<WEBENV> npm --prefix web run test -- src/features/save src/screens/Save` → FAIL.

- [ ] **Step 3: Implement** per Interfaces.

- [ ] **Step 4: Verify** — `run test`, `run lint`, `run build` → pass.

- [ ] **Step 5: Commit**

```bash
git add web/src/screens/Save.tsx web/src/features/save
git commit -m "feat(web): save screen with name and folder rules, capability-gated formats, disk estimate, safety line and result card"
```

---

### Task 15: End-to-end, accessibility, offline/CSP, packaging, committed bundle

UX §2.8, §3.8, P2-12; AR F13, R3; scope-skeptic §4 item 5 ("fully local").

**Files:**
- Create: `web/playwright.config.ts`, `web/e2e/{server.ts,serve_e2e.py,happy-path.spec.ts,restore.spec.ts,leave.spec.ts,offline-csp.spec.ts,a11y.spec.ts}`
- Modify: `web/package.json` (`e2e` script), `tests/test_web_dist.py` (bundle tests), `web/DEV.md` (E2E section)
- Commit: `neshter/web/dist/**` (built)

**Interfaces:**
- Consumes: everything above; `tests/tiny_models.build_tiny_qwen2`; `neshter.server.app.create_app`; `NESHTER_HOME`.
- Produces:
  - `web/e2e/serve_e2e.py --port P --workdir W [--token T]` (run with `D:/neshter/venv/Scripts/python`): sets `NESHTER_HOME=W/home`; builds the tiny model once at `W/hub/models--neshter--tiny-e2e/snapshots/e2e` (kept across restarts, so the SHA is stable); `cli._run_server(create_app(T, P, roots={"huggingface": [W/"hub"]}), "127.0.0.1", P)` (Task 3; so `POST /api/shutdown` works in E2E). `W` must be under `D:\neshter\e2e-work`.
  - `server.ts`: `startServer(opts: { workdir: string; port?: number; token?: string }): Promise<{ url: string; port: number; token: string; stop(): Promise<void> }>` (free port via `net`; waits for `GET /api/session` with the token; `NESHTER_PYTHON` env default `D:/neshter/venv/Scripts/python.exe`), `freshWorkdir(name): string` (wipes `D:\neshter\e2e-work\<name>`).
  - `playwright.config.ts`: Chromium only, `workers: 1`, `outputDir: 'node_modules/.pw-results'`, `reporter: 'list'`, no `webServer` (specs start their own servers).
  - `tests/test_web_dist.py` additions: `test_built_ui_is_committed` (`git ls-files neshter/web/dist/index.html` is non-empty when `web/package.json` exists) and `test_committed_dist_matches_web_sources` (recomputes the Task 1 source hash in Python with the same rules and compares with `build-info.json`; skipped when dist is not tracked).

- [ ] **Step 1: Install the browser** — `<WEBENV> npx --prefix web playwright install chromium` → `D:\neshter\pw-browsers` contains `chromium-*`.

- [ ] **Step 2: Write the failing specs**

- `happy-path.spec.ts`: Home → `İçine bak` → row `neshter/tiny-e2e` → `Aç` → Explore; prompt `merhaba dünya` → `Çalıştır`; Liste view shows 8 rows; select `K0·H1` and `K1·H2` with the keyboard on the canvas (Tab, arrows, Space); `Gerçek etkiyi ölç` → a verdict sentence appears; `Kes (geçici)` → header `Kaydedilmemiş 2 kesim`; `Karşılaştır` → `İkisine de sor` → two outputs; `Test setini koş` (Hızlı) → table with `(hedef)`; Save: name `tiny-e2e-neshter`, folder default, `safetensors`, `Kaydet` → `Kaydedildi` card with a 64-hex SHA and `Orijinal dosyaya dokunulmadı`; header chip gone; the output folder exists under `<workdir>/home`.
- `restore.spec.ts`: load, measure, cut 1 head, wait for `PUT /api/draft/`; stop the server; old tab within 12 s shows `Sunucu kapandı; kesimlerin taslakta`; start a new server on a **different port** with the same workdir and a new token; open its URL → Home lists `Kaydedilmemiş çalışmalar: neshter/tiny-e2e · 1 kesim`; load the model → `Önceki oturumdan kaydedilmemiş 1 kesim var: geri yükle?` → `Geri yükle` → basket has 1 row. Then restart once more and assert the first tab (still open on the old port, server restarted on the same port with a new token) shows `Oturum bitti, sunucu yeniden başlatılmış`.
- `leave.spec.ts`: a second page in the same context on the same URL shows `Neşter başka bir sekmede açık` (close it); with 1 unsaved cut click `Kapat` → dialog with `Kaydet`, `Kaydetmeden çık`, `Vazgeç`; `Vazgeç` keeps the app; `page.reload()` raises a `beforeunload` dialog (`page.on('dialog')`, type `beforeunload`, dismissed); `Kapat` → `Kaydetmeden çık` → `Neşter kapandı` and the server process exits within 10 s.
- `offline-csp.spec.ts`: `page.addInitScript` collects `securitypolicyviolation` events into `window.__csp`; record every request; walk Home → Models → Explore (run) → Concept (preset selected) → Compare → Save; assert every request URL starts with `http://127.0.0.1:<port>/`, `window.__csp` is empty, `localStorage.length === 0`, and `document.fonts.check('16px "Inter Variable"', 'ĞğİıŞş')` is true after `document.fonts.ready`.
- `a11y.spec.ts`: `@axe-core/playwright` on Home, Models, Explore (2D and Liste), Concept, Compare, Save → no `serious`/`critical` violations; Liste parity: select, measure and cut entirely from the Liste table; `emulateMedia({ reducedMotion: 'reduce' })` → `html[data-motion="reduce"]`; focus ring visible on the heatmap (computed outline not `none`).

Python (`tests/test_web_dist.py`): `test_built_ui_is_committed`, `test_committed_dist_matches_web_sources`.

- [ ] **Step 3: Run to verify failure** — `<WEBENV> npm --prefix web run build && <WEBENV> npm --prefix web run e2e` → failures are product bugs found by the specs (fix them in the owning feature files, with a unit test where the bug is logic). `<PYTEST> tests/test_web_dist.py` → FAIL `test_built_ui_is_committed`.

- [ ] **Step 4: Fix until green** — `run e2e` → all specs pass; `run test`, `run lint` pass; `run build` prints `check-dist: ok` with the entry chunk under 250 000 bytes gzipped (record the measured size in `web/DEV.md`).

- [ ] **Step 5: Wheel check** — `UV_CACHE_DIR='D:\neshter\uv-cache' uv build --wheel -o 'D:\neshter\wheel-check'` then `D:/neshter/venv/Scripts/python -c "import zipfile,glob; z=zipfile.ZipFile(sorted(glob.glob(r'D:\neshter\wheel-check\*.whl'))[-1]); n=z.namelist(); assert 'neshter/web/dist/index.html' in n and any('latin-ext' in x for x in n); print('wheel ok')"` → `wheel ok`.

- [ ] **Step 6: Commit the bundle and tests, then verify**

```bash
git add web/playwright.config.ts web/e2e web/package.json web/DEV.md tests/test_web_dist.py neshter/web/dist
git commit -m "feat(web): Playwright e2e (happy path, restore across ports, leave, offline/CSP, a11y) and committed UI bundle"
```
Then `<PYTEST>` → all pass including `test_built_ui_is_committed` and `test_committed_dist_matches_web_sources`.

---

## Dependencies between tasks

Tasks 1 → 2 → 3 → 4 (backend prerequisites; 2–4 can run in parallel after 1 if branches are merged in order). Task 5 needs 2–4 (fixtures). Task 6 needs 5's types. Tasks 7 → 8 → 9 → 10 → 11 in order (shell, then flows that use it). Tasks 12, 13, 14 need 8 and 11. Task 15 last. The demoable "İçine bak" path is Tasks 1–11.
