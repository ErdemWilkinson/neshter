# Plan 2 (2D UI) — ledger and final review (kept for v0.2 triage)

# SDD ledger — plan: docs/superpowers/plans/2026-11-11-neshter-plan2-2d-ui.md

Spec: docs/superpowers/specs/2026-11-10-neshter-design.md (+ 3D spec 2026-11-10-neshter-3d-graph-design.md for shared contracts; debate synthesis .superpowers/sdd/2026-11-11-decisions-debate/synthesis.md). Owner approved all A–F 2026-11-11; autopilot overnight.

Preflight scan: preflight-scan.md (54 pair rows, 15 task rows, 46 findings F1–F36 + E1–E10).
Ruling: adopt every proposed ruling in preflight-scan.md as written (spec binding; smallest change) and patch the plan before Task 1; E1 → junction target <work>/web\node_modules (path keeps a node_modules segment); E4 → no worktree, branch feat/plan2-ui in the main checkout (editable venv points there) — cost if wrong: a plan patch round to revert.
Plan patched 0bfe685: all 46 applied. Patcher's own rulings (F3,F5,F22,F23,F27,F31,F32,F36 incl. 1000-item basket cap, heartbeat; F30 badge in T9+T11, T11 depends on T9; checkOutDir(defaultOutDir)) — Ruling: accept — cost if wrong: small UI rework.
Task 1: implementer DONE_WITH_CONCERNS 92fd9c2 (BASE 0bfe685).
Ruling: `npm ci`/`npm install` are never run with --prefix web (npm replaces the junction with a real dir on C:); they run in <work>/web per web/DEV.md, lock copied back — carry into every later dispatch — cost if wrong: a C: node_modules inside OneDrive.
Ruling: ESLint override globs use `**/`-prefixed patterns (İ in repo path breaks literal src/ globs) — carry forward — cost if wrong: an unenforced ban, caught by negative tests.
Ruling: CSP assertion change in test_security_headers_on_all_responses ("script-src 'self'") accepted — plan's GC mandates explicit script-src.
Note for owner: implementer wrote eslint.config.js via Bash after the ecc config-protection hook blocked Edit (new file required by plan; reviewer to confirm no rule weakened).
Task 1: review → Needs fixes: I1 e2e lint override dead under İ path bug.
Task 1: minor (deferred): check-dist test tmp dirs not cleaned; check-dist tests miss css/external-url + data:application/font + CLI exit; CSP script-src guard order-dependent (parse directives); loose **/core/** globs; data:, favicon; globalIgnores fixtures pattern also dead under path bug.
Task 1: fix round 1/5 (I1 fixed by implementer; commits 92fd9c2..ef87748) — re-review pending
Task 1: fix round 1/5 (1 addressed, 0 open; commits 92fd9c2..ef87748)
Task 1: minor (deferred): toLowerCase wrapper uses String(this) (null/undefined don't throw); note vitest isolate dependency in DEV.md
Task 1: complete (commits 0bfe685..ef87748, review clean)
Task 2: implementer DONE b9af72a (BASE ef87748); exit_line wiring is Task 3's job
Task 2: ⚠ exit_line wiring → confirmed owned by Task 3 (Ctrl+C draft line). ⚠ draft basket/shelf opaque → Task 6 encoder must exclude source_path/model output (carry into T6 dispatch).
Task 2: minor (deferred): request.body() buffers before size check (plan-mandated; stream + abort); chunked >limit branch untested; _scan re-parses all drafts per put/list; fixed .tmp name, no fsync; _remove pops _written even when unlink raises PermissionError.
Task 2: complete (commits ef87748..b9af72a, review clean)
Task 3: implementer DONE a0fc2b7 (BASE b9af72a)
Task 3: review → Needs fixes: I1 shutdown hangs while an SSE job stream is open (18.6 s measured).
Task 3: minor (deferred): no explicit bad Origin/Host test for /api/shutdown; errors.py:84 line >120 chars.
Task 3: fix round 1/5 (1 addressed, 0 open; commits a0fc2b7..8abcba0)
Task 3: minor (deferred): SSE shutdown test can hang forever if client thread fails (add try/finally shutdown); _free_port race; JobManager.shutdown leaves queued jobs 'queued' (mark cancelled like cancel()).
Task 3: complete (commits b9af72a..8abcba0, review clean)
Task 4: implementer DONE a501703 (BASE 8abcba0); noted race cancel-after-measure_count (JobManager scope)
Task 4: minor (deferred): last-draw futility branch untested though report claims (add explicit cases + mixed draw distribution); mid-file imports noqa E402; measure_count cancel race (JobManager).
Task 4: note → carry to T5: /api/session unloaded shape {"loaded": false} lacks measure_count — client schema treats it optional.
Task 4: complete (commits 8abcba0..a501703, review clean)
Task 5: implementer DONE 5780b90 (BASE a501703)
Task 5: review → Needs fixes: I1 modelEntry.arch ints must be nullable (scan.py emits None) — one bad model breaks whole list.
Ruling: upgrade reviewer Minor 1 (SSE onEvent TypeError swallowed → endless 1 s reconnect) to Important and Minor 5 (heartbeat onUp after stop) into round 1 — both are fragile behaviour in code later tasks build on — cost if wrong: one extra small diff.
Task 5: minor (deferred): reconnect budget resets on each first event; result payloads never runtime-validated (validate in consumers T6+ or drop unused loadResult/sampleResult); strict prefs/draft enums; contract test leaves draft in session NESHTER_HOME; weak asserts (bootstrap localStorage, sse events<=1, no reader-cancel test); layering api→app, test helper cross-import; _ABS_PATH misses UNC/embedded paths; real ram/free bytes in fixtures.
Task 5: fix round 1/5 (I1–I3 fixed; commits 5780b90..b0e9e55) — re-review pending
Task 5: fix round 1/5 (3 addressed, 0 open; commits 5780b90..b0e9e55)
Task 5: minor (deferred): fix report lacks raw runner output; archInfo (loaded session) counts non-nullable — loaded ArchInfo is int dataclass, likely fine; activeJob cleared on consumer error.
Task 5: complete (commits a501703..b0e9e55, review clean)
Task 6: implementer DONE_WITH_CONCERNS 67f50f9 (BASE b0e9e55). Carry to T7: App must call setApi(createApi(...)) after bootstrap (web/src/api/currentApi.ts). Carry to T10: fromEffectResult throws ZodError on malformed payload — catch it.
Task 1 (late): lint-scope.test.ts times out at 5 s on ESLint cold start (8–16 s) → full `npm run test` red. Ruling: fix in T7's dispatch (raise that file's timeout, e.g. 60 s) — cost if wrong: one red suite until fixed.
Task 6: minor (deferred): ExtensionOp makes OpaqueOp assignable to KnownOp (add opaque?: undefined); opKey re-stringifies vectors (WeakMap cache); recipe.ts imports clip from draft.ts; useJob clears active on network-lost/reset (doc overstates); undoRemove loses item when add returns 0; no captured RED; single bad item invalidates whole draft.
Carry to T8: autosave must handle 413 too_large (large direction vectors).
Task 6: complete (commits b0e9e55..67f50f9, review clean)
Task 7: implementer DONE_WITH_CONCERNS 49168ae (BASE 67f50f9); lint-scope timeout fixed (60 s). Carry to T8: leave dialog + real draft mirror for StaleScreen hasDraft; reuse web/src/app/__tests__/fakeServer.ts.
Task 7: minor (deferred): check-i18n CLI guard can no-op (spawn test + realpath); DeviceBanner shows Turkish loader note in en UI; BusyBanner busy inference + duplicate app.busy key; prefs write-back after failed GET overwrites / concurrent patches lost; errors.oom/unsupported_arch wording; ProgressCard job.message Turkish in en.
Ruling: fold T7 minors 4 (closeApp swallows shutdown failure → stale 'down' while server alive) and 6 (Dialog native close desync → resync via close event) into Task 8, which owns the leave/close dialog flows — cost if wrong: slightly larger T8 diff.
Task 7: complete (commits 67f50f9..49168ae, review clean)
Ruling (owner choice "1", budget 24% weekly): economy mode for Tasks 8–15 — implementer only (sonnet), no per-task review; ONE final whole-branch review + ONE fix wave at the end; Plan 3 not started this week — cost if wrong: defects caught later in one batch instead of per task.
Task 8: implementer hit rate limit (resets 14:40); work saved in tree; resume agent a8dada48a4e3e15e3 via SendMessage.
Task 8: complete (commits 49168ae..fe69541, economy mode — no task review). For final review: autosave never deletes drafts it didn't write (old-session drafts linger); prompt edits only saved on next basket change; discard dialog labels by op.type (Plan 3 needs label hook); DraftPayload interface→type; no RED.
Carry: UnsavedWork, useSaveFlow, discard + nothing-to-save dialogs exist but are unmounted → mount in T9 (Home) and T13/T14 (Compare/Save).
Task 9: complete (commits fe69541..5520d68, economy). Note: loading a different-sha model clears basket/shelf/compare/selection (old cuts flushed to own draft) — measure store NOT reset → handle in T10/T12. useSaveFlow still unmounted → T13/T14.
Task 10: complete (commits 5520d68..b752552, economy). Notes for final review: added exploreStore.ts + format.ts; eslint-disable for canvas role=group; MLP aria label "MLP birimleri" (forbidden-copy); measure/cut buttons → T11 via selectionStore.
Task 11: complete (commits b752552..7907ba6, economy).
Ruling: direction row label "Yön · {d} boyut" accepted (DirectionOp has no layer); T12 may attach a display-only label (e.g. "Yön · K12 sonrası") to hunt-produced items without changing the op schema — cost if wrong: less informative label.
Task 12: complete (commits 7907ba6..7804584, economy). Final-review notes: App.test router assertion wrapped in waitFor; BasketItem.foundAt display-only, not in drafts; prefs write-back after failed GET (T7 minor) also hits refusal-card PUT; preset examples read-only.
Task 13: complete (commits 7804584..1e6c4ef, economy). Final-review notes: compareStore.lastSuite lacks target → stale "(hedef)" after basket change; BasketPanel not on Compare. Carry to T14: reuse features/saveflow/SaveFlowDialogs.tsx on Save; Save must call saveClicked().
Task 14: complete (commits 1e6c4ef..f9087e2, economy). Final-review notes: extra "Yeni bir kayıt yap" DISMISS button; export ErrorCard lost if user leaves Save mid-export; SaveForm imports formatBytes cross-feature.
Task 15: complete (commits f9087e2..3563679, economy). Final-review notes: e2e/flow.ts extra; tsconfig includes e2e; locale tr-TR in Playwright; server --token=<t> needed because a '-'-leading token crashed CLI parsing (check real launcher path).
Final review (opus, scope 49168ae..HEAD product code): With fixes. I1 restore-prompt "Sonra" + new cut → autosave overwrites prior draft (autosave.ts:63, RestorePrompt.tsx). I2 Save re-run İptal never reaches server (different useJob instance; SaveForm.tsx:288). Must-fix minors: DiscardDialog raw op types; markSaved marks cuts added mid-export; model switch doesn't reset save flow. T15 '-' token: e2e-only, no action.
Fix wave dispatched (base 3563679).
Fix wave: 42e066a (I1,I2,M1–M3 fixed; 'Sonra' button removed — restore prompt stays until Geri yükle/Sil; residual: cut flushed before draft GET could still overwrite). Re-review pending.
Fix wave re-review: all 5 addressed; new minors: leave/stale copy promises cuts are in draft while pendingSha blocks autosave; pendingSha single-sha. Plan 2 complete.

---

# Final whole-branch review: Plan 2 (Neşter 2D React UI), feat/plan2-ui

Range: 49168ae..3563679 (Tasks 8–15, economy mode, no per-task review). Package diff: `final-review.diff`. Tasks 1–7 were
taken as reviewed; they were checked only where 8–15 build on them (basket store, save-flow reducer, useJob, draft codec).
Single pass by one reviewer, done in two read-throughs of the diff (app/draft/leave/saveflow/save first, then
map/measure/concept/compare/models/recipe). Read-only: one focused pytest run (`tests/test_web_dist.py`: 7 passed).

## Strengths

- **Explore-only cuts cannot be exported.** The export body is `savableOps(items)` (`SaveForm.tsx`), and compare,
  suite and the re-run use `sendableOps` (known ops, not pending discard). Opaque ops count as explore-only, so they are
  discarded too and never sent.
- **The second confirmation is the refreshed Compare plus an ordinary Save.** `RefreshedBanner` only navigates, and
  `SAVE_CLICKED` is dispatched only from the Save form, which matches F19. The basket moves items to the shelf only
  after the quick suite comes back `done`. If the re-run fails, nothing on the basket changes (`clearPendingDiscard`).
- **The safety label cannot be dropped by the UI.** The server decides it (`_is_safety`: concept `refusal`, safety
  heads and safety directions recorded during a refusal hunt; `export.py` ORs with `is_refusal`). The UI only shows
  the result (`estimate.safety_ablated`, `ResultCard`). A preset with `safety_ablation` records `refusal` as the basket
  concept, and `basketConcept` lets `refusal` win over every other concept.
- **Imported directions** show `doğrulanmamış yön` in the recipe preview and on basket rows (`isUnverifiedDirection`).
- **Measurement copy matches the global constraints exactly** (significant, not significant, futility, the session
  line, zero-ablation, the heatmap legend with `tahmini`). There is no rate-level random-control line: the random
  control appears only as a KL band and a KL p-value. The ETA appears only after 5 % progress and is labelled `en çok`
  (the plan allows this; it is extrapolated from progress, not invented).
- **No XSS path found.** Token pieces, model output, model and recipe names, the server summary and notes are all
  rendered as React text. There is no `innerHTML` or `dangerouslySetInnerHTML`, and `localStorage` appears only in a
  comment. The committed bundle contains no request host other than the serving origin (the only URLs are string
  literals from zod, SVG and React). The dist hash test passes.
- **Model switch is handled in the right order.** `setModel` runs first, so autosave flushes the pending cuts under
  the *old* sha and never deletes them. After that, basket, compare, selection, measure and explore are reset. The
  drafts carry an explicit field list, so measurements, `source_path` and model output cannot leak into them.
- **The single-tab lock** is clean: one answer per lock manager, which is safe under StrictMode, and a broken lock
  manager fails open.
- **T7 folded minors are fixed.** A failed shutdown now shows a close-failure card and the app stays as it was;
  `Dialog` resyncs when the browser closes the native dialog on its own.

## Issues

### Critical
None.

### Important

**I1. A pending draft from an earlier session is silently overwritten by the first new cut (loss of unsaved work).**
- Where: `web/src/features/draft/autosave.ts:63` (unconditional `putDraft`), `web/src/features/draft/RestorePrompt.tsx:67`
  (the prompt hides once the basket is non-empty) and `:126` (`Sonra` only hides the prompt).
- Trigger: after a restart the restore prompt offers "N kesim var: geri yükle?". The person presses `Sonra`, or
  ignores the prompt and cuts a head. 500 ms later autosave PUTs the one-item basket to the same `/api/draft/{sha}`,
  which replaces the N earlier cuts.
- Effect: the prompt disappears, and the earlier cuts are gone with no warning.
- Contradictions: the autosave doc comment claims "It never deletes a draft it did not write: while the restore
  prompt is still open, the old draft is the only copy", but an overwrite is a delete. `Sonra` ("later") also
  promises a later that cannot come.
- Related T6 minor: a draft with one bad item fails to decode, so no prompt is shown at all, and the next cut
  overwrites that draft as well.
- Fix: autosave must not PUT for a sha while that sha has an unresolved server draft. The restore prompt (or
  `draftStatusStore`) holds a "draft pending" flag for that sha, set when it finds a draft and cleared on
  `Geri yükle` or `Sil`. While the flag is set:
  - either keep the prompt visible even when the basket is non-empty, and offer to merge (the basket store's `add`
    already de-duplicates by `opKey`);
  - or skip the write and show a notice.

  The smallest version: merge the draft items into the basket on restore instead of requiring an empty basket, and
  block the write until the person chooses.

**I2. `İptal` during the save flow's re-run does not cancel the server job.**
- Where: `web/src/features/save/SaveForm.tsx:288` (with `useSaveFlow.ts`).
- Cause: every `useSaveFlow()` call creates its own `useJob()`. The re-run suite is started by the instance inside
  `SaveFlowDialogs` (its `confirmDiscard`). The progress card's `İptal` calls `flow.cancel()` on the *SaveForm*
  instance, and that instance's `cancelJob()` has no current job, so it does nothing.
- Effect:
  - The flow goes to idle and the progress card disappears, but the quick suite keeps running and holding the model.
  - The Save button stays disabled (`locked`) with no visible reason until the suite finishes.
  - When the suite finishes, `fail()` runs silently.
- Why the tests miss it: the unit test passes because its harness cancels from the same hook instance.
- Fix (either):
  - make `cancel` in `useSaveFlow` cancel by the job id in the flow state (`api.cancelJob(before.jobId)`) instead of
    through the instance's `useJob`;
  - or hold the flow's `useJob` in a module-level/store singleton.

  Add a test that renders `Save` (SaveForm + SaveFlowDialogs) and cancels from the progress card.

### Minor

1. **The discard dialog lists raw op types.** `saveflow/DiscardDialog.tsx:11` builds the list from `item.op.type`, so
   the most important save-flow dialog reads "…kaydedilemez: direction." (English) in the Turkish UI. Use
   `itemLabel(item)` from `features/basket/labels.ts`; keep the opaque label. This is also the label hook Plan 3 needs.
   **Must fix.**
2. **`markSaved()` marks what is in the basket at the end of the export, not what was exported.** See
   `core/stores/basket.ts:88` and its call at `SaveForm.tsx:162`.
   - Race: while an export runs, the person can still cut a head whose selection was measured earlier, because
     `CutButton` does not check the job lock.
   - Effect: that head is marked saved without being exported. The unsaved chip and the leave guard then show
     nothing, and autosave can delete the draft (`unsaved == 0`, shelf empty).
   - Fix: snapshot `opKey` of the exported ops in `runExport` and pass them to `markSaved(keys)`.

   **Must fix (cheap; save-flow honesty).**
3. **The save flow is not reset when the model changes.** `resetForNewModel` (`features/models/LoadScreen.tsx:35`)
   does not reset `saveFlowStore`. After a save on model A, loading model B shows A's `Kaydedildi` result card, with
   A's path and SHA, on the Save screen; a `reviewReady` state would also show "Sayılar yenilendi" on B's Compare
   screen. Dispatch `DISMISS`/`CANCEL`, or add `reset()`. **Must fix (one line).**
4. **An export failure is lost when the person leaves the Save screen mid-export.** The ledger notes this for T14 at
   `SaveForm.tsx:145`. The flow returns to idle with no reason shown. Store the export failure in
   `saveFlowStore.error`, the same as re-run failures. Should fix; can wait.
5. **The leave dialog and the stale screen can promise a draft that is out of date.** After a 413, `hasDraft` stays
   true from an older write, and the copy "kesimleriniz taslakta kalır" is then only partly true. `DraftNotice`
   already warns; with an 8 MB limit this is rare. Can wait.
6. **Shelf rows have no badges.** `ShelfView.tsx:17` drops `doğrulanmamış yön` (F30 says "every row"). Can wait.
7. **`beforeunload` stays armed on the closed screen.** After `Kapat` → `Kaydetmeden çık` → shutdown, closing the tab
   still raises the browser confirmation while the basket is dirty. Disarm when `closed`. Can wait.
8. **Unformatted scores.** `FoundHeads` uses `score.toFixed(2)`, which is not locale-formatted (`formatFixed`).
   Can wait.
9. **Two leave-guard listeners.** `useLeaveGuard` is mounted twice (App and Models), so two `beforeunload` listeners
   are registered. This is harmless. Can wait.
10. **Custom concepts cannot get the safety label from their name.** A custom concept named "refusal" gets the id
    `refusal-ozel`, so the client never records `refusal` for it. The server's best-effort detection (safety
    directions and heads) is the only label path for a custom refusal-like concept. This is acceptable under the
    "best-effort" copy. Can wait.

## Triage of the ledger's deferred minors and "final-review notes"

**Must fix before merge** (all in the web save/draft path):
- T8 "discard dialog labels by op.type" → Minor 1 above.
- T8 "autosave never deletes drafts it didn't write" is fine as stated, but the real defect next to it is I1
  (overwrite). Fix I1.
- T6 "single bad item invalidates whole draft" → handle inside the I1 fix: if decoding fails, do not overwrite;
  item-level skipping can wait.
- T14 "export ErrorCard lost if user leaves Save mid-export" → recommended (Minor 4); acceptable to defer if the fix
  wave is tight.

**Can wait (no user harm found, or test or doc hygiene only):**
- **T1:** check-dist tmp dirs and coverage gaps, CSP guard order, loose globs, the `data:` favicon, the dead
  globalIgnores pattern, the `toLowerCase` wrapper, the vitest isolate note.
- **T2:** body buffered before the size check, the chunked branch untested, `_scan` re-parsing, the fixed `.tmp` name
  without fsync, `_remove` with PermissionError.
  - Single-tab use plus the serialized autosave queue make the `.tmp` collision practically unreachable.
- **T3:** no Origin/Host test for shutdown, the long line, the SSE test hang guard, the `_free_port` race, queued jobs
  left `queued` on shutdown.
- **T4:** futility last-draw tests, E402, the `measure_count` cancel race.
- **T5:** reconnect budget, strict enums, the contract test's NESHTER_HOME, weak asserts, layering, `_ABS_PATH`
  missing UNC paths, real RAM numbers in fixtures, activeJob cleared on consumer error.
  - "Result payloads never runtime-validated" is now resolved in practice: the suite, hunt, export and sample results
    are `safeParse`d, and effect results go through `fromEffectResult` inside try/catch.
- **T6:** `ExtensionOp` assignability, `opKey` caching, the `clip` import, the `useJob` doc wording, `undoRemove` when
  `add` returns 0, no RED.
- **T7:** check-i18n CLI guard, the DeviceBanner and ProgressCard Turkish server text in the en UI, the duplicate
  BusyBanner key, `errors.oom` wording.
  - Prefs write-back after a failed GET (it also hits the refusal-card PUT in T12) should be fixed in the next wave:
    a failed GET followed by `Anladım` can reset `lang` and other prefs. It is not a merge blocker.
- **T8:** prompt edits are saved only on the next basket change; `DraftPayload` is now a type; no RED.
- **T10:** `exploreStore` and `format.ts` were added; the eslint-disable for `role=group` on the canvas is required by
  the spec; the MLP labels ("En güçlü MLP birimleri", "MLP · en güçlü 50") are the plan's MLP top-50 list, not the
  forbidden "MLP nöronları" tab.
- **T12:** the `waitFor` in App.test, `foundAt` being display-only (verified: it is not in drafts), read-only presets.
- **T13:** `lastSuite` has no target, so a stale "(hedef)" appears after a basket change (the stale banner already
  shows); `BasketPanel` is not on Compare.
- **T14:** the extra `Yeni bir kayıt yap` DISMISS button is fine; the cross-feature `formatBytes` import.
- **T15:** `e2e/flow.ts`, tsconfig including `e2e`, the `tr-TR` locale. All fine.
  - **The CLI token starting with `-` is not a product issue.** The real launcher (`neshter.bat` →
    `neshter serve` → `cli._serve`, `neshter/cli.py:96`) generates the token in-process with
    `secrets.token_urlsafe` and never parses it as a command-line argument. Only `e2e/serve_e2e.py --token` takes it
    as an argument, and `--token=<t>` already handles that. The client's `#token=([^&]+)` accepts a leading `-`. No
    action.

## Declined to judge

- Server-side safety detection quality (`_is_safety`, cosine thresholds): this is Plan 1/server scope; the UI only
  displays it.
- Visual design and token fidelity in the 782-line `app.css`: outside the focus list and the budget.
- The e2e specs' own robustness (timeouts, flakiness): reported passing by the implementer; not re-run, because
  that would need a server and a browser.
- Tasks 1–7 code beyond what Tasks 8–15 depend on: those tasks were already reviewed.

## Assessment

**Ready to merge: With fixes.**

Reasoning: the save flow's core guarantees hold. Explore-only and opaque ops are never exported, the second
confirmation is Compare plus an ordinary Save, the safety label is decided by the server, the copy is exact, and
there is no XSS or external host. Two defects hurt real users:
- I1 can silently destroy an earlier session's unsaved cuts, which is the exact thing drafts exist to protect.
- I2 makes `İptal` during the re-run a no-op that leaves the model locked with no explanation.

Fix these with Minors 1–3 (all small) in the planned fix wave, then rebuild and recommit `dist`; the hash test
enforces the rebuild.
