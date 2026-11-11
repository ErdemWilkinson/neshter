import { useEffect, useState } from 'react';
import { useStore } from 'zustand';
import { getApi } from '@/api/currentApi';
import { ApiError, type ErrorCode } from '@/api/errors';
import type { BasketItem } from '@/core/basket';
import { decodeDraft } from '@/core/draft';
import { opKey } from '@/core/ops';
import { itemsToRecipe } from '@/core/recipe';
import { basketStore, type BasketState } from '@/core/stores/basket';
import { compareStore } from '@/core/stores/compare';
import { isLocked, jobsStore } from '@/core/stores/jobs';
import { sessionStore } from '@/core/stores/session';
import { plural, useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';
import { draftStatusStore } from './draftStatus';

type Decoded = Extract<ReturnType<typeof decodeDraft>, { ok: true }>;
interface Offer {
  sha: string;
  name: string;
  count: number;
  /** `null`: the server holds unsaved cuts this version cannot read; they can only be deleted. */
  draft: Decoded | null;
}
interface Failure {
  code: ErrorCode;
  detail: unknown;
}

/** The earlier draft first, then what was cut since; one cut once (by `opKey`) in each list. */
function mergeDraft(current: BasketState, draft: Decoded) {
  const union = (first: readonly BasketItem[], second: readonly BasketItem[]) => {
    const seen = new Set(first.map((item) => opKey(item.op)));
    return [...first, ...second.filter((item) => !seen.has(opKey(item.op)))];
  };
  return {
    items: union(draft.items, current.items),
    shelf: union(draft.shelf, current.shelf),
    savedOps: [...new Set([...draft.savedOps, ...current.savedOps])],
  };
}

const asFailure = (error: unknown): Failure =>
  error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) };

/**
 * After a model is loaded (or the page finds one already loaded), asks the server for that model's draft and offers
 * it when it holds unsaved cuts (F4). The offer stays until the person restores or deletes it, also when the basket
 * is not empty: while it is open `pendingSha` keeps autosave from overwriting the earlier cuts (I1). `Geri yükle`
 * checks the savable ops with `recipeCheck` first (a failed check lists the server's errors and restores nothing),
 * then merges the draft into the basket. A draft that cannot be decoded is protected the same way and can only be
 * deleted.
 */
export function RestorePrompt() {
  const t = useT();
  const sha = useStore(sessionStore, (s) => s.model?.source_sha256 ?? null);
  // One state object, so another model's stale offer is told apart by its sha instead of being reset in an effect.
  const [shown, setShown] = useState<{ offer: Offer; failure: Failure | null } | null>(null);
  const [busy, setBusy] = useState(false);
  const pendingSha = useStore(draftStatusStore, (s) => s.pendingSha);
  // The recipe check needs the model: while a job (a load, a run) holds it the server answers 409.
  const jobRunning = useStore(jobsStore, isLocked);
  const flaggedBusy = useStore(sessionStore, (s) => s.busy);
  const modelBusy = jobRunning || flaggedBusy;
  // A draft deleted elsewhere (Home list) clears `pendingSha`: the offer is gone with it.
  const offer = shown !== null && shown.offer.sha === sha && pendingSha === sha ? shown.offer : null;
  const failure = shown?.failure ?? null;
  const setOffer = (next: Offer | null) => setShown(next === null ? null : { offer: next, failure: null });
  const setFailure = (next: Failure | null) => setShown((current) => (current === null ? null : { ...current, failure: next }));

  useEffect(() => {
    if (sha === null) return;
    let cancelled = false;
    // Ask the list first: `GET /api/draft/<sha>` answers 404 when there is none, and the browser logs that as an error.
    getApi()
      .drafts()
      .then((list) => (list.some((d) => d.sha256 === sha && d.unsaved_count > 0) ? getApi().draft(sha) : null))
      .then((body) => {
        if (body === null) return;
        if (cancelled) return;
        const decoded = decodeDraft(body);
        if (body.unsaved_count <= 0 || (decoded.ok && decoded.items.length === 0)) return;
        const name = sessionStore.getState().model?.name ?? body.model_name;
        draftStatusStore.getState().setPendingSha(sha);
        setShown({ offer: { sha, name, count: body.unsaved_count, draft: decoded.ok ? decoded : null }, failure: null });
        draftStatusStore.getState().setHasDraft(true);
      })
      .catch(() => undefined); // a draft that vanished meanwhile (404) or an unreachable server: nothing to offer
    return () => {
      cancelled = true;
    };
  }, [sha]);

  if (offer === null) return null;

  const resolve = () => {
    draftStatusStore.getState().setPendingSha(null); // autosave resumes and writes the merged basket
    setOffer(null);
  };

  const restore = async () => {
    setBusy(true);
    setFailure(null);
    try {
      const { draft } = offer;
      if (draft === null) return;
      const recipe = itemsToRecipe(draft.items, { name: offer.name, sha256: offer.sha }, null, new Date().toISOString());
      if (recipe.interventions.length > 0) {
        const check = await getApi().recipeCheck({ recipe });
        if (!check.ok) {
          setFailure({ code: 'invalid_input', detail: check.errors });
          return;
        }
      }
      const current = basketStore.getState();
      const merged = mergeDraft(current, draft);
      basketStore.getState().replaceAll(merged);
      const last = draft.prompts.at(-1);
      if (last !== undefined && compareStore.getState().lastPrompt === null) compareStore.getState().setLastPrompt(last);
      resolve();
    } catch (error) {
      setFailure(asFailure(error));
    } finally {
      setBusy(false);
    }
  };

  const discard = async () => {
    setBusy(true);
    try {
      await getApi().deleteDraft(offer.sha);
      draftStatusStore.getState().setHasDraft(false);
      resolve();
    } catch (error) {
      setFailure(asFailure(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="restore-prompt glass" aria-label={t('restore.region')}>
      {offer.draft === null ? <p>{t('restore.unreadable')}</p> : <p>{plural('restore.prompt', offer.count)}</p>}
      {offer.draft !== null && offer.draft.unsupported > 0 && <p>{t('restore.unsupported', { n: offer.draft.unsupported })}</p>}
      <p>{t('restore.pending')}</p>
      {offer.draft !== null && modelBusy && <p role="status">{t('restore.waitModel')}</p>}
      {failure !== null && (
        <div>
          <p>{t('restore.failed')}</p>
          <ErrorCard code={failure.code} detail={failure.detail} />
        </div>
      )}
      <div className="restore-actions">
        {offer.draft !== null && (
          <Button variant="primary" disabled={busy || modelBusy} onClick={() => void restore()}>
            {t('restore.restore')}
          </Button>
        )}
        <Button disabled={busy} onClick={() => void discard()}>
          {t('restore.delete')}
        </Button>
      </div>
    </section>
  );
}
