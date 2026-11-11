import { useEffect, useState } from 'react';
import { useStore } from 'zustand';
import { getApi } from '@/api/currentApi';
import { ApiError, type ErrorCode } from '@/api/errors';
import { decodeDraft } from '@/core/draft';
import { itemsToRecipe } from '@/core/recipe';
import { basketStore } from '@/core/stores/basket';
import { compareStore } from '@/core/stores/compare';
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
  draft: Decoded;
}
interface Failure {
  code: ErrorCode;
  detail: unknown;
}

const asFailure = (error: unknown): Failure =>
  error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) };

/**
 * After a model is loaded (or the page finds one already loaded), asks the server for that model's draft and offers
 * it, but only when it decodes, holds unsaved cuts and the basket is still empty (F4). `Geri yükle` checks the
 * savable ops with `recipeCheck` first; a failed check lists the server's errors and restores nothing.
 */
export function RestorePrompt() {
  const t = useT();
  const sha = useStore(sessionStore, (s) => s.model?.source_sha256 ?? null);
  const basketEmpty = useStore(basketStore, (s) => s.items.length === 0);
  // One state object, so another model's stale offer is told apart by its sha instead of being reset in an effect.
  const [shown, setShown] = useState<{ offer: Offer; failure: Failure | null } | null>(null);
  const [busy, setBusy] = useState(false);
  const offer = shown !== null && shown.offer.sha === sha ? shown.offer : null;
  const failure = shown?.failure ?? null;
  const setOffer = (next: Offer | null) => setShown(next === null ? null : { offer: next, failure: null });
  const setFailure = (next: Failure | null) => setShown((current) => (current === null ? null : { ...current, failure: next }));

  useEffect(() => {
    if (sha === null) return;
    let cancelled = false;
    getApi()
      .draft(sha)
      .then((body) => {
        if (cancelled) return;
        const draft = decodeDraft(body);
        if (!draft.ok || body.unsaved_count <= 0 || draft.items.length === 0) return;
        if (basketStore.getState().items.length > 0) return;
        const name = sessionStore.getState().model?.name ?? body.model_name;
        setShown({ offer: { sha, name, count: body.unsaved_count, draft }, failure: null });
        draftStatusStore.getState().setHasDraft(true);
      })
      .catch(() => undefined); // no draft (404) or an unreachable server: nothing to offer
    return () => {
      cancelled = true;
    };
  }, [sha]);

  if (offer === null || !basketEmpty) return null;

  const restore = async () => {
    setBusy(true);
    setFailure(null);
    try {
      const { draft } = offer;
      const recipe = itemsToRecipe(draft.items, { name: offer.name, sha256: offer.sha }, null, new Date().toISOString());
      if (recipe.interventions.length > 0) {
        const check = await getApi().recipeCheck({ recipe });
        if (!check.ok) {
          setFailure({ code: 'invalid_input', detail: check.errors });
          return;
        }
      }
      if (basketStore.getState().items.length === 0) {
        basketStore.getState().replaceAll({ items: draft.items, shelf: draft.shelf, savedOps: draft.savedOps });
        const last = draft.prompts.at(-1);
        if (last !== undefined && compareStore.getState().lastPrompt === null) compareStore.getState().setLastPrompt(last);
      }
      setOffer(null);
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
      setOffer(null);
    } catch (error) {
      setFailure(asFailure(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="restore-prompt glass" aria-label={t('restore.region')}>
      <p>{plural('restore.prompt', offer.count)}</p>
      {offer.draft.unsupported > 0 && <p>{t('restore.unsupported', { n: offer.draft.unsupported })}</p>}
      {failure !== null && (
        <div>
          <p>{t('restore.failed')}</p>
          <ErrorCard code={failure.code} detail={failure.detail} />
        </div>
      )}
      <div className="restore-actions">
        <Button variant="primary" disabled={busy} onClick={() => void restore()}>
          {t('restore.restore')}
        </Button>
        <Button disabled={busy} onClick={() => void discard()}>
          {t('restore.delete')}
        </Button>
        <Button variant="ghost" disabled={busy} onClick={() => setOffer(null)}>
          {t('restore.later')}
        </Button>
      </div>
    </section>
  );
}
