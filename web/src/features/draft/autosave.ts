import type { StoreApi } from 'zustand';
import type { Api } from '@/api/client';
import { ApiError } from '@/api/errors';
import { unsavedCount } from '@/core/basket';
import { encodeDraft } from '@/core/draft';
import type { BasketState } from '@/core/stores/basket';
import type { SessionState } from '@/core/stores/session';
import { draftStatusStore } from './draftStatus';

export interface AutosaveDeps {
  api: Pick<Api, 'putDraft' | 'deleteDraft'>;
  basket: Pick<StoreApi<BasketState>, 'getState' | 'subscribe'>;
  session: Pick<StoreApi<SessionState>, 'getState' | 'subscribe'>;
  prompts: () => string[];
  debounceMs?: number;
}

interface ModelRef {
  name: string;
  sha256: string;
}

interface Snapshot {
  items: BasketState['items'];
  shelf: BasketState['shelf'];
  savedOps: string[];
  prompts: string[];
}

const modelOf = (session: AutosaveDeps['session']): ModelRef | null => {
  const model = session.getState().model;
  return model === null ? null : { name: model.name, sha256: model.source_sha256 };
};

/**
 * Mirrors the basket into `PUT /api/draft/{sha}`: 500 ms after the last change, and at once when the tab goes
 * hidden. An empty shelf with nothing unsaved (nothing added, or right after `markSaved()`) deletes the draft this
 * run wrote instead (F4). It never deletes a draft it did not write, and it never overwrites one either: while the
 * restore prompt has an unresolved earlier draft for the model (`draftStatusStore.pendingSha`), that draft is the
 * only copy of those cuts, so nothing is written for the model until it is restored (merged) or deleted. A 413 (large direction vectors) raises `draftStatusStore.tooLarge`; other failures
 * are left to the heartbeat and the stale screen. Returns the stop function.
 */
export function startAutosave(deps: AutosaveDeps): () => void {
  const wait = deps.debounceMs ?? 500;
  const written = new Set<string>(); // shas whose server draft this autosave wrote
  const status = draftStatusStore.getState();
  let timer: ReturnType<typeof setTimeout> | null = null;
  let queue: Promise<void> = Promise.resolve();

  // The basket and prompts are read when the flush is requested, not when the queued write runs.
  async function write(model: ModelRef, allowDelete: boolean, snapshot: Snapshot): Promise<void> {
    const { items, shelf, savedOps, prompts } = snapshot;
    if (draftStatusStore.getState().pendingSha === model.sha256) return;
    try {
      if (shelf.length === 0 && (items.length === 0 || unsavedCount(items, savedOps) === 0)) {
        if (allowDelete && written.has(model.sha256)) {
          await deps.api.deleteDraft(model.sha256);
          written.delete(model.sha256);
          status.setHasDraft(false);
          status.setTooLarge(false);
        }
        return;
      }
      const payload = encodeDraft({ items, shelf, savedOps, model, prompts });
      await deps.api.putDraft(model.sha256, payload);
      written.add(model.sha256);
      status.setHasDraft(payload.unsaved_count > 0);
      status.setTooLarge(false);
    } catch (error) {
      if (error instanceof ApiError && error.code === 'too_large') status.setTooLarge(true);
    }
  }

  function flush(model: ModelRef | null, allowDelete = true): void {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    if (model === null) return;
    const { items, shelf, savedOps } = deps.basket.getState();
    const snapshot = { items, shelf, savedOps, prompts: deps.prompts() };
    queue = queue.then(() => write(model, allowDelete, snapshot));
  }

  const schedule = () => {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => flush(modelOf(deps.session)), wait);
  };

  const unsubBasket = deps.basket.subscribe((s, prev) => {
    if (s.items !== prev.items || s.shelf !== prev.shelf || s.savedOps !== prev.savedOps) schedule();
  });
  // A model switch: write what is pending under the model it belongs to, and never delete (the leave dialog
  // promised that the draft of the model being left stays restorable).
  const unsubSession = deps.session.subscribe((s, prev) => {
    const before = prev.model?.source_sha256 ?? null;
    if (before === (s.model?.source_sha256 ?? null)) return;
    if (timer !== null && prev.model !== null) flush({ name: prev.model.name, sha256: prev.model.source_sha256 }, false);
    else if (timer !== null) flush(null);
    if (before !== null) written.delete(before);
  });
  const onVisibility = () => {
    if (document.visibilityState === 'hidden' && timer !== null) flush(modelOf(deps.session));
  };
  document.addEventListener('visibilitychange', onVisibility);
  // The earlier draft was restored or deleted: what was held back goes out now.
  const unsubStatus = draftStatusStore.subscribe((s, prev) => {
    if (prev.pendingSha !== null && s.pendingSha === null) schedule();
  });

  return () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    unsubBasket();
    unsubSession();
    unsubStatus();
    document.removeEventListener('visibilitychange', onVisibility);
  };
}
