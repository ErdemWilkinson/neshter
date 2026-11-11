import { useCallback } from 'react';
import { useStore } from 'zustand';
import type { JsonOp } from '@/api/client';
import { getApi } from '@/api/currentApi';
import { ApiError } from '@/api/errors';
import { suiteResult } from '@/api/schemas';
import { useJob } from '@/api/useJob';
import { navigate } from '@/app/router';
import { sendableOps } from '@/core/basket';
import { basketConcept } from '@/core/basketConcept';
import { opsFingerprint } from '@/core/fingerprint';
import type { SaveFlowState } from '@/core/saveflow';
import { basketStore } from '@/core/stores/basket';
import { compareStore } from '@/core/stores/compare';
import { saveFlowStore, type SaveFlowFailure } from './saveFlowStore';

export interface SaveFlow {
  state: SaveFlowState;
  /** Why the last re-run failed, or `null`. */
  error: SaveFlowFailure | null;
  /** Only the Save screen's `Kaydet` calls this (F19). */
  saveClicked(): void;
  confirmDiscard(): void;
  cancel(): void;
  dismiss(): void;
}

const toFailure = (error: unknown): SaveFlowFailure =>
  error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) };

/**
 * The save flow on top of the pure reducer. After the discard is confirmed the explore-only items are marked
 * pending-discard (so `sendableOps` leaves them out), then, in this order: (1) when there is a compare prompt,
 * `/api/compare` (synchronous, 409 while a suite job holds the model: F18b) and `lastPair` is stored; (2) a quick
 * suite job. Only when that job is done do the items move to the shelf. Any failure or cancellation clears the
 * pending flags: the basket is left as it was.
 */
export function useSaveFlow(): SaveFlow {
  const state = useStore(saveFlowStore, (s) => s.state);
  const error = useStore(saveFlowStore, (s) => s.error);
  const job = useJob<unknown>();
  const { start, cancel: cancelJob } = job;

  const recompare = useCallback(
    async (ids: string[]) => {
      const flow = saveFlowStore.getState();
      const api = getApi();
      const running = () => saveFlowStore.getState().state.kind === 'recomparing';
      const fail = (cause?: SaveFlowFailure) => {
        basketStore.getState().clearPendingDiscard();
        flow.dispatch({ type: 'RECOMPARE_FAILED' });
        if (cause !== undefined) flow.setError(cause);
      };
      try {
        basketStore.getState().markPendingDiscard(ids);
        const items = basketStore.getState().items;
        const ops = sendableOps(items) as JsonOp[];
        const concepts = await api.concepts().catch(() => []);
        const { conceptId, suiteTarget } = basketConcept(items, concepts);

        const prompt = compareStore.getState().lastPrompt;
        if (prompt !== null) {
          const pair = await api.compare({ prompt, recipe_ops: ops, concept_id: conceptId });
          if (!running()) return fail(); // cancelled while comparing
          compareStore.getState().setLastPair(pair);
        }
        if (!running()) return fail();

        const final = await start(async () => {
          const started = await api.suite({ recipe_ops: ops, target: suiteTarget, quick: true });
          if (running()) flow.dispatch({ type: 'RECOMPARE_STARTED', jobId: started.job_id });
          else await api.cancelJob(started.job_id).catch(() => undefined); // cancelled while the job was starting
          return started;
        });
        const parsed = suiteResult.safeParse(final.result);
        if (final.state !== 'done' || !running() || !parsed.success) {
          return fail(final.state === 'error' ? { code: final.error_code ?? 'internal', detail: final.error } : undefined);
        }
        compareStore.getState().setLastSuite({ fp: opsFingerprint(sendableOps(basketStore.getState().items)), result: parsed.data });
        flow.dispatch({ type: 'RECOMPARE_DONE' });
        basketStore.getState().moveToShelf(ids);
        compareStore.getState().setRefreshedDropped(ids.length);
        navigate('compare');
      } catch (cause) {
        fail(toFailure(cause));
      }
    },
    [start],
  );

  const saveClicked = useCallback(() => {
    saveFlowStore.getState().dispatch({ type: 'SAVE_CLICKED' });
  }, []);

  const confirmDiscard = useCallback(() => {
    const flow = saveFlowStore.getState();
    const before = flow.state;
    const after = flow.dispatch({ type: 'DISCARD_CONFIRMED' });
    if (before.kind === 'needsDiscard' && after.kind === 'recomparing') void recompare(after.dropping);
  }, [recompare]);

  const cancel = useCallback(() => {
    const flow = saveFlowStore.getState();
    const before = flow.state;
    flow.dispatch({ type: 'CANCEL' });
    if (before.kind === 'recomparing' && before.jobId !== null) void cancelJob().catch(() => undefined);
  }, [cancelJob]);

  const dismiss = useCallback(() => {
    saveFlowStore.getState().dispatch({ type: 'DISMISS' });
  }, []);

  return { state, error, saveClicked, confirmDiscard, cancel, dismiss };
}
