import { useCallback, useEffect, useRef, useState } from 'react';
import type { LoadRequest } from '@/api/client';
import { getApi } from '@/api/currentApi';
import { ApiError, isAbortError, type ErrorCode } from '@/api/errors';
import { sampleResult } from '@/api/schemas';
import { useJob } from '@/api/useJob';
import { navigate } from '@/app/router';
import { basketStore } from '@/core/stores/basket';
import { compareStore } from '@/core/stores/compare';
import { measureStore } from '@/core/stores/measure';
import { selectionStore } from '@/core/stores/selection';
import { sessionStore } from '@/core/stores/session';
import { intentRoute, uiStore } from '@/core/stores/ui';
import { exploreStore } from '@/features/map/exploreStore';
import { RecipeApply } from '@/features/recipe/RecipeApply';
import { saveFlowStore } from '@/features/saveflow/saveFlowStore';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';
import { ProgressCard } from '@/ui/ProgressCard';

/** Name of the sample model's session (the snapshot folder is only a hash). */
export const SAMPLE_NAME = 'Qwen/Qwen2.5-0.5B-Instruct';

export type LoadPlan = { kind: 'load'; request: LoadRequest } | { kind: 'sample' };

interface Failure {
  code: ErrorCode;
  detail: unknown;
}

const asFailure = (error: unknown): Failure =>
  error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) };

/** A different model brings a different basket: the old cuts stay in their own server draft, not on this model. */
function resetForNewModel(): void {
  basketStore.getState().replaceAll({ items: [], shelf: [], savedOps: [] });
  compareStore.setState({ lastPrompt: null, lastPair: null, lastSuite: null, refreshedDropped: null });
  selectionStore.getState().clear();
  measureStore.getState().reset(); // measurements and the explore view belong to one model
  exploreStore.getState().reset();
  // The save flow's result card and refreshed notice describe the old model.
  saveFlowStore.getState().dispatch({ type: 'CANCEL' });
  saveFlowStore.getState().dispatch({ type: 'DISMISS' });
}

/**
 * Runs the load (after the sample download, for the sample plan) with a progress card and `İptal`. When the job is
 * done the session model is read from `/api/session`, a pending recipe import is applied, and the person goes on to
 * the screen of their intent. A failed or cancelled job offers the way back to the list.
 */
export function LoadScreen({ plan, onBack }: { plan: LoadPlan; onBack: () => void }) {
  const t = useT();
  const { job, start, cancel } = useJob();
  const [failure, setFailure] = useState<Failure | null>(null);
  const [applying, setApplying] = useState(false);
  const started = useRef(false);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const run = useCallback(async () => {
    const api = getApi();
    try {
      let request: LoadRequest;
      if (plan.kind === 'sample') {
        const downloaded = await start(() => api.sample());
        if (downloaded.state === 'cancelled') return onBack();
        if (downloaded.state !== 'done') return setFailure({ code: downloaded.error_code ?? 'internal', detail: downloaded.error });
        const result = sampleResult.safeParse(downloaded.result);
        if (!result.success) return setFailure({ code: 'internal', detail: result.error.issues });
        request = { path: result.data.path, name: SAMPLE_NAME, dtype: 'auto' };
      } else {
        request = plan.request;
      }
      const loaded = await start(() => api.loadModel(request));
      if (loaded.state === 'cancelled') return onBack();
      if (loaded.state !== 'done') return setFailure({ code: loaded.error_code ?? 'internal', detail: loaded.error });
      const session = await api.session();
      if (!alive.current) return;
      if (!session.loaded) return setFailure({ code: 'internal', detail: null });
      const before = sessionStore.getState().model?.source_sha256 ?? null;
      sessionStore.getState().setModel(session); // autosave flushes the old model's cuts on this change
      if (before !== session.source_sha256) resetForNewModel();
      setApplying(true);
    } catch (error) {
      if (!isAbortError(error) && alive.current) setFailure(asFailure(error));
    }
  }, [plan, start, onBack]);

  useEffect(() => {
    if (started.current) return; // StrictMode runs effects twice; one load is enough
    started.current = true;
    void run();
  }, [run]);

  const finish = useCallback(() => navigate(intentRoute(uiStore.getState().intent)), []);

  return (
    <section className="load-screen" aria-busy={failure === null}>
      <h1>{failure !== null ? t('models.fail') : t('models.loading')}</h1>
      {failure !== null ? (
        <>
          {failure.code === 'not_found' && <p>{t('models.fail.notFound')}</p>}
          <ErrorCard code={failure.code} detail={failure.detail} />
          <Button onClick={onBack}>{t('models.back')}</Button>
        </>
      ) : applying ? (
        <RecipeApply onSettled={finish} />
      ) : job !== null ? (
        <ProgressCard job={job} onCancel={() => void cancel()} />
      ) : (
        <p>{t('app.loading')}</p>
      )}
    </section>
  );
}
