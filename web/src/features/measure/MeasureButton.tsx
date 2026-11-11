import { useEffect, useId, useState } from 'react';
import { useStore } from 'zustand';
import { etaSeconds } from '@/api/eta';
import { getApi } from '@/api/currentApi';
import { ApiError, BusyError, type ErrorCode } from '@/api/errors';
import type { EffectResult } from '@/api/schemas';
import { useJob } from '@/api/useJob';
import { opsFingerprint } from '@/core/fingerprint';
import { fromEffectResult } from '@/core/measurement';
import type { HeadOp } from '@/core/ops';
import { isLocked, jobsStore } from '@/core/stores/jobs';
import { measureStore } from '@/core/stores/measure';
import { sessionStore } from '@/core/stores/session';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { useProgressShown } from '@/ui/useProgressShown';
import { ErrorCard } from '@/ui/ErrorCard';
import { duration } from '@/ui/ProgressCard';
import { formatStat } from './format';
import { clampNMax, nMaxBounds, smallestP } from './nMax';

interface Failure {
  code: ErrorCode;
  detail: unknown;
}

const asFailure = (error: unknown): Failure =>
  error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) };

function Progress({ message, progress, startedAt }: { message: string; progress: number; startedAt: string | null }) {
  useProgressShown();
  const t = useT();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const started = startedAt === null ? Number.NaN : Date.parse(startedAt);
  const elapsedMs = Number.isNaN(started) ? null : Math.max(0, now - started);
  // Early stopping can end the run sooner, so the extrapolation is only ever an upper bound ("en çok").
  const eta = elapsedMs === null ? null : etaSeconds(progress, elapsedMs);
  return (
    <div className="measure-progress" role="status">
      <progress value={Math.min(1, Math.max(0, progress))} max={1} aria-label={t('jobs.progress')} />
      <p>{message === '' ? t('measure.running') : message}</p>
      {elapsedMs !== null && <p className="num">{t('jobs.elapsed', { t: duration(elapsedMs / 1000) })}</p>}
      {eta !== null && <p className="num">{t('measure.eta', { t: duration(eta) })}</p>}
    </div>
  );
}

/** Spec §2 step 3a: "Gerçek etkiyi ölç". The result lands in `measureStore`, keyed by the ops fingerprint. */
export function MeasureButton({ ops }: { ops: readonly HeadOp[] }) {
  const t = useT();
  const inputId = useId();
  const device = useStore(sessionStore, (s) => s.model?.device ?? 'cpu');
  const locked = useStore(jobsStore, isLocked);
  const { job, start, cancel } = useJob<EffectResult>();
  const [failure, setFailure] = useState<Failure | null>(null);
  const bounds = nMaxBounds(device);
  const [text, setText] = useState(String(bounds.default));
  const nMax = clampNMax(text, bounds);
  const running = job !== null && (job.state === 'queued' || job.state === 'running');

  const measure = async () => {
    setFailure(null);
    const fp = opsFingerprint(ops);
    try {
      const final = await start(() =>
        getApi().effect({ heads: ops.map((op): [number, number] => [op.layer, op.head]), n_max: nMax, seed: 0 }),
      );
      if (final.state === 'error') {
        setFailure({ code: final.error_code ?? 'internal', detail: final.error });
      } else if (final.state === 'done' && final.result !== null) {
        try {
          const m = fromEffectResult(final.result, fp, new Date().toISOString());
          measureStore.getState().put(m);
          const model = sessionStore.getState().model;
          if (model !== null) {
            sessionStore.getState().setModel({ ...model, measure_count: Math.max(model.measure_count, m.session_index) });
          }
        } catch (error) {
          setFailure({ code: 'internal', detail: String(error) }); // a malformed payload is never shown as a measurement
        }
      }
    } catch (error) {
      if (error instanceof BusyError) sessionStore.getState().setBusy(true);
      else setFailure(asFailure(error));
    }
  };

  return (
    <div className="measure">
      <Button variant="primary" onClick={() => void measure()} disabled={ops.length === 0 || locked || running}>
        {t('measure.button')}
      </Button>
      <details className="measure-advanced">
        <summary>{t('measure.advanced')}</summary>
        <label htmlFor={inputId}>{t('measure.nmax')}</label>
        <input
          id={inputId}
          type="number"
          inputMode="numeric"
          min={bounds.min}
          max={bounds.max}
          step={1}
          value={text}
          disabled={running}
          onChange={(event) => setText(event.target.value)}
          onBlur={() => setText(String(nMax))}
        />
        <p className="num">{t('measure.pmin', { p_min: formatStat(smallestP(nMax)) })}</p>
      </details>
      {running && job !== null && (
        <>
          <Progress message={job.message} progress={job.progress} startedAt={job.started_at} />
          <Button onClick={() => void cancel()}>{t('ui.cancel')}</Button>
        </>
      )}
      {failure !== null && (
        <>
          <p>{t('measure.failed')}</p>
          <ErrorCard code={failure.code} detail={failure.detail} />
        </>
      )}
    </div>
  );
}
