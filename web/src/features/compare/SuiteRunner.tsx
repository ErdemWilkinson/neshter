import { useEffect, useRef, useState } from 'react';
import { useStore } from 'zustand';
import type { JsonOp } from '@/api/client';
import { getApi } from '@/api/currentApi';
import { ApiError, BusyError, isAbortError, type ErrorCode } from '@/api/errors';
import { isTerminal } from '@/api/jobs';
import { suiteResult } from '@/api/schemas';
import { useJob } from '@/api/useJob';
import { opsFingerprint } from '@/core/fingerprint';
import type { KnownOp } from '@/core/ops';
import { compareStore } from '@/core/stores/compare';
import { sessionStore } from '@/core/stores/session';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';
import { ProgressCard } from '@/ui/ProgressCard';
import { Segmented } from '@/ui/Segmented';

type Speed = 'quick' | 'full';

/**
 * `Test setini koş`: the suite job over the sendable ops. Quick is the default on CPU. The result is stored with the
 * fingerprint of the ops it ran with, so a later basket change is recognisable as stale.
 */
export function SuiteRunner({
  ops,
  target,
  disabled,
}: {
  ops: KnownOp[];
  target: string | null;
  disabled: boolean;
}) {
  const t = useT();
  const device = useStore(sessionStore, (s) => s.model?.device ?? null);
  const [speed, setSpeed] = useState<Speed>(device === 'cpu' ? 'quick' : 'full');
  const [failure, setFailure] = useState<{ code: ErrorCode; detail: unknown } | null>(null);
  const { job, start, cancel, reset } = useJob<unknown>();
  const alive = useRef(true);
  const running = job !== null && !isTerminal(job);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const run = async () => {
    setFailure(null);
    const fp = opsFingerprint(ops);
    try {
      const final = await start(() =>
        getApi().suite({ recipe_ops: ops as JsonOp[], target, quick: speed === 'quick' }),
      );
      if (!alive.current) return;
      if (final.state === 'cancelled') return reset();
      if (final.state !== 'done') return setFailure({ code: final.error_code ?? 'internal', detail: final.error });
      const parsed = suiteResult.safeParse(final.result);
      if (!parsed.success) return setFailure({ code: 'internal', detail: parsed.error.issues });
      compareStore.getState().setLastSuite({ fp, result: parsed.data });
    } catch (error) {
      if (isAbortError(error) || !alive.current) return;
      if (error instanceof BusyError) sessionStore.getState().setBusy(true);
      else setFailure(error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) });
    }
  };

  return (
    <section className="suite-runner">
      <h2>{t('compare.suite.title')}</h2>
      <Segmented
        label={t('compare.suite.speed')}
        value={speed}
        onChange={setSpeed}
        options={[
          { value: 'quick', label: t('compare.suite.quick') },
          { value: 'full', label: t('compare.suite.full') },
        ]}
      />
      <Button variant="primary" disabled={disabled || running} onClick={() => void run()}>
        {t('compare.suite.run')}
      </Button>
      {running && <ProgressCard job={job} onCancel={() => void cancel()} />}
      {failure !== null && (
        <>
          <p>{t('compare.suite.failed')}</p>
          <ErrorCard code={failure.code} detail={failure.detail} />
        </>
      )}
    </section>
  );
}
