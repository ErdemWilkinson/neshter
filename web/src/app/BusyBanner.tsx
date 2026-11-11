import { useEffect } from 'react';
import { useStore } from 'zustand';
import { getApi } from '@/api/currentApi';
import { isLocked, jobsStore } from '@/core/stores/jobs';
import { sessionStore } from '@/core/stores/session';
import { useT } from '@/i18n';

const RECHECK_MS = 3000;

/**
 * "Model meşgul" while a job holds the model: our own (`jobsStore`) or one a 409 told us about (`session.busy`).
 * A 409 flag has no job of ours to clear it, so while it is set the server's job list is polled until none runs.
 */
export function BusyBanner() {
  const t = useT();
  const flagged = useStore(sessionStore, (s) => s.busy);
  const locked = useStore(jobsStore, isLocked);

  useEffect(() => {
    if (!flagged) return;
    let stopped = false;
    const timer = setInterval(() => {
      getApi()
        .jobs()
        .then((jobs) => {
          const running = jobs.some((job) => job.state === 'queued' || job.state === 'running');
          if (!stopped && !running) sessionStore.getState().setBusy(false);
        })
        .catch(() => undefined); // unreachable server: the heartbeat reports it
    }, RECHECK_MS);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [flagged]);

  if (!flagged && !locked) return null;
  return (
    <div className="busy-banner" role="status">
      {t('app.busy')}
    </div>
  );
}
