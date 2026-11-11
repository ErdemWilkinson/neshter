import { useEffect, useState } from 'react';
import { etaSeconds } from '@/api/eta';
import type { JobState } from '@/api/schemas';
import { t, useT } from '@/i18n';
import { Button } from './Button';

function duration(totalSeconds: number): string {
  const seconds = Math.max(0, Math.round(totalSeconds));
  if (seconds < 60) return t('jobs.duration.sec', { s: seconds });
  return t('jobs.duration.min', { m: Math.floor(seconds / 60), s: seconds % 60 });
}

/** Progress bar, the server's message, elapsed time, and an ETA only once the job is 5 % in (never invented). */
export function ProgressCard({ job, onCancel }: { job: JobState; onCancel: () => void }) {
  useT();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  const started = job.started_at === null ? Number.NaN : Date.parse(job.started_at);
  const elapsedMs = Number.isNaN(started) ? null : Math.max(0, now - started);
  const eta = elapsedMs === null ? null : etaSeconds(job.progress, elapsedMs);
  const progress = Math.min(1, Math.max(0, job.progress));

  return (
    <div className="progress-card glass">
      <progress value={progress} max={1} aria-label={t('jobs.progress')} />
      <span className="num">{t('jobs.percent', { pct: Math.round(progress * 100) })}</span>
      <p className="progress-message">{job.message}</p>
      {elapsedMs !== null && <p className="num">{t('jobs.elapsed', { t: duration(elapsedMs / 1000) })}</p>}
      {eta !== null && <p className="num">{t('jobs.eta', { t: duration(eta) })}</p>}
      <Button onClick={onCancel}>{t('ui.cancel')}</Button>
    </div>
  );
}
