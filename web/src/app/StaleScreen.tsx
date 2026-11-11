import { useT } from '@/i18n';

/**
 * The one screen that replaces the app when the session is over: `restarted` (the token no longer fits) or `down`
 * (the server does not answer). `hasDraft`: unsaved cuts exist, so their draft mirror is worth mentioning.
 */
export function StaleScreen({ kind, hasDraft }: { kind: 'restarted' | 'down'; hasDraft: boolean }) {
  const t = useT();
  const title =
    kind === 'restarted' ? t('app.stale.restarted') : hasDraft ? t('app.stale.down') : t('app.stale.downNoDraft');
  return (
    <main className="stale-screen" role="alert">
      <h1>{title}</h1>
      {kind === 'restarted' && <p>{t('app.stale.reopen')}</p>}
      {kind === 'restarted' && hasDraft && <p>{t('app.stale.restoreHint')}</p>}
    </main>
  );
}
