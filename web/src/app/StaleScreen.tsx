import { useDraftHeld } from '@/features/draft/useDraftHeld';
import { useT } from '@/i18n';

/**
 * The one screen that replaces the app when the session is over: `restarted` (the token no longer fits) or `down`
 * (the server does not answer). `hasDraft`: unsaved cuts exist, so their draft mirror is worth mentioning.
 */
export function StaleScreen({ kind, hasDraft }: { kind: 'restarted' | 'down'; hasDraft: boolean }) {
  const t = useT();
  const held = useDraftHeld(); // an earlier draft holds autosave back: this session's cuts are not in it
  const down = !hasDraft ? t('app.stale.downNoDraft') : held ? t('app.stale.down.held') : t('app.stale.down');
  const title = kind === 'restarted' ? t('app.stale.restarted') : down;
  return (
    <main className="stale-screen" role="alert">
      <h1>{title}</h1>
      {kind === 'restarted' && <p>{t('app.stale.reopen')}</p>}
      {kind === 'restarted' && hasDraft && <p>{t(held ? 'app.stale.restoreHint.held' : 'app.stale.restoreHint')}</p>}
    </main>
  );
}
