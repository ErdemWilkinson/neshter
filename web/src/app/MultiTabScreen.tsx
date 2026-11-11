import { useT } from '@/i18n';

/** Shown instead of the app when another tab already holds the `neshter-ui` lock; nothing else is started. */
export function MultiTabScreen() {
  const t = useT();
  return (
    <main className="stale-screen" role="alert">
      <h1>{t('multitab.title')}</h1>
      <p>{t('multitab.body')}</p>
    </main>
  );
}
