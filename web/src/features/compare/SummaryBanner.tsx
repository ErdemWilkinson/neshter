import { useStore } from 'zustand';
import type { SuiteResult } from '@/api/schemas';
import { uiStore } from '@/core/stores/ui';
import { useT } from '@/i18n';

/** The server's one-sentence verdict (by language), or the fallback when the concept has no test category. */
export function SummaryBanner({ summary }: { summary: SuiteResult['summary'] }) {
  const t = useT();
  const lang = useStore(uiStore, (s) => s.lang);
  if (summary === null) {
    return (
      <p className="summary-banner summary-none" role="status">
        {t('compare.summary.none')}
      </p>
    );
  }
  return (
    <p className={`summary-banner summary-${summary.level}`} role="status">
      {lang === 'tr' ? summary.tr : summary.en}
    </p>
  );
}
