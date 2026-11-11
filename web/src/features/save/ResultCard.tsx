import type { ExportResult } from '@/api/schemas';
import { useT } from '@/i18n';

/** The finished export. Every value is shown as text (the path and name come from the user and the server). */
export function ResultCard({ result }: { result: ExportResult }) {
  const t = useT();
  return (
    <section className="result-card glass" aria-labelledby="result-title">
      <h2 id="result-title">{t('save.result.title')}</h2>
      {result.safety_ablated && <p className="badge badge-safety">{t('save.result.safety')}</p>}
      <dl>
        <dt>{t('save.result.path')}</dt>
        <dd className="num result-path">{result.path}</dd>
        <dt>{t('save.result.name')}</dt>
        <dd>{result.name}</dd>
        <dt>{t('save.result.sha')}</dt>
        <dd className="num result-sha">{result.source_sha256}</dd>
      </dl>
      {result.source_untouched && <p>{t('save.result.untouched')}</p>}
    </section>
  );
}
