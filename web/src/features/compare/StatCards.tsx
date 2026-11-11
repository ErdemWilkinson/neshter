import type { SuiteResult } from '@/api/schemas';
import { formatStat } from '@/features/measure/format';
import { useT } from '@/i18n';
import { rateWithCi } from './format';

/** Target rate (original to cut, with Wilson intervals), KL on neutral prompts, perplexity, and the file-size note. */
export function StatCards({ result, target }: { result: SuiteResult; target: string | null }) {
  const t = useT();
  const { metrics } = result;
  const row = target === null ? undefined : result.categories.find((c) => c.category === target);
  const klParts = [
    metrics.kl_random === null ? null : t('compare.stat.klRandom', { kl: formatStat(metrics.kl_random) }),
    metrics.kl_random_p === null ? null : t('compare.stat.klP', { p: formatStat(metrics.kl_random_p) }),
    metrics.kl_percentile === null ? null : t('compare.stat.klPercentile', { pct: formatStat(metrics.kl_percentile) }),
  ].filter((part): part is string => part !== null);

  return (
    <div className="stat-cards">
      {row !== undefined && (
        <section className="stat-card glass">
          <h3>{t('compare.stat.target')}</h3>
          <p className="num">
            {t('compare.stat.change', {
              from: rateWithCi(row.k_orig, row.n, row.orig_ci),
              to: rateWithCi(row.k_int, row.n, row.int_ci),
            })}
          </p>
        </section>
      )}
      <section className="stat-card glass">
        <h3>{t('compare.stat.kl')}</h3>
        <p className="num">{formatStat(metrics.kl)}</p>
        {klParts.length > 0 && <p className="num">{klParts.join(' · ')}</p>}
        {metrics.kl_random_note !== null && <p className="stat-note">{metrics.kl_random_note}</p>}
      </section>
      <section className="stat-card glass">
        <h3>{t('compare.stat.ppl')}</h3>
        <p className="num">
          {t('compare.stat.change', { from: formatStat(metrics.perplexity_orig), to: formatStat(metrics.perplexity_int) })}
        </p>
      </section>
      <section className="stat-card glass">
        <h3>{t('compare.stat.sizeTitle')}</h3>
        <p>{t('compare.stat.size')}</p>
      </section>
    </div>
  );
}
